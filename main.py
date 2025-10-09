from __future__ import annotations

import argparse
import datetime
import time
import sys
import os
import copy
import shutil
from typing import List, Tuple, Optional

import numpy as np
from skopt import gp_minimize, load
from skopt.callbacks import CheckpointSaver
from skopt.space import Real, Integer, Categorical, Space
from tensorflow.keras import backend as K

from components.colors import colors
from components.controller import controller
from components.dataset import get_datasets
from components.search_space import search_space
from components.random_search import RandomSearch

import config as cfg


# --------------------------- pretty printing ---------------------------------

def print_experiment_config() -> None:
    """
    Print a summary of the user-provided configuration.
    """
    print(colors.MAGENTA, "|  ----------- USER PARAMS CONFIGURATION ----------  |\n", colors.ENDC)
    print("EXPERIMENT NAME: ", cfg.NAME_EXP)
    print("OPTIMIZATION METHOD: ", cfg.OPT, flush=True)
    print("DATASET NAME: ", cfg.DATA_NAME)
    print("MAX NET EVAL: ", cfg.MAX_EVAL)
    print("EPOCHS FOR TRAINING: ", cfg.EPOCHS)
    print("MODULE LIST: ", cfg.MOD_LIST, flush=True)
    if cfg.DATA_NAME == "gesture":
        print("MODE: ", cfg.MODE, flush=True)
        print("POLARITY: ", cfg.POLARITY, flush=True)
        print("CHANNELS: ", cfg.NUM_CHANNELS, flush=True)
    print("SEED: ", cfg.SEED, flush=True)


# ------------------------------ filesystem -----------------------------------

def create_experiment_folders() -> None:
    """
    Ensure the experiment directory structure exists.
    """
    required_dirs = [
        "Model", "Weights", "database", "checkpoints", "log_folder",
        "algorithm_logs", "dashboard/model", "symbolic"
    ]
    try:
        os.makedirs(cfg.NAME_EXP, exist_ok=True)
        for folder in required_dirs:
            os.makedirs(f"{cfg.NAME_EXP}/{folder}", exist_ok=True)
    except OSError as e:
        print(colors.FAIL, f"Failed to create folder: {e}", colors.ENDC)
        sys.exit(1)


def copy_symbolic_files() -> None:
    """
    Copy the base symbolic files into this experiment's symbolic folder.
    """
    src = "./symbolic_base"
    dst = f"{cfg.NAME_EXP}/symbolic"
    if not os.path.isdir(src):
        print(colors.WARNING, f"Symbolic base folder not found at '{src}'. Skipping copy.", colors.ENDC)
        return
    try:
        # Copy only files (flat copy). We avoid removing user-edited files in dst.
        for name in os.listdir(src):
            src_path = os.path.join(src, name)
            dst_path = os.path.join(dst, name)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
    except OSError as e:
        print(colors.FAIL, f"Failed to copy symbolic directory: {e}", colors.ENDC)
        sys.exit(1)


# ------------------------------ objective ------------------------------------

class ObjectiveWrapper:
    """
    Helper to convert a sampled vector from skopt Space to a `dict` for the controller.
    """
    def __init__(self, space: Space, controller: controller):
        self.search_space = space
        self.controller = controller

    def objective(self, params: List) -> float:
        """
        Convert vector `params` to {name: value}, log it, train once, and return the score.
        """
        space_dict = {dim.name: val for dim, val in zip(self.search_space.dimensions, params)}
        print("Chosen point:", space_dict)
        print("Actual search space:")
        print_space(self.search_space)

        log_path = f"{cfg.NAME_EXP}/algorithm_logs/hyper-neural.txt"
        try:
            with open(log_path, "a") as f:
                f.write(str(space_dict) + "\n")
        except OSError:
            pass

        score = self.controller.training(space_dict)
        # Clear session to avoid graph accumulation across BO iterations
        K.clear_session()
        print("Score:", score, flush=True)
        return float(score)


# ------------------------------- helpers -------------------------------------

def print_space(space: Space) -> None:
    """Pretty print the current Space dimensions with names and bounds."""
    for i, dim in enumerate(space.dimensions):
        print(f"Dimension {i}: {dim.name} - {dim}")


def print_diff(old_space: Space, new_space: Space) -> None:
    """Show differences between two Spaces dimension-by-dimension."""
    print(colors.WARNING, "Differences in search space:", colors.ENDC)
    for i, (old_dim, new_dim) in enumerate(zip(old_space.dimensions, new_space.dimensions)):
        if old_dim != new_dim:
            print(colors.CYAN, f"Dimension {i}: {old_dim.name} changed from {old_dim} to {new_dim}", colors.ENDC)
        else:
            print(colors.FAIL, f"Dimension {i}: {old_dim.name} unchanged", colors.ENDC)


def apply_constraints(space: Space, params: List) -> bool:
    """
    Check whether a sampled parameter vector fits within the (possibly updated) space.

    When running RS variants we skip constraints (returns True).
    """
    if "RS" in cfg.OPT:
        return True

    for i, dim in enumerate(space.dimensions):
        val = params[i]
        if isinstance(dim, Categorical):
            if val not in dim.categories:
                return False
        elif isinstance(dim, (Integer, Real)):
            if not (dim.low <= val <= dim.high):
                return False
        else:
            print(f"Type space dimension {dim} - {type(dim)} not valid")
            sys.exit(1)
    return True


# ------------------------------ optimization ---------------------------------

def run_optimization(search_space: Space, controller: controller, max_iter: int):
    """
    Outer optimization loop:
      - initialize (RS or BO),
      - iteratively let the controller diagnose and (optionally) mutate the space,
      - reuse past evaluations when the space is unchanged,
      - checkpoint after each evaluation.
    """
    all_x, all_y = [], []
    ckpt_path = f"{cfg.NAME_EXP}/checkpoints/checkpoint.pkl"
    callback = None # CheckpointSaver(ckpt_path, compress=9)

    obj_fn = ObjectiveWrapper(search_space, controller)
    no_rules = ["RS", "standard"]
    with_rules = ["filtered", "RS_ruled", "basic"]
    use_filter = (cfg.OPT == "filtered") 

    # Initialize the chosen optimizer for the very first evaluation
    if "RS" in cfg.OPT:
        random_search = RandomSearch(random_state=cfg.SEED, total_iter=max_iter)
        res = random_search(obj_fn.objective, search_space,
                            # callback=[callback]
                            )
    else:
        res = gp_minimize(
            obj_fn.objective,
            search_space,
            acq_func="EI",
            random_state=cfg.SEED,
            n_calls=1,
            n_random_starts=1,
            # callback=[callback],
        )

    # Accumulate results
    all_x.extend(res.x_iters)
    all_y.extend(res.func_vals)

    # Decide initial new_space (rule-driven or fixed)
    new_space = copy.deepcopy(search_space) if cfg.OPT in no_rules else controller.diagnosis()[0]

    it = 0
    while it < max_iter and not controller.convergence:
        print(colors.MAGENTA, f"--- ITERATION {it + 1} ---", colors.ENDC)
        it += 1

        # Reload last checkpointed result to keep skopt internal state consistent
        # try:
        #     res = load(ckpt_path)
        # except Exception:
        #     # If the checkpoint does not exist yet (or is corrupted), we proceed with current `res`
        #     pass

        # If the space shape didn't change, we can warm-start BO with previous data
        if len(new_space.dimensions) == len(search_space.dimensions):
            if use_filter:
                # Re-inject past points that satisfy current constraints (avoid duplicates)
                for x, y in zip(all_x, all_y):
                    if apply_constraints(new_space, x) and x not in res.x_iters:
                        res.x_iters.append(list(x))
                        res.func_vals = np.append(res.func_vals, y)

            x0, y0 = res.x_iters, res.func_vals
            obj_fn = ObjectiveWrapper(new_space, controller)

            try:
                if "RS" in cfg.OPT:
                    # Random search: keep drawing one more sample/eval
                    res = random_search(obj_fn.objective, search_space,
                                        # callback=[callback]
                                        )
                else:
                    # BO: one more call using warm-start data
                    res = gp_minimize(
                        obj_fn.objective,
                        new_space,
                        x0=list(x0),
                        y0=list(y0),
                        acq_func="EI",
                        n_calls=1,
                        n_random_starts=0,
                        random_state=cfg.SEED,
                        # callback=[callback],
                    )
            except Exception as e:
                print(colors.FAIL, f"Optimization error: {e}", colors.ENDC)
                if "RS" in cfg.OPT:
                    res = random_search(obj_fn.objective, search_space,
                                        # callback=[callback]
                                        )
                else:
                    res = gp_minimize(
                        obj_fn.objective,
                        new_space,
                        acq_func="EI",
                        n_calls=1,
                        n_random_starts=1,
                        random_state=cfg.SEED,
                        # callback=[callback],
                    )

            if cfg.OPT in with_rules:
                # Accumulate unique x's and all y's
                all_x.extend(x for x in res.x_iters if x not in all_x)
                all_y.extend(res.func_vals)

        else:
            # Space changed (structure or bounds) -> restart the optimizer on the new space
            search_space = copy.deepcopy(new_space)
            print(colors.WARNING, "Search space changed. Restarting BO...", colors.ENDC)

            obj_fn = ObjectiveWrapper(search_space, controller)
            if "RS" in cfg.OPT:
                # Reset RS history so it doesn't bias sampling with stale configs
                random_search.Xi, random_search.Yi = [], []
                res = random_search(obj_fn.objective, search_space,
                                    # callback=[callback]
                                    )
            else:
                res = gp_minimize(
                    obj_fn.objective,
                    new_space,
                    acq_func="EI",
                    n_calls=1,
                    n_random_starts=1,
                    random_state=cfg.SEED,
                    # callback=[callback],
                )

            if cfg.OPT in with_rules:
                all_x, all_y = list(res.x_iters), list(res.func_vals)

        # Ask controller again: either keep the same space (no_rules) or update by diagnosis (with_rules)
        new_space = copy.deepcopy(search_space) if cfg.OPT in no_rules else controller.diagnosis()[0]

    return res


# ---------------------------------- main -------------------------------------

if __name__ == "__main__":
    # Optional cosmetic for Windows terminals (enables ANSI colors)
    os.system("")

    # TensorFlow performance/logging toggles
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    print_experiment_config()
    create_experiment_folders()
    copy_symbolic_files()

    # Load dataset by normalized key (e.g., "imagenet16-120" -> "imagenet16120")
    X_train, Y_train, X_test, Y_test, n_classes = get_datasets(cfg.DATA_NAME.strip().lower().replace("-", ""))

    # Base search space
    sp = search_space()
    first_space = sp.search_sp()
    print(colors.MAGENTA, "|  ----------- SEARCH SPACE ----------  |\n", colors.ENDC)
    print(first_space)

    # Controller orchestrates training & symbolic tuning
    ctrl = controller(X_train, Y_train, X_test, Y_test, n_classes)

    start_time = time.time()
    print(colors.OKGREEN, "\nSTARTING ALGORITHM \n", colors.ENDC)
    res = run_optimization(first_space, ctrl, cfg.MAX_EVAL)
    print(colors.OKGREEN, "\nALGORITHM FINISHED \n", colors.ENDC)
    print(colors.CYAN, "\nTOTAL TIME --------> \n", time.time() - start_time, colors.ENDC)

    # Visual diagnostics & persistence
    ctrl.plotting_obj_function()
    ctrl.save_experience()