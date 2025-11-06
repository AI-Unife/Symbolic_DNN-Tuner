from __future__ import annotations

import argparse
import datetime
import time
import sys
import os
import copy
import shutil
from typing import List

from skopt import gp_minimize, load
from skopt.callbacks import CheckpointSaver
from skopt.space import Real, Integer, Categorical, Space
from tensorflow.keras import backend as K

from components.colors import colors
from components.controller import controller
from components.dataset import get_datasets
from components.search_space import search_space
from components.random_search import RandomSearch

from pathlib import Path
from exp_config import create_config_file, set_active_config, load_cfg
import tensorflow as tf

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
        os.makedirs(cfg.name, exist_ok=True)
        for folder in required_dirs:
            os.makedirs(f"{cfg.name}/{folder}", exist_ok=True)
    except OSError as e:
        print(colors.FAIL, f"Failed to create folder: {e}", colors.ENDC)
        sys.exit(1)


def copy_symbolic_files() -> None:
    """
    Copy the base symbolic files into this experiment's symbolic folder.
    """
    src = "./symbolic_base"
    dst = f"{cfg.name}/symbolic"
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

        log_path = f"{cfg.name}/algorithm_logs/hyper-neural.txt"
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

class ConstraintsWrapper:
    def __init__(self, space: Space):
        self.space = space

    def apply_constraints(self, params: List) -> bool:
        """
        Check whether a sampled parameter vector fits within the (possibly updated) space.

        When running RS variants we skip constraints (returns True).
        """
        if "RS" in cfg.opt:
            return True

        for i, dim in enumerate(self.space.dimensions):
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
    ckpt_path = f"{cfg.name}/checkpoints/checkpoint.pkl"
    callback = None # CheckpointSaver(ckpt_path, compress=9)

    obj_fn = ObjectiveWrapper(search_space, controller)
    const_fn = ConstraintsWrapper(search_space)
    no_rules = ["RS", "standard"]
    with_rules = ["filtered", "RS_ruled", "basic"]
    use_filter = (cfg.opt == "filtered")
    # Initialize the chosen optimizer for the very first evaluation
    if "RS" in cfg.opt:
        random_search = RandomSearch(random_state=cfg.seed, total_iter=max_iter)
        res = random_search(obj_fn.objective, search_space,
                            callback=callback
                            )
    else:
        res = gp_minimize(
            obj_fn.objective,
            search_space,
            acq_func="EI",
            random_state=cfg.seed,
            n_calls=1,
            n_random_starts=1,
            callback=callback,
            space_constraint=const_fn.apply_constraints if use_filter else None
        )

    # Decide initial new_space (rule-driven or fixed)
    new_space = copy.deepcopy(search_space) if cfg.opt in no_rules else controller.diagnosis()

    while controller.iter <= max_iter and not controller.convergence:
        print(colors.MAGENTA, f"--- ITERATION {controller.iter} ---", colors.ENDC)

        # If the space shape didn't change, we can warm-start BO with previous data
        if len(new_space.dimensions) == len(search_space.dimensions):
            if not use_filter:
                # Re-inject past points that satisfy current constraints (avoid duplicates)
                search_space = copy.deepcopy(new_space)

            x0, y0 = res.x_iters, res.func_vals
            obj_fn = ObjectiveWrapper(search_space, controller)
            const_fn = ConstraintsWrapper(new_space)

            try:
                if "RS" in cfg.opt:
                    # Random search: keep drawing one more sample/eval
                    res = random_search(obj_fn.objective, search_space,
                                        callback=callback
                                        )
                else:
                    # BO: one more call using warm-start data
                    res = gp_minimize(
                        obj_fn.objective,
                        search_space,
                        x0=list(x0),
                        y0=list(y0),
                        acq_func="EI",
                        n_calls=1,
                        n_random_starts=0,
                        random_state=cfg.seed,
                        callback=callback,
                        space_constraint=const_fn.apply_constraints if use_filter else None
                    )
            except Exception as e:
                print(colors.FAIL, f"Optimization error: {e}", colors.ENDC)
                if "RS" in cfg.opt:
                    res = random_search(obj_fn.objective, search_space,
                                        callback=callback
                                        )
                else:
                    res = gp_minimize(
                        obj_fn.objective,
                        search_space,
                        acq_func="EI",
                        n_calls=1,
                        n_random_starts=1,
                        random_state=cfg.seed,
                        callback=callback,
                        space_constraint=const_fn.apply_constraints if use_filter else None
                    )


        else:
            # Space changed (structure or bounds) -> restart the optimizer on the new space
            print(colors.WARNING, "Search space changed. Restarting BO...", colors.ENDC)
            search_space = copy.deepcopy(new_space)
            obj_fn = ObjectiveWrapper(search_space, controller)
            const_fn = ConstraintsWrapper(new_space)
            if "RS" in cfg.opt:
                # Reset RS history so it doesn't bias sampling with stale configs
                random_search.Xi, random_search.Yi = [], []
                res = random_search(obj_fn.objective, search_space,
                                    callback=callback
                                    )
            else:
                res = gp_minimize(
                    obj_fn.objective,
                    search_space,
                    acq_func="EI",
                    n_calls=1,
                    n_random_starts=1,
                    random_state=cfg.seed,
                    callback=callback,
                    space_constraint=const_fn.apply_constraints if use_filter else None
                )

            if cfg.opt in with_rules:
                all_x, all_y = list(res.x_iters), list(res.func_vals)

        # Ask controller again: either keep the same space (no_rules) or update by diagnosis (with_rules)
        new_space = copy.deepcopy(search_space) if cfg.opt in no_rules else controller.diagnosis()

    return res

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Symbolic DNN Tuner configuration.

    Returns:
        argparse.Namespace containing all configuration values.
    """
    parser = argparse.ArgumentParser(
        description="Symbolic DNN Tuner Configuration"
    )

    parser.add_argument("--eval", type=int, default=300,
                        help="Max number of evaluations")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epochs for training")
    parser.add_argument(
        "--mod_list", nargs="+", default=[],
        help="List of active modules (e.g., hardware_module flops_module)"
    )
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name")
    parser.add_argument("--name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--frames", type=int, default=16,
                        help="Number of frames for gesture dataset")
    parser.add_argument("--mode", type=str, default="fwdPass",
                        choices=["fwdPass", "depth", "hybrid"],
                        help="Experiment mode (fwdPass, depth, hybrid)")
    parser.add_argument("--channels", type=int, default=2,
                        help="Number of channels for the dataset")
    parser.add_argument("--polarity", type=str, default="both",
                        choices=["both", "sum", "sub", "drop"],
                        help="Polarity for event-based datasets")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument(
        "--opt", type=str, default="filtered",
        choices=["standard", "filtered", "basic", "RS", "RS_ruled"],
        help="Optimizer type for the analysis"
    )

    args = parser.parse_args()

    # Validate module list
    valid_modules = {"hardware_module", "flops_module"}
    for mod in args.mod_list:
        if mod not in valid_modules:
            parser.error(
                f"Invalid module '{mod}'. Choose from: {', '.join(valid_modules)}"
            )

    return args

# ---------------------------------- main -------------------------------------

if __name__ == "__main__":
    # Optional cosmetic for Windows terminals (enables ANSI colors)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    os.system("")

    # TensorFlow performance/logging toggles
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    args = parse_args()
    exp_dir = Path(args.name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = create_config_file(exp_dir, overrides=args.__dict__)

    set_active_config(cfg_path)
    cfg = load_cfg(force=True)

    create_experiment_folders()
    copy_symbolic_files()

    # Load dataset by normalized key (e.g., "imagenet16-120" -> "imagenet16120")
    X_train, Y_train, X_test, Y_test, n_classes = get_datasets(cfg.dataset.strip().lower().replace("-", ""))


    # Base search space
    # Controller orchestrates training & symbolic tuning
    ctrl = controller(X_train, Y_train, X_test, Y_test, n_classes)
    # if cfg.opt in ["RS", "standard"]:
    #     print(colors.OKBLUE, "Running optimization WITHOUT tuning rules.", colors.ENDC)
    #     print(colors.OKBLUE, "add all possible dimensions at max bounds.", colors.ENDC)
    #     for i, dim in enumerate(ctrl.space.dimensions):
    #         if 'new_conv' in dim.name:
    #             ctrl.space = ctrl.ss.add_params({dim.name: 16})
    #         if 'new_fc' in dim.name:
    #             ctrl.space = ctrl.ss.add_params({dim.name: 32})
    #         if dim.name in ['reg_l2', 'data_augmentation', 'skip_connection']:
    #             new_categories = [True, False]
    #             ctrl.space.dimensions[i] = Categorical(new_categories, name=dim.name)
    first_space = ctrl.space
    print(colors.MAGENTA, "|  ----------- SEARCH SPACE ----------  |\n", colors.ENDC)
    print(first_space)


    start_time = time.time()
    print(colors.OKGREEN, "\nSTARTING ALGORITHM \n", colors.ENDC)
    res = run_optimization(first_space, ctrl, cfg.eval)
    print(colors.OKGREEN, "\nALGORITHM FINISHED \n", colors.ENDC)
    print(colors.CYAN, "\nTOTAL TIME --------> \n", time.time() - start_time, colors.ENDC)
