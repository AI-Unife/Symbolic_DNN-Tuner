# [Imports remain the same]
from __future__ import annotations
import argparse
import datetime
import time
import sys
import os
import copy
import shutil
from typing import List
import ast
from pathlib import Path # Used for modern path management

from skopt import load
from skopt.callbacks import CheckpointSaver
from skopt.space import Real, Integer, Categorical, Space
from tensorflow.keras import backend as K
import tensorflow as tf

from components.colors import colors
from components.controller import controller
from components.dataset import get_datasets
from components.search_space import search_space
from components.random_search import RandomSearch
from exp_config import create_config_file, set_active_config, load_cfg
# Import the correct gp_minimize will be handled dynamically in main
# (This avoids a top-level import issue if the component isn't needed)


# ------------------------------ objective ------------------------------------

class ObjectiveWrapper:
    """
    Helper to convert a sampled vector from skopt Space to a `dict` for the controller.
    """
    def __init__(self, space: Space, ctrl: controller):
        self.search_space = space
        self.controller = ctrl

    def objective(self, params: List) -> float:
        """
        Convert vector `params` to {name: value}, log it, train once, and return the score.
        """
        space_dict = {dim.name: val for dim, val in zip(self.search_space.dimensions, params)}
        print("Chosen point:", space_dict)

        log_path = Path(cfg.name) / "algorithm_logs" / "hyper-neural.txt"
        try:
            # 'a' (append mode) is correct for logs
            with open(log_path, "a") as f:
                f.write(str(space_dict) + "\n")
        except OSError as e:
            ### IMPROVEMENT: Don't silently 'pass'. Log the error. ###
            print(colors.WARNING, f"Could not write to log file {log_path}: {e}", colors.ENDC)

        score = self.controller.training(space_dict)
        K.clear_session()
        print("Score:", score, flush=True)
        return float(score)


# ------------------------------- helpers -------------------------------------

def print_space(space: Space) -> None:
    """Pretty print the current Space dimensions with names and bounds."""
    for i, dim in enumerate(space.dimensions):
        if isinstance(dim, Categorical):
            print(f"Dimension {i}: {dim.name} - categories: {dim.categories}")
        else:
            if dim.high != 0:
                print(f"Dimension {i}: {dim.name} - {dim}")

class ConstraintsWrapper:
    def __init__(self, space: Space):
        self.space = space

    def apply_constraints(self, params: List) -> bool:
        # [Function remains the same, except for the fix below]
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
                ### IMPROVEMENT: Do not sys.exit(1) from a helper class. ###
                raise TypeError(f"Type space dimension {dim} - {type(dim)} not valid")
        return True


# ----------------- optimization helpers (new) --------------------------

### NEW HELPER FUNCTION ###
def _load_previous_results(cfg_name: str, base_space: Space) -> (list, list):
    """
    Loads x0 (parameters) and y0 (scores) from previous log files.
    Handles missing files and mismatched lengths.
    """
    x0, y0 = [], []
    param_log_path = Path(cfg_name) / "algorithm_logs" / "hyper-neural.txt"
    acc_log_path = Path(cfg_name) / "algorithm_logs" / "acc_report.txt"

    # --- Load parameters (x0) ---
    try:
        with open(param_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data_dict = ast.literal_eval(line)
                    if isinstance(data_dict, dict):
                        x0.append(list(data_dict.values()))
                        # print(colors.OKBLUE, f"\nLoaded {x0[-1]}.", colors.ENDC)
                        # print(colors.OKBLUE, f"point in space? {x0[-1] in base_space}.", colors.ENDC)
                except (ValueError, SyntaxError) as e:
                    print(colors.WARNING, f"Skipping malformed line in {param_log_path.name}: {e}", colors.ENDC)
    except FileNotFoundError:
        print(colors.WARNING, f"{param_log_path.name} not found. Assuming no prior data.", colors.ENDC)
        return [], [] # No points, so no scores

    # --- Load scores (y0) ---
    try:
        # Costruiamo il percorso per il file dei flops basandoci su quello dell'accuracy
        # Assumiamo che acc_log_path sia un oggetto Path (pathlib), dato che usi .name
        flops_log_path = Path(cfg_name) / "flops_report.txt"
        
        with open(acc_log_path, "r") as f_acc, open(flops_log_path, "r") as f_flops:
            # Leggiamo tutte le righe pulite dai due file
            acc_lines = [line.strip() for line in f_acc if line.strip()]
            flops_lines = [line.strip() for line in f_flops if line.strip()]

            # Iteriamo accoppiando le righe (zip si ferma alla lista più corta)
            for acc_line, flops_line in zip(acc_lines, flops_lines):
                try:
                    # Gestione caso "None" o stringa vuota per l'accuracy
                    if acc_line == "None" or not acc_line:
                        score = 100000.0
                    else:
                        accuracy = float(acc_line)
                        
                        # Parse della riga flops: prendiamo la seconda colonna
                        # Esempio riga: "Net_0 123456" -> parts[1] è 123456
                        flops_parts = flops_line.split()
                        if len(flops_parts) < 2:
                            print(colors.WARNING, f"Invalid format in flops file: {flops_line}", colors.ENDC)
                            continue
                            
                        n_flops = float(flops_parts[1])
                        
                        # Calcolo dello score secondo la formula richiesta
                        # Score = -0.77*accuracy - 0.33*(1 - NFlops/150000000)
                        score = -0.77 * accuracy - 0.33 * (1 - n_flops / 150000000.0)

                    y0.append(score)

                except ValueError:
                    print(colors.WARNING, f"Value Error processing: Acc='{acc_line}', Flops='{flops_line}'", colors.ENDC)

    except FileNotFoundError as e:
        print(colors.WARNING, f"File not found: {e}. Assuming no prior data.", colors.ENDC)
        return [], [] # No scores, so no points

    # --- Synchronize x0 and y0 ---
    min_len = min(len(x0), len(y0))
    if len(x0) != len(y0):
        print(colors.WARNING, f"Log file mismatch: {len(x0)} points vs {len(y0)} scores. Using {min_len} points.", colors.ENDC)
    
    return x0[:min_len], y0[:min_len]

### NEW HELPER FUNCTION ###
def _restore_symbolic_space(cfg_name: str, ctrl: controller, base_space: Space, first_ss: search_space) -> Space:
    """
    Restores the search space by re-playing symbolic logs.
    """
    sym_log_path = Path(cfg_name) / "algorithm_logs" / "tuning_symbolic_logs.txt"
    diag_log_path = Path(cfg_name) / "algorithm_logs" / "diagnosis_symbolic_logs.txt"
    param_log_path = Path(cfg_name) / "algorithm_logs" / "hyper-neural.txt"

    # Start from the base space
    const_space = copy.deepcopy(base_space)
    
    # Tell the controller to initialize its tuning rules
    ctrl.continue_learning(const_space)

    try:
        # Open all three logs simultaneously
        with open(sym_log_path, "r") as sym_tuning_logs, \
             open(diag_log_path, "r") as diagnosis_logs, \
             open(param_log_path, "r") as params_log:

            # Use zip to read line-by-line
            for sym_line, diag_line, param_line in zip(sym_tuning_logs, diagnosis_logs, params_log):
                try:
                    # Safely parse all log lines
                    sym_tuning = ast.literal_eval(sym_line.strip())
                    diagnosis = ast.literal_eval(diag_line.strip())
                    params = ast.literal_eval(param_line.strip() or "{}")
                except (ValueError, SyntaxError) as e:
                    print(colors.WARNING, f"Skipping log line due to parsing error: {e}", colors.ENDC)
                    continue

                # Replay the repair step to get the *next* space
                const_space = ctrl.tr.repair(
                    sym_tuning, diagnosis, params, const_space
                )
                # Expand the *base* space if new dimensions were added
                if cfg.opt == 'basic':
                    base_space = const_space
                else:
                    base_space = first_ss.expand_space(base_space, const_space)
        
        # Return the final space, reconstructed from all log entries
        return const_space

    except FileNotFoundError:
        print(colors.WARNING, f"Symbolic log files not found. Cannot continue learning. Returning base space.", colors.ENDC)
        return base_space
    except Exception as e:
        print(colors.FAIL, f"Unexpected error during log restoration: {e}", colors.ENDC)
        return base_space


# ------------------------------ optimization ---------------------------------

###
### REFACTORED run_optimization
###
def run_optimization(base_space: Space, first_ss: search_space, ctrl: controller, max_iter: int):
    """
    Outer optimization loop:
      - Load previous state (if any),
      - Restore symbolic space (if using rules),
      - Iteratively optimize, diagnose, and mutate the space.
    """
    # --- 1. Initialization ---
    callback = None # CheckpointSaver(ckpt_path, compress=9)
    no_rules = ["RS", "standard"]
    with_rules = ["filtered", "RS_ruled", "basic"]
    use_filter = (cfg.opt == "filtered")
    res = None # This will hold the skopt result object

    # --- 2. Restore Initial Search Space ---
    # This logic is now fixed and outside the loop
    if cfg.opt in no_rules:
        const_space = copy.deepcopy(base_space)
    else:
        # This calls the new helper function
        const_space = _restore_symbolic_space(cfg.name, ctrl, base_space, first_ss)

    if cfg.verbose > 0:
        print("Restored search space at start:")
        print_space(const_space)
    # --- 3. Load Previous State ---
    x0, y0 = _load_previous_results(cfg.name, base_space)
    ctrl.iter = len(y0)  # Start from the number of already evaluated points
    print(colors.OKBLUE, f"Loaded {len(y0)} previous evaluations.", colors.ENDC)
    
    # This was the bug: The function no longer returns here.
    # It now correctly proceeds to the while loop.
    
    # Initialize the optimizer instance if using custom RandomSearch
    random_search_optimizer = None
    if "RS" in cfg.opt:
        random_search_optimizer = RandomSearch(random_state=cfg.seed, total_iter=max_iter)


    # --- 4. Main Optimization Loop ---
    while ctrl.iter < max_iter and not ctrl.convergence:
        print(colors.MAGENTA, f"--- ITERATION {ctrl.iter + 1}/{max_iter} ---", colors.ENDC) # +1 for user-friendly 1-based index

        # If the space *structure* changed, we must "cold-start"
        if res is not None and len(const_space.dimensions) != len(res.space.dimensions):
            ### BUG FIX ###
            # This logic replaces the old, buggy 'else' block
            print(colors.WARNING, "Search space dimensions have changed! Restarting optimizer.", colors.ENDC)
            x0, y0 = [], [] # Reset warm-start data
            base_space = copy.deepcopy(const_space) # Sync base_space
        elif res is not None:
            # Standard "warm-start"
            x0, y0 = res.x_iters, res.func_vals

        # (Re)create functions with the *current* space
        obj_fn = ObjectiveWrapper(const_space, ctrl)
        const_fn = ConstraintsWrapper(const_space)

        if cfg.verbose > 0:
            print("Actual search space for this iteration:")
            print_space(const_space)

        # --- 5. Run One Step of Optimization ---
        try:
            if "RS" in cfg.opt:
                res = random_search_optimizer(obj_fn.objective, const_space,
                                              callback=callback
                                              )
            else:
                is_warm_start = len(y0) > 0
                res = gp_minimize(
                    obj_fn.objective,
                    base_space, 
                    x0=list(x0) if is_warm_start else None,
                    y0=list(y0) if is_warm_start else None,
                    acq_func="EI",
                    n_calls=1,
                    n_random_starts= 0 if is_warm_start else 1,
                    random_state=cfg.seed,
                    callback=callback,
                    space_constraint=const_fn.apply_constraints if use_filter else None
                )
        except Exception as e:
            # Fallback on error
            print(colors.FAIL, f"Optimization error: {e}. Retrying with a random sample.", colors.ENDC)
            if "RS" in cfg.opt:
                res = random_search_optimizer(obj_fn.objective, const_space, callback=callback)
            else:
                res = gp_minimize(
                    obj_fn.objective, const_space,
                    acq_func="EI", n_calls=1, n_random_starts=1,
                    random_state=cfg.seed, callback=callback,
                    space_constraint=const_fn.apply_constraints if use_filter else None
                )
        
        # Update history from the result object
        x0, y0 = res.x_iters, res.func_vals

        # --- 6. Diagnosis and Space Update ---
        if cfg.opt in no_rules:
            next_space = copy.deepcopy(base_space)
        else:
            next_space = ctrl.diagnosis(const_space)
            if cfg.opt == 'basic':
                base_space = next_space
            else:
                base_space = first_ss.expand_space(base_space, next_space)

        const_space = copy.deepcopy(next_space) # Set space for *next* iteration
        
        # Note: ctrl.iter is incremented inside ctrl.training() via the objective
    
    print(colors.OKGREEN, "\nOptimization loop finished.", colors.ENDC)
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
    parser.add_argument("--early_stop", type=int, default=60,
                        help="Early stopping patience")
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
    parser.add_argument("--verbose", type=int, default=1, 
                        help="Verbosity level (0: silent, 1: print space, 2: print space and model summary)")
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


### NEW HELPER FUNCTION for main loop ###
def run_single_experiment(args: argparse.Namespace):
    """
    Sets up and runs a complete optimization experiment based on the provided args.
    """
    print(colors.HEADER, f"\n=== Experiment: {args.name} | Dataset: {args.dataset} | Opt: {args.opt} | Seed: {args.seed} ===", colors.ENDC)
    
    # --- 1. Config and Filesystem Setup ---
    exp_dir = Path(args.name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_path = create_config_file(exp_dir, overrides=args.__dict__)
    set_active_config(cfg_path)
    
    # Force load to ensure cfg is updated for this loop iteration
    global cfg 
    cfg = load_cfg(force=True)
    
    # Dynamically import the correct gp_minimize
    global gp_minimize
    if cfg.opt == 'basic':
        from skopt import gp_minimize
    else:
        from components.gp import gp_minimize

    # --- 2. Dataset and Controller Setup ---
    dataset_key = cfg.dataset.strip().lower().replace("-", "")
    X_train, Y_train, X_test, Y_test, n_classes = get_datasets(dataset_key)

    ctrl = controller(X_train, Y_train, X_test, Y_test, n_classes)
    first_ss = search_space()
    first_space = first_ss.search_sp(max_block=ctrl.max_conv, max_dense=ctrl.max_fc)
    
    print(colors.MAGENTA, "|  ----------- BASIC SEARCH SPACE ----------  |\n", colors.ENDC)
    print_space(first_space)

    # --- 3. Run Optimization ---
    start_time = time.time()
    print(colors.OKGREEN, "\nSTARTING ALGORITHM \n", colors.ENDC)
    
    res = run_optimization(
        base_space=first_space, 
        first_ss=first_ss, 
        ctrl=ctrl, 
        max_iter=cfg.eval
    )
    
    end_time = time.time()
    print(colors.OKGREEN, "\nALGORITHM FINISHED \n", colors.ENDC)
    print(colors.CYAN, f"\nTOTAL TIME --------> {end_time - start_time:.2f} seconds\n", colors.ENDC)


# ---------------------------------- main -------------------------------------

if __name__ == "__main__":
    # --- Global Setup (runs once) ---
    if os.name == 'nt': # Windows ANSI color fix
        os.system("")

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Parse base command-line arguments (can be overridden by the file)
    base_args = parse_args()

    # --- Experiment Loop ---
    run_single_experiment(base_args)
    # except Exception as e:
    #     print(colors.FAIL, f"An error occurred during the experiment batch: {e}", colors.ENDC)
    #     sys.exit(1)