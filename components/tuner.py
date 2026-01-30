import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Type, Union
import copy
import shutil
from pathlib import Path 
import sys

from skopt.space import Categorical, Space
from skopt.callbacks import CheckpointSaver

from components.colors import colors
from components.controller import controller
from components.search_space import search_space as SearchSpace
from components.neural_network import NeuralNetwork
from components.backend_interface import BackendInterface
from components.dataset import TunerDataset
from components.random_search import RandomSearch
from components.objFunction import ObjectiveWrapper
from components.constraints import ConstraintsWrapper

from exp_config import load_cfg





def _default_folders() -> List[str]:
    return [
        "Model",
        "database",
        "log_folder",
        "algorithm_logs",
        "dashboard/model",
        "symbolic"
    ]


def _default_gp_kwargs() -> Dict[str, Any]:
    return {
        "acq_func": "EI",
        "n_calls": 1,
        "n_random_starts": 1,
    }


@dataclass
class TunerConfig:
    """
    Configuration payload used to bootstrap the tuner with backend specific components.
    """
    neural_network_cls: Type[NeuralNetwork]
    module_backend_cls: Type[BackendInterface]
    dataset: TunerDataset
    search_space: Optional[Sequence[Any]] = None
    max_evals: int = 1
    checkpoint_path: str = "checkpoints/checkpoint.pkl"
    objective_log_path: str = "algorithm_logs/hyper-neural.txt"
    clear_session_callback: Optional[Callable[[], None]] = None
    gp_minimize_kwargs: Dict[str, Any] = field(default_factory=_default_gp_kwargs)
    folders_to_create: Iterable[str] = field(default_factory=_default_folders)
    fixed_hyperparams: Optional[Dict[str, Any]] = None


class Tuner:
    """
    Framework agnostic tuner orchestrator. Backend specific projects
    should instantiate this class, provide their neural network and module backend
    implementations, and call `run`.
    """

    def __init__(self, config: TunerConfig):
        self.config = config
        self.exp_cfg = load_cfg()
        self._ensure_directories()
        self._copy_symbolic_files()

        self.dataset = self.config.dataset

        # self.neural_network = self.config.neural_network_cls(self.dataset)
        self.module_backend = self.config.module_backend_cls()

        self.controller = controller(
            self.config.neural_network_cls,
            self.module_backend,
            self.dataset,
            clear_session_callback=self.config.clear_session_callback,
        )

        self.gp_kwargs = {**_default_gp_kwargs(), **self.config.gp_minimize_kwargs}
        self._start_time: Optional[float] = None


    def run(self, max_evals: Optional[int] = None):
        """
        Execute Bayesian optimisation and symbolic diagnosis loop.
        :param max_evals: optional override for number of diagnosis iterations.
        :return: skopt optimization result.
        """

        print(colors.OKGREEN, "\nSTART ALGORITHM \n", colors.ENDC)

        self._start_time = time.time()
        first_ss = SearchSpace()
        first_space = first_ss.search_sp(max_block=self.controller.max_conv, max_dense=self.controller.max_fc)
        result = self._start(base_space=first_space, first_ss=first_ss)

        print(result)
        print(colors.OKGREEN, "\nEND ALGORITHM \n", colors.ENDC)

        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            print(colors.CYAN, "\nTIME --------> \n", elapsed, colors.ENDC)


        return result


    def _ensure_directories(self):
        for folder in self.config.folders_to_create:
            try:
                os.makedirs(f"{self.exp_cfg.exp_dir}/{folder}", exist_ok=True)
            except OSError:
                print(colors.FAIL, "|  ----------- FAILED TO CREATE FOLDER ----------  |\n", colors.ENDC)
                
    def _copy_symbolic_files(self) -> None:
        """
        Copy the base symbolic files into this experiment's symbolic folder.
        """
        src_dir = Path("./symbolic_base")
        dst_dir = Path(f"{self.exp_cfg.exp_dir}/symbolic") # or Path(cfg.name) / "symbolic"

        if not src_dir.is_dir():
            print(colors.WARNING, f"Symbolic base folder not found at '{src_dir}'. Skipping copy.", colors.ENDC)
            return
        try:
            # Copy only files (flat copy). Avoids deep-copying and is simpler.
            for src_path in src_dir.glob('*'): # glob('*') gets files and dirs
                if src_path.is_file():
                    dst_path = dst_dir / src_path.name
                    shutil.copy2(src_path, dst_path)
        except OSError as e:
            print(colors.FAIL, f"Failed to copy symbolic directory: {e}", colors.ENDC)
            sys.exit(1) # Critical setup failure
            
    def _print_space(self, space: Space) -> None:
        """Pretty print the current Space dimensions with names and bounds."""
        for i, dim in enumerate(space.dimensions):
            if isinstance(dim, Categorical):
                print(f"Dimension {i}: {dim.name} - categories: {dim.categories}")
            else:
                if dim.high != 0:
                    print(f"Dimension {i}: {dim.name} - {dim}")


    def _start_analysis(self):
        """
        Run symbolic diagnosis and return the updated search space with the corresponding value.
        """
        return self.controller.diagnosis()

    def _check_continuing_bo(self, new_space, x_iters, func_vals):
        x_iters = list(x_iters)
        func_vals = func_vals.tolist()
        for x in list(x_iters):
            for dim, value in zip(new_space, x):
                if value < dim.low or value > dim.high:
                    index = x_iters.index(x)
                    func_vals.pop(index)
                    x_iters.remove(x)
                    break
        return x_iters, func_vals

    def _start(self, base_space: Space, first_ss: SearchSpace):
        """
        Execute the optimisation loop.
        """
        
        # --- 1. Initialization ---
        # Dynamically import the correct gp_minimize based on config
        if self.exp_cfg.opt == 'basic':
            from skopt import gp_minimize
        else:
            from components.bo.gp import gp_minimize
        callback = None # CheckpointSaver(ckpt_path, compress=9)
        no_rules = ["RS", "standard"]
        with_rules = ["filtered", "RS_ruled", "basic"]
        use_filter = (self.exp_cfg.opt == "filtered")
        
        # const_space is the *dynamic* space, which may be mutated by rules.
        # base_space is the *static* reference, which we expand if new dims are found.
        const_space = SearchSpace().search_sp(max_block=self.controller.max_conv, max_dense=self.controller.max_fc)
        
        # x0, y0 hold the history of points. Starts empty.
        x0, y0 = [], []
        res = None
        
        # Initialize the optimizer instance if using custom RandomSearch
        random_search_optimizer = None
        if "RS" in self.exp_cfg.opt:
            random_search_optimizer = RandomSearch(random_state=self.exp_cfg.seed, total_iter=self.exp_cfg.eval)

        # --- 2. Main Optimization Loop ---
        # We use self.controller.iter (which starts at 1) and loop *while* it's <= max_iter
        while self.controller.iter <= self.exp_cfg.eval and not self.controller.convergence:
            print(colors.MAGENTA, f"--- ITERATION {self.controller.iter} ---", colors.ENDC)

            # (Re)create the objective and constraint functions for this iteration.
            # This ensures they capture the *current* const_space
            obj_fn = ObjectiveWrapper(const_space, self.controller)
            const_fn = ConstraintsWrapper(const_space)

            if self.exp_cfg.verbose > 0:
                print("Actual constrained space for this iteration:")
                self._print_space(const_space)
                print("Actual base space for this iteration:")
                self._print_space(base_space)

            # --- 3. Run One Step of Optimization ---
            try:
                if "RS" in self.exp_cfg.opt:
                    # Random search: draw one more sample/eval
                    res = random_search_optimizer(obj_fn.objective, const_space,
                                                callback=callback
                                                )
                else:
                    # Bayesian Optimization: one call.
                    # If x0, y0 are empty, it's a "cold-start" (n_random_starts=1)
                    # If x0, y0 are present, it's a "warm-start" (n_random_starts=0)
                    is_warm_start = len(y0) > 0
                    res = gp_minimize(
                        obj_fn.objective,
                        base_space,  
                        x0=list(x0) if is_warm_start else None,
                        y0=list(y0) if is_warm_start else None,
                        acq_func="EI",
                        n_calls=1,
                        n_random_starts= 0 if is_warm_start else 1,
                        random_state=self.exp_cfg.seed,
                        callback=callback,
                        space_constraint=const_fn.apply_constraints if use_filter else None
                    )
            except Exception as e:
                # Fallback: If gp_minimize fails (e.g., matrix inversion),
                # try a random step to get new data.
                print(colors.FAIL, f"Optimization error: {e}. Retrying with a random sample.", colors.ENDC)
                if "RS" in self.exp_cfg.opt:
                    res = random_search_optimizer(obj_fn.objective, base_space, callback=callback)
                else:
                    # Use standard gp_minimize with n_random_starts=1 to get a new point
                    res = gp_minimize(
                        obj_fn.objective, base_space,
                        acq_func="EI", n_calls=1, n_random_starts=1,
                        random_state=self.exp_cfg.seed, callback=callback,
                        space_constraint=const_fn.apply_constraints if use_filter else None
                    )
            
            # Update our history of points
            x0, y0 = res.x_iters, res.func_vals

            # --- 4. Diagnosis and Space Update ---
            # Ask the controller to propose the *next* space based on this iteration
            if self.exp_cfg.opt in no_rules:
                # 'no_rules' means the space never changes
                next_space = copy.deepcopy(base_space)
            else:
                # 'with_rules' means the controller diagnoses and mutates the space
                next_space = self.controller.diagnosis(const_space)
                # Expand the base_space if the controller added new dimensions
                if self.exp_cfg.opt == 'basic':
                    base_space = next_space
                else:
                    base_space = first_ss.expand_space(base_space, next_space)

            # --- 5. Check for Space Change (This is the critical bug fix) ---
            if len(next_space.dimensions) != len(const_space.dimensions):
                # Space dimensions have changed!
                # We must reset the optimizer and start cold (no x0, y0).
                print(colors.WARNING, "Search space dimensions have changed! Restarting optimizer.", colors.ENDC)
                x0, y0 = [], [] # Reset warm-start data
            
            # The space for the *next* iteration is the one just diagnosed
            const_space = copy.deepcopy(next_space)

            # Note: ctrl.iter is incremented *inside* ctrl.training()
        
        print(colors.OKGREEN, "\nOptimization loop finished.", colors.ENDC)
        return res
