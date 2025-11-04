from __future__ import annotations

import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np  # only for type hints; safe if arrays are numpy-like
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import gc

try:
    from components.colors import colors
except Exception:  # pragma: no cover
    class _NoColor:
        OKBLUE = CYAN = FAIL = ENDC = ""
    colors = _NoColor()  # type: ignore

from components.neural_network import neural_network
from components.search_space import search_space
from components.tuning_rules_symbolic import tuning_rules_symbolic
from components.neural_sym_bridge import NeuralSymbolicBridge
from components.lfi_integration import LfiIntegration
from components.storing_experience import StoringExperience
from components.improvement_checker import ImprovementChecker
from components.integral import integrals
from shutil import copyfile

from modules.module import module

from exp_config import load_cfg
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class controller:
    """
    Orchestrates training and iterative tuning of the neural network.

    Responsibilities:
      - Hold data references and high-level training loop state.
      - Delegate architecture/hyperparameter edits to search/tuning components.
      - Bridge numeric signals to the symbolic reasoning layer.
      - Persist run artifacts (DB, best model) and provide diagnostic utilities.
    """

    # ---------------------------- Lifecycle ----------------------------------

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        n_classes: int,
    ) -> None:
        """
        Initialize state and all helper components.

        Args:
            X_train, Y_train, X_test, Y_test: Training/validation arrays.
            n_classes: Number of target classes for the classifier.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.n_classes = n_classes


        self.cfg = load_cfg()
        
        # Internal counters
        self.count_new_fc = 0
        self.count_new_cv = 0
        self.max_fc = 3
        self.start_conv, self.start_fc = self.ss.count_initial_layers(self.space)
        self.max_conv = self.count_max_conv(base_blocks=self.start_conv)
        self.count_no_probs = 0
        self.layer_x_block = 2
        self.max_layer_x_block = 6

        # Search space + tuning rules
        self.ss = search_space()
        self.space = self.ss.search_sp()
        self.tr = tuning_rules_symbolic(self.space, self.ss, self)

        # Symbolic reasoning & learning-from-interpretations
        self.nsb = NeuralSymbolicBridge()
        self.db = StoringExperience()
        self.db.create_db()
        self.lfi = LfiIntegration(self.db)

        # Iterative evidence from symbolic reasoning
        self.symbolic_tuning: List[str] = []
        self.symbolic_diagnosis: List[str] = []
        self.issues: List[str] = []

        # Exponential smoothing weight for history curves
        self.weight: float = 0.6

        # Action flags (reset each training call)
        self.da: Optional[bool] = None
        self.reg: Optional[bool] = None
        self.residual: Optional[bool] = None

        # Model/training bookkeeping
        self.nn: Optional[neural_network] = None
        self.model: Any = None
        self.params: Optional[Dict[str, Any]] = None
        self.iter: int = 0

        # Dynamic thresholds (updated at given epochs)
        self.lacc: float = 0.30
        self.hloss: float = np.log(n_classes)
        self.acc_w = 0.5  # weight of accuracy in combined score
        self.vanish_th = 1e-8
        self.exploding_th = 100.0

        # Improvement checker + modules
        self.imp_checker = ImprovementChecker(self.db, self.lfi)
        self.modules = module(self.cfg.mod_list)
        if "flops_module" in self.cfg.mod_list:
            self.flops_th = self.modules.get_module("flops_module").flops_th

        # Optimization objective bookkeeping
        self.best_score: float = 1e10 #float("inf")  # lower is better if we minimize
        self.convergence: bool = False
        self.best_iter: int = -1

        

    # # ------------------- Signals from tuning_rules_symbolic -------------------

    def set_data_augmentation(self, da: bool) -> None:
        """Enable/disable data augmentation for the next training call."""
        self.da = da

    def set_residual(self, residual: bool) -> None:
        """Enable/disable residual connection for the next training call."""
        self.residual = residual

    def set_reg_l2(self, reg: bool) -> None:
        """Enable/disable data augmentation for the next training call."""
        self.reg = reg

    # ----------------------------- Utilities ---------------------------------

    def count_max_conv(self, base_blocks: int = 2) -> int:
        """
        Compute how many additional (Conv... -> MaxPool(2x2)) blocks can fit
        given the current input spatial size.

        Assumptions (matching `build_network`):
        - Convs use `padding="same"` → do NOT change H/W.
        - Each conv block ends with MaxPooling2D(2,2) → halves H and W.
        - After all conv stacks you use GlobalAveragePooling2D → only need H,W >= 1.
        - We exclude the initial 2 pools (c1 & c2 blocks) via `base_blocks`.

        Args:
            base_blocks: number of pool operations already used by the fixed stem
                        (default 2 for the c1 and c2 sections).

        Returns:
            int: maximum number of *additional* conv+pool blocks that can be appended.
        """
        # Extract (H, W) robustly from train_data:
        # - gesture fwdPass/hybrid: (N, T, H, W, C)  -> H=shape[2], W=shape[3]
        # - generic images:         (N, H, W, C)     -> H=shape[1], W=shape[2]
        # - single sample:          (H, W, C)        -> H=shape[0], W=shape[1]
        td = getattr(self, "train_data", None)
        if td is None or not hasattr(td, "shape"):
            # Fallback to controller.X_train if presente nel tuo progetto
            td = getattr(self, "X_train", None)
            if td is None or not hasattr(td, "shape"):
                # ultimo fallback
                return 0

        dims = td.shape
        if len(dims) >= 5:
            # (N, T, H, W, C)
            h, w = int(dims[2]), int(dims[3])
        elif len(dims) == 4:
            # (N, H, W, C)
            h, w = int(dims[1]), int(dims[2])
        elif len(dims) == 3:
            # (H, W, C)
            h, w = int(dims[0]), int(dims[1])
        else:
            return 0

        # Number of MaxPool(2,2) steps possible until min(H,W) would drop below 1:
        # after k pools: floor(min(h,w) / 2^k) >= 1  ->  k <= floor(log2(min(h,w)))
        from math import log2, floor
        min_side = max(1, min(h, w))
        max_pools_total = floor(log2(min_side))  # e.g., 32→5, 28→4, 3→1, 2→1, 1→0

        # Additional blocks available beyond the fixed stem (base_blocks):
        extra_blocks = max(0, max_pools_total - max(0, int(base_blocks)))
        return extra_blocks

    def smooth(self, scalars: List[float]) -> List[float]:
        """
        Exponential moving average smoothing over a scalar sequence.

        Args:
            scalars: Values to be smoothed (e.g., accuracy or loss history).

        Returns:
            Smoothed values (same length). If input is empty/len=1, returns input.
        """
        if not scalars:
            return []
        if len(scalars) == 1:
            return scalars[:]

        last = scalars[0]
        smoothed: List[float] = []
        for point in scalars:
            smoothed_val = last * self.weight + (1.0 - self.weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def manage_configuration(self) -> None:
        """
        If present, invoke the energy module to select a better runtime configuration.
        """
        energy_name = "energy_module"
        if energy_name in self.modules.modules_name:
            index = self.modules.modules_name.index(energy_name)
            try:
                self.modules.modules_obj[index].fix_configuration()
            except Exception as e:  # robust to module-specific issues
                logger.warning("Energy module failed to fix configuration: %s", e)

    # ------------------------------ Training ---------------------------------

    def training(self, params: Dict[str, Any]) -> float:
        """
        Train and evaluate the neural network once.

        Args:
            params: Hyperparameters for this run.

        Returns:
            Objective score to minimize (negative accuracy by default or module-provided).
        """
        self.params = params
        print(colors.OKBLUE, "|  --> START TRAINING\n", colors.ENDC)

        # Reset low-level session state to avoid graph buildup across iterations
        K.clear_session()

        # Build and train model
        self.set_data_augmentation(params.get("data_augmentation", False))
        self.set_reg_l2(params.get("reg_l2", False))
        self.set_residual(params.get("skip_connection", False))
        print("Action flags for this training: ")
        print(f"  Data Augmentation: {self.da}")
        print(f"  L2 Regularization: {self.reg}")
        print(f"  Residual Connections: {self.residual}")
        self.nn = neural_network(self.X_train, self.Y_train, self.X_test, self.Y_test, self.n_classes, self.reg, self.da, self.residual)
        self.nn.build_network(params, self.layer_x_block)
        if self.nn.flops is None or self.nn.flops <= self.flops_th:
            self.scoreNN, self.history, self.model = self.nn.training(params)
            self.log()
            # Update external modules and log their state
            self.modules.state(self.model, self.nn.flops, self.nn.nparams)
            self.modules.print()
            self.modules.log()
            
            # Decide the optimization target:
            # - If there are no/invalid modules, optimize accuracy (minimize -acc).
            # - Otherwise use module-provided objective.
            if (len(self.modules.modules_obj) == 0) or (not self.modules.all_zeros_weights()) or (not self.modules.ready()):
                self.score = -float(self.scoreNN[1])  # usually validation accuracy
            else:
                _, _, opt_value = self.modules.optimiziation()
                self.score = float(opt_value)-self.scoreNN[1]*self.acc_w  # combine module opt with accuracy

        else:
            print(f"Model FLOPs {self.nn.flops} exceed maximum {self.flops_th}. Skipping training.")
            self.scoreNN, self.history, self.model = None, None, self.nn.model
    
            self.log()
            # Update external modules and log their state
            self.modules.state(self.model, self.nn.flops, self.nn.nparams)
            self.modules.print()
            self.modules.log()

            self.score = 1e10 #float('inf')
            
        self.iter += 1
        # Track the best score and persist the model artifact
        if self.score < self.best_score:
            self.best_score = self.score
            self.best_iter = self.iter
            try:
                os.makedirs(os.path.join(self.cfg.name, "Model"), exist_ok=True)
                self.model.save(f"{self.cfg.name}/Model/best-model.keras")
            except Exception as e:
                logger.warning("Failed to save best model: %s", e)

        # if we have no improv in 10 iter end tuner evaluations
        if self.iter > self.best_iter + 20:
            self.convergence = True
        try:
            del self.nn.model
            K.clear_session()
            gc.collect()
        except Exception as e:
            logger.warning("Failed to clear session: %s", e)
        return self.score
    # ------------------------------ Diagnosis --------------------------------

    def diagnosis(self) -> Tuple[Any]:
        """
        Diagnose potential issues (e.g., overfitting) and propose tuning actions.

        Returns:
            (new_search_space, objective_value) where objective_value is what the
            Bayesian optimizer should minimize on the next iteration.
        """
        print(colors.CYAN, "| START SYMBOLIC DIAGNOSIS ----------------------------------  |\n", colors.ENDC)

        # Open logs with context managers to guarantee closure
        os.makedirs(f"{self.cfg.name}/algorithm_logs", exist_ok=True)
        diag_path = f"{self.cfg.name}/algorithm_logs/diagnosis_symbolic_logs.txt"
        tune_path = f"{self.cfg.name}/algorithm_logs/tuning_symbolic_logs.txt"

        with open(diag_path, "a") as diagnosis_logs, open(tune_path, "a") as tuning_logs:
            # Check last-iteration improvement; persist scores (score and loss) to DB
            improv = self.imp_checker.checker(self.score)
            self.db.insert_ranking(self.score)

            if self.nn.flops is None or self.nn.flops <= self.flops_th:
                # Integral features of validation loss (used by symbolic layer)
                val_loss_hist = self.history.get("val_loss", [])
                int_loss, int_slope = integrals(val_loss_hist)

                # Base fact list (raw + smoothed histories)
                gnorm = getattr(self.model.optimizer, "last_grad_global_norm", 1.0)
                if not isinstance(gnorm, float):
                    gnorm = float(gnorm.numpy())
                facts_list_module = [
                    self.history.get("loss", []),
                    self.smooth(self.history.get("loss", [])),
                    self.history.get("accuracy", []),
                    self.smooth(self.history.get("accuracy", [])),
                    self.history.get("val_loss", []),
                    self.history.get("val_accuracy", []),
                    int_loss,
                    int_slope,
                    self.lacc,
                    self.hloss,
                    gnorm,
                    self.vanish_th,
                    self.exploding_th
                ]

                # Add facts from loaded modules
                facts_list_module += list(self.modules.values().values())
                self.only_modules = False

            else:
                # Add facts from loaded modules
                facts_list_module = list(self.modules.values().values())
                self.only_modules = True
            # First diagnosis iteration: assemble rule base from modules and build logic program
            if self.iter == 1:
                self.rules, self.actions, self.problems = self.modules.get_rules()

                for module, no_err in zip(self.modules.modules_obj, self.modules.modules_ready):
                    # if there are no errors in the module, dynamically add facts and problems to the symbolic part
                    if no_err:
                        self.nsb.initial_facts += module.facts
                        self.nsb.problems += module.problems

                # Create/refresh the symbolic problem file
                self.nsb.build_sym_prob(self.problems)

            # If we have improvement data, update the symbolic model probabilities
            if improv is not None:
                _, lfi_problem = self.lfi.learning(
                    improv, self.symbolic_tuning, self.symbolic_diagnosis, self.actions
                )
                sy_model = lfi_problem.get_model()
                self.nsb.edit_probs(sy_model)

            # Run rule-based reasoning to produce candidate repairs and diagnoses
            self.symbolic_tuning, self.symbolic_diagnosis = self.nsb.symbolic_reasoning(
                facts_list_module, diagnosis_logs, tuning_logs, self.rules, self
            )


        print(colors.CYAN, "| END SYMBOLIC DIAGNOSIS   ----------------------------------  |\n", colors.ENDC)

        # If we have candidate repairs, perform tuning; otherwise return current space
        if self.symbolic_tuning:
            new_space = self.tuning()
            return new_space
        else:
            return self.space

    # ------------------------------- Tuning ----------------------------------

    def tuning(self) -> Tuple[Any]:
        """
        Apply symbolic tuning actions to the hyperparameter space and/or architecture.

        Returns:
            (new_hp_space, objective_value, model)
        """
        print(colors.FAIL, "| START SYMBOLIC TUNING    ----------------------------------  |\n", colors.ENDC)

        # Perform the actual repairs via the tuning rules engine
        new_space, self.model = self.tr.repair(
            self.symbolic_tuning, self.symbolic_diagnosis, self.model, self.params or {}
        )

        # Simple convergence heuristic: too many iterations without clear problems
        if self.tr.count_no_probs > 5:
            self.convergence = True

        self.issues = []
        print(colors.FAIL, "| END SYMBOLIC TUNING      ----------------------------------  |\n", colors.ENDC)

        return new_space

    def log(self) -> None:
        f = open(f"{self.cfg.name}/algorithm_logs/acc_report.txt", "a")
        if self.scoreNN is not None:
            print(f"\nACCURACY: {self.scoreNN[1]}\n")
            f.write(str(self.scoreNN[1]) + "\n")
        else:
            print(f"\nACCURACY: {0.0}\n")
            f.write("None \n")
        f.close()
