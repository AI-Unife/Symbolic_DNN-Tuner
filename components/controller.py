from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np  # only for type hints; safe if arrays are numpy-like
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

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

import config as cfg
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

        # Model/training bookkeeping
        self.nn: Optional[neural_network] = None
        self.model: Any = None
        self.params: Optional[Dict[str, Any]] = None
        self.iter: int = 0

        # Dynamic thresholds (updated at given epochs)
        self.lacc: float = 0.15
        self.hloss: float = 1.2
        self.levels: List[int] = [7, 10, 13]

        # Improvement checker + modules
        self.imp_checker = ImprovementChecker(self.db, self.lfi)
        self.modules = module(cfg.MOD_LIST)

        # Optimization objective bookkeeping
        self.best_score: float = float("inf")  # lower is better if we minimize
        self.convergence: bool = False

    # ------------------- Signals from tuning_rules_symbolic -------------------

    def set_data_augmentation(self, da: bool) -> None:
        """Enable/disable data augmentation for the next training call."""
        self.da = da

    # ----------------------------- Utilities ---------------------------------

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
        self.nn = neural_network(self.X_train, self.Y_train, self.X_test, self.Y_test, self.n_classes)
        self.score, self.history, self.model = self.nn.training(params, self.da)

        # Update external modules and log their state
        self.modules.state(self.score[0], self.score[1], self.model)
        self.modules.print()
        self.modules.log()

        self.da = False

        self.iter += 1

        # Decide the optimization target:
        # - If there are no/invalid modules, optimize accuracy (minimize -acc).
        # - Otherwise use module-provided objective.
        if (len(self.modules.modules_obj) == 0) or (not self.modules.all_zeros_weights()) or (not self.modules.ready()):
            score = -float(self.score[1])  # usually validation accuracy
        else:
            _, _, opt_value = self.modules.optimiziation()
            score = float(opt_value)

        # Track the best score and persist the model artifact
        if score < self.best_score:
            self.best_score = score
            try:
                os.makedirs(os.path.join(cfg.NAME_EXP, "Model"), exist_ok=True)
                self.model.save(f"{cfg.NAME_EXP}/Model/best-model.keras")
            except Exception as e:
                logger.warning("Failed to save best model: %s", e)

        return score

    # ------------------------------ Diagnosis --------------------------------

    def diagnosis(self) -> Tuple[Any, float]:
        """
        Diagnose potential issues (e.g., overfitting) and propose tuning actions.

        Returns:
            (new_search_space, objective_value) where objective_value is what the
            Bayesian optimizer should minimize on the next iteration.
        """
        print(colors.CYAN, "| START SYMBOLIC DIAGNOSIS ----------------------------------  |\n", colors.ENDC)

        # Open logs with context managers to guarantee closure
        os.makedirs(f"{cfg.NAME_EXP}/algorithm_logs", exist_ok=True)
        diag_path = f"{cfg.NAME_EXP}/algorithm_logs/diagnosis_symbolic_logs.txt"
        tune_path = f"{cfg.NAME_EXP}/algorithm_logs/tuning_symbolic_logs.txt"

        with open(diag_path, "a") as diagnosis_logs, open(tune_path, "a") as tuning_logs:
            # Check last-iteration improvement; persist scores to DB
            improv = self.imp_checker.checker(self.score[1], self.score[0])
            self.db.insert_ranking(self.score[1], self.score[0])

            # Integral features of validation loss (used by symbolic layer)
            val_loss_hist = self.history.get("val_loss", [])
            int_loss, int_slope = integrals(val_loss_hist)

            # Dynamically update detection thresholds at specific iterations
            if self.iter in self.levels:
                self.lacc = self.lacc / 2.0 + 0.05
                self.hloss = self.hloss / 2.0 + 0.15

            # Base fact list (raw + smoothed histories)
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
            ]

            # Add facts from loaded modules
            facts_list_module += list(self.modules.values().values())

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
                print("improv: ", improv)
                _, lfi_problem = self.lfi.learning(
                    improv, self.symbolic_tuning, self.symbolic_diagnosis, self.actions
                )
                sy_model = lfi_problem.get_model()
                self.nsb.edit_probs(sy_model)

            # Run rule-based reasoning to produce candidate repairs and diagnoses
            self.symbolic_tuning, self.symbolic_diagnosis = self.nsb.symbolic_reasoning(
                facts_list_module, diagnosis_logs, tuning_logs, self.rules
            )

        print(colors.CYAN, "| END SYMBOLIC DIAGNOSIS   ----------------------------------  |\n", colors.ENDC)

        # If we have candidate repairs, perform tuning; otherwise return current space
        if self.symbolic_tuning:
            new_space, to_optimize, _ = self.tuning()
            return new_space, to_optimize
        else:
            return self.space, -float(self.score[1])

    # ------------------------------- Tuning ----------------------------------

    def tuning(self) -> Tuple[Any, float, Any]:
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

        return new_space, -float(self.score[1]), self.model
