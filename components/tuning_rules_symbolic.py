from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple
import logging

try:
    # Pretty terminal colors (optional). Falls back gracefully if unavailable.
    from components.colors import colors
except Exception:  # pragma: no cover
    class _NoColor:
        FAIL = ""
        ENDC = ""
    colors = _NoColor()  # type: ignore


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class tuning_rules_symbolic:
    """
    Rule-based helper to tweak a model architecture and its hyperparameter search space.

    This class centralizes small, composable "repair" actions (add layers, widen search
    bounds, enable regularization, etc.) that can be invoked by a higher-level
    symbolic/diagnostic system when specific issues are detected during training.
    """

    # ----------------------------- Lifecycle ---------------------------------

    def __init__(self, params, ss, controller) -> None:
        """
        Initialize tuning context and bookkeeping.

        Args:
            params: The hyperparameter search space container used by `ss`.
            ss:     A search-space manager exposing `add_params`, `remove_params`,
                    and `count_initial_layers(...)`.
            controller: Object that owns the current model and exposes mutators like
                    `add_fc_layer`, `add_conv_section`, `remove_*`, `set_case`,
                    `set_data_augmentation`, and `manage_configuration`. It must also
                    provide training data shape via `controller.X_train.shape`.
        """
        self.space = params
        self.ss = ss
        self.controller = controller
        self.count_no_probs = 0

    # --------------------------- Regularization ------------------------------

    def reg_l2(self) -> None:
        """
        Enable L2 regularization once.
        """
        self.controller.set_reg_l2(True)
        logger.info("Enabled L2 regularization")

    def remove_reg_l2(self) -> None:
        """
        Disable reg in the controller.
        """
        self.controller.set_reg_l2(False)
        logger.info("Disable L2 regularization")

    def data_augmentation(self) -> None:
        """
        Enable data augmentation in the controller.
        """
        self.controller.set_data_augmentation(True)
        logger.info("Enabled data augmentation")

    def remove_data_augmentation(self) -> None:
        """
        Enable data augmentation in the controller.
        """
        self.controller.set_data_augmentation(False)
        logger.info("Disabled data augmentation")

    def add_residual(self) -> None:
        """
        Enable residual connection in the controller.
        """
        self.controller.set_residual(True)
        logger.info("Enabled residual connection")
    # -------------------------- Architecture edits ---------------------------

    def new_fc_layer(self) -> None:
        """
        Add a fully-connected (dense) layer, respecting a soft upper bound to avoid bloat.
        """
        if self.controller.count_new_fc + self.controller.start_fc > self.controller.max_fc:
            print(
                colors.FAIL,
                "Max number of dense layers reached",
                colors.ENDC,
            )
            print(
                colors.FAIL,
                "start dense layers: ",
                self.controller.start_fc,
                " Dense layers: ",
                self.controller.count_new_fc + self.controller.start_fc,
                " Max dense layers: ",
                self.controller.max_fc,
                colors.ENDC,
            )
            return

        self.controller.count_new_fc += 1
        new_p = {f"new_fc_{self.controller.count_new_fc}": 512}
        self.space = self.ss.add_params(new_p)
        logger.info("Added dense layer #%d with 512 units", self.controller.count_new_fc)

    def dec_fc_layer(self) -> None:
        """
        Remove one dense layer if present.
        """
        print("Removing dense layer")
        print(self.controller.count_new_fc)
        if self.controller.count_new_fc < 1:
            print(colors.FAIL, "No more dense layers to remove", colors.ENDC)
            self.controller.count_new_fc = 0
            return

        new_p = {f"new_fc_{self.controller.count_new_fc}": 512}
        self.space = self.ss.remove_params(new_p)
        logger.info("Removed one dense layer; remaining added layers: %d", self.controller.count_new_fc)
        self.controller.count_new_fc -= 1

    def new_conv_block(self) -> None:
        """
        Add a convolutional *section* (e.g., conv block). Respect input-size constraints.
        """
        if not self.controller.max_conv:
            return

        tot_conv = self.controller.count_new_cv
        if tot_conv > self.controller.max_conv:
            print(colors.FAIL, "Max number of convolutional layers reached", colors.ENDC)
            print(
                colors.FAIL,
                "Convolutional layers: ",
                tot_conv - 2,
                " Max convolutional layers: ",
                self.controller.max_conv,
                colors.ENDC,
            )
            return

        self.controller.count_new_cv += 1
        new_p = {f"new_conv_{self.controller.count_new_cv}": 512}
        self.space = self.ss.add_params(new_p)
        logger.info("Added conv section #%d with 512 filters", self.controller.count_new_cv)

    def dec_conv_block(self) -> None:
        """
        Remove one convolutional section if present.
        """
        if self.controller.count_new_cv < 1:
            logger.info("No more convolutional section to remove")
            self.count_new_cv = 0
            return

        # Remove param key associated with the *next* (now absent) conv
        new_p = {f"new_conv_{self.controller.count_new_cv}": 512}
        self.space = self.ss.remove_params(new_p)
        logger.info("Removed one conv section; remaining added sections: %d", self.controller.count_new_cv)
        self.controller.count_new_cv -= 1

    def inc_conv_layer(self) -> None:
        if self.controller.layer_x_block < self.controller.max_layer_x_block:
            self.controller.layer_x_block += 1
            logger.info("Add one conv layer per section; new layers per sections: %d", self.controller.layer_x_block)
        logger.info("No more convolutional layers per section to add")

    def dec_conv_layer(self) -> None:
        if self.controller.layer_x_block < self.controller.max_layer_x_block:
            self.controller.layer_x_block -= 1
            logger.info("Removed one conv layer per section; remaining layers per sections: %d", self.controller.layer_x_block)
        logger.info("No more convolutional layers per section to remove")

    # -------------------------- Hyperparam tweaks ----------------------------

    def decr_lr(self, params):
        """
        method used to decrement learning rate
        """
        for hp in self.space:
            # check if learning rate is present in the search space and in that case
            # proceed by increasing the upper range adding half of its value
            if hp.name == 'learning_rate':
                hp.high = params['learning_rate'] + (params['learning_rate'] / 2)

    def inc_lr(self, params):
        """
        method used to increment learning rate
        """
        for hp in self.space:
            # check if learning rate is present in the search space and in that case
            # proceed by incresing the upper range with the current learning rate value
            if hp.name == 'learning_rate':
                hp.high = params['learning_rate'] + hp.high

    def inc_neurons(self, params):
        """
        method used to increment the number of neurons
        """
        # itereate over each hyperparameter and if one of these is a convolutional
        # or dense layer decrease the lower value of the range
        for hp in self.space:
            if 'unit_c1' in hp.name:
                hp.low = max(params['unit_c1'] - 16, 16)
            if 'unit_c2' in hp.name:
                hp.low = max(params['unit_c2'] - 16, 64)
            if 'unit_d' in hp.name:
                hp.low = max(params['unit_d'] - 16, 0)
            if 'new_conv' in hp.name:
                hp.low = max(params[hp.name] - 16, abs(int(512 / self.space.epsilon_d)))
            if 'new_fc' in hp.name:
                hp.low = max(params[hp.name] - 16, 0)
    
    def inc_batch_size(self, params):
        """
        method used to increment batch_size
        """
        for hp in self.space:
            if hp.name == 'batch_size':
                hp.low = params['batch_size'] - 1

    def inc_dropout(self, params):
        """
        method used to increment dropout
        """

        # iterate over hyperparameters space
        for hp in self.space:
            # check if dropout is present in the search space and in that case
            # proceed by increasing the lower range
            if 'dr' in hp.name:
                hp.low = min(params[hp.name] - params[hp.name] / 100, 0.0)

    def dec_neurons(self, params):
        """
        method used to decrement the number of neurons
        """
        # itereate over each hyperparameter and if one of these is a convolutional
        # or dense layer increase the upper value of the range
        for hp in self.space:
            if 'unit_c1' in hp.name:
                hp.high = min(params['unit_c1'] + 16, 64)
            if 'unit_c2' in hp.name:
                hp.high = min(params['unit_c2'] + 16, 128)
            if 'unit_d' in hp.name:
                hp.high = min(params['unit_d'] + 16, 2048)
            if 'new_conv' in hp.name:
                try:
                    hp.high = min(params[hp.name] + 16, 512)
                except KeyError:
                    continue
            if 'new_fc' in hp.name:
                try:
                    hp.high = min(params[hp.name] + 16, 2048)
                except KeyError:
                    continue

    def dec_batch_size(self, params):
        """
        method used to decrement batch_size
        """
        for hp in self.space:
            if hp.name == 'batch_size':
                hp.high = params['batch_size'] + 1

    def dec_dropout(self, params):
        """
        method used to decrement dropout
        """
        for hp in self.space:
            if 'dr' in hp.name:
                hp.high = max(params[hp.name] + params[hp.name] / 100, 0.8)

    # ------------------------- System configuration --------------------------

    def new_config(self) -> None:
        """
        Ask the controller to try a different hardware/run configuration.
        """
        self.controller.manage_configuration()
        logger.info("Requested a new hardware/run configuration")

    # ------------------------------- Orchestration ---------------------------

    def repair(self, sym_tuning: Iterable[str], diagnosis: Iterable[str], model: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Execute a sequence of symbolic tuning actions produced by a diagnosis step.

        Args:
            sym_tuning: Iterable of method names (strings) to invoke on this instance.
                        Methods that require `params` should be listed without arguments;
                        this function detects whether to pass `params` or not based on
                        the method name (matching the original behavior).
            diagnosis:  Iterable of human-readable diagnoses parallel to `sym_tuning`.
            model:      The current model object (will be deleted from controller to force rebuild).
            params:     Dict of current parameter values to guide the tuning actions.

        Returns:
            Tuple of (updated_search_space, model). Note that `model` is returned
            unchanged here—the rebuild is assumed to be handled downstream.
        """
        # Drop the existing model object from the controller to avoid stale state.
        if hasattr(self.controller, "model"):
            del self.controller.model

        diag_list = list(diagnosis)
        for i, action_name in enumerate(sym_tuning):
            # Preserve original control flow semantics:
            # - Some methods are called with params, others without.
            needs_params = (
                action_name in {
                    "decr_lr",
                    "inc_lr",
                    "inc_dropout",
                    "inc_neurons",
                    "inc_batch_size",
                    "dec_neurons",
                    "dec_batch_size",
                    "dec_dropout",
                }
            )

            if "X" in action_name:
                print("I have not found problem, but I'll try some new configurations anyway.")
                self.count_no_probs += 1
                break

            # Resolve the method safely (avoid eval)
            method = getattr(self, action_name, None)
            if not callable(method):
                logger.warning("Skipping unknown tuning action: %s", action_name)
                continue

            self.count_no_probs = 0
            readable_diag = diag_list[i] if i < len(diag_list) else "unspecified issue"
            verb = f"self.{action_name}({'params' if needs_params else ''})"
            print(f"I've find {readable_diag} and I'm trying to fix it with {verb}.")  # Keep original print UX

            # Invoke with or without params according to original rule
            if needs_params:
                method(params)
            else:
                method()

        return self.space, model
        