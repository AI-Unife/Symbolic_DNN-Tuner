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

        # Internal counters / caps
        self.count_lr = 0
        self.count_da = 0
        self.count_br = 0
        self.count_new_fc = 0
        self.count_new_cv = 0
        self.max_fc = 5
        self.max_conv = self.count_max_conv()
        self.start_conv, self.start_fc = self.ss.count_initial_layers(self.space)
        self.count_no_probs = 0

    # ----------------------------- Utilities ---------------------------------

    def _iter_hparams(self) -> Iterable[Any]:
        """Shorthand to iterate the search space hyperparameters."""
        return self.space

    def _safe_param_get(self, params: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Fetch a param by key with a safe default to avoid KeyErrors."""
        return params.get(key, default)

    def count_max_conv(self) -> int:
        """
        Heuristic upper bound for how many conv+pool blocks fit given the input size.

        The estimate assumes a pattern:
            Conv2D(kernel=3x3, stride=1, valid padding) -> MaxPool2D(2x2)
        and stops before spatial dims drop below 3x3.

        Returns:
            int: Maximum number of stacked (conv+pool) sections that can fit.
        """
        # Expecting X_train to be an image-like array with shape [H, W, ...] or [N, H, W, ...].
        shape = getattr(self.controller, "X_train", None)
        if shape is None or not hasattr(shape, "shape"):
            logger.warning("controller.X_train not found or missing shape; defaulting max_conv to 0")
            return 0

        # Support either (H, W, C) or (N, H, W, C)
        dims = shape.shape
        if len(dims) >= 2 and len(dims) <= 3:
            h, w = dims[0], dims[1]
        elif len(dims) >= 4:
            h, w = dims[1], dims[2]
        else:
            logger.warning("Unexpected X_train shape %s; defaulting max_conv to 0", dims)
            return 0

        max_layers = 0
        while h >= 3 and w >= 3:
            # After a valid 3x3 conv: size shrinks by 2 each dim
            h -= 2
            w -= 2
            if h < 1 or w < 1:
                break

            # Then a 2x2 maxpool (floor division by 2)
            h //= 2
            w //= 2

            if h < 3 or w < 3:
                break

            max_layers += 1

        return max_layers

    # --------------------------- Regularization ------------------------------

    def reg_l2(self) -> None:
        """
        Enable L2 regularization (and potentially batch norm) once.

        Idempotent-ish: applied only the first time it's requested.
        """
        self.count_br += 1
        if self.count_br <= 1:
            self.controller.set_case(True)
            new_p = {"reg": 1e-4}
            self.space = self.ss.add_params(new_p)
            logger.info("Enabled L2 regularization with reg=1e-4")

    # -------------------------- Architecture edits ---------------------------

    def new_fc_layer(self) -> None:
        """
        Add a fully-connected (dense) layer, respecting a soft upper bound to avoid bloat.
        """
        if self.count_new_fc + self.start_fc > self.max_fc:
            print(
                colors.FAIL,
                "Max number of dense layers reached",
                colors.ENDC,
            )
            print(
                colors.FAIL,
                "start dense layers: ",
                self.start_fc,
                " Dense layers: ",
                self.count_new_fc + self.start_fc,
                " Max dense layers: ",
                self.max_fc,
                colors.ENDC,
            )
            return

        self.count_new_fc += 1
        self.controller.add_fc_layer(True, self.count_new_fc)
        new_p = {f"new_fc_{self.count_new_fc}": 512}
        self.space = self.ss.add_params(new_p)
        logger.info("Added dense layer #%d with 512 units", self.count_new_fc)

    def new_conv_layer(self) -> None:
        """
        Add a convolutional *section* (e.g., conv block). Respect input-size constraints.
        """
        if not self.max_conv:
            return

        tot_conv = self.start_conv + self.count_new_cv * 2
        if tot_conv > self.max_conv:
            print(colors.FAIL, "Max number of convolutional layers reached", colors.ENDC)
            print(
                colors.FAIL,
                "Convolutional layers: ",
                tot_conv - 2,
                " Max convolutional layers: ",
                self.max_conv,
                colors.ENDC,
            )
            return

        self.count_new_cv += 1
        new_p = {f"new_conv_{self.count_new_cv}": 512}
        self.space = self.ss.add_params(new_p)
        self.controller.add_conv_section(True, self.count_new_cv)
        logger.info("Added conv section #%d with 512 filters", self.count_new_cv)

    def dec_layers(self) -> None:
        """
        Remove one convolutional section if present.
        """
        print("Removing convolutional layer")
        print(self.count_new_cv)
        if self.count_new_cv < 1:
            print(colors.FAIL, "No more convolutional layers to remove", colors.ENDC)
            self.count_new_cv = 0
            return

        self.count_new_cv -= 1
        # Remove param key associated with the *next* (now absent) conv
        new_p = {f"new_conv_{self.count_new_cv}": 512}
        self.space = self.ss.remove_params(new_p)
        self.controller.remove_conv_section(True)
        logger.info("Removed one conv section; remaining added sections: %d", self.count_new_cv)

    def dec_fc(self) -> None:
        """
        Remove one dense layer if present.
        """
        print("Removing dense layer")
        print(self.count_new_fc)
        if self.count_new_fc < 1:
            print(colors.FAIL, "No more dense layers to remove", colors.ENDC)
            self.count_new_fc = 0
            return

        self.count_new_fc -= 1
        new_p = {f"new_fc_{self.count_new_fc}": 512}
        self.space = self.ss.remove_params(new_p)
        self.controller.remove_fully_connected(True)
        logger.info("Removed one dense layer; remaining added layers: %d", self.count_new_fc)

    # -------------------------- Data augmentation ----------------------------

    def data_augmentation(self) -> None:
        """
        Enable data augmentation in the controller.
        """
        self.controller.set_data_augmentation(True)
        logger.info("Enabled data augmentation")

    # -------------------------- Hyperparam tweaks ----------------------------

    def decr_lr(self, params):
        for hp in self._iter_hparams():
            if hp.name == "learning_rate":
                hp.low  = max(hp.low * 0.5, 1e-7)
                hp.high = max(hp.low * 4, min(hp.high * 0.5, 1e-1))  # mantieni ordine low<high

    def inc_lr(self, params):
        for hp in self._iter_hparams():
            if hp.name == "learning_rate":
                hp.low  = min(hp.low * 2.0, 1e-1)
                hp.high = min(hp.high * 2.0, 1.0)

    def inc_dropout(self, params):
        self.controller.set_data_augmentation(False)
        for hp in self._iter_hparams():
            if "dr" in hp.name:
                base = self._safe_param_get(params, hp.name, hp.low)
                hp.low = min(max(base + 0.05, 0.0), 0.8)  # evita over-regularization >0.8
                hp.high = max(hp.high, hp.low + 0.05)

    def inc_neurons(self, params):
        for hp in self._iter_hparams():
            if any(k in hp.name for k in ["unit_c1", "unit_c2", "unit_d", "new_conv", "new_fc"]):
                hp.low = int(max(hp.low * 1.2, hp.low + 16))
                hp.high = max(hp.high, hp.low + 16)

    def dec_neurons(self, params):
        for hp in self._iter_hparams():
            if any(k in hp.name for k in ["unit_c1", "unit_c2", "unit_d", "new_conv", "new_fc"]):
                hp.high = int(max(hp.low + 8, hp.high / 1.25))

    def inc_batch_size(self, params):
        for hp in self._iter_hparams():
            if hp.name == "batch_size":
                base = int(self._safe_param_get(params, "batch_size", max(8, hp.low)))
                new_low = max(8, int(base * 1.5))
                new_high = max(new_low + 16, int(hp.high * 1.25), 128)
                hp.low, hp.high = new_low, min(new_high, 512)

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
                action_name not in {
                    "reg_l2",
                    "data_augmentation",
                    "new_fc_layer",
                    "new_conv_layer",
                    "dec_layers",
                    "dec_fc",
                    "new_config",
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