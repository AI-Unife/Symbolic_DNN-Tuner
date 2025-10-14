from __future__ import annotations

import os
import re
import json
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np  # for type hints / array-like
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import regularizers as reg
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
    Dropout,
    Input,
)
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # (left for compatibility)
from tensorflow.keras.models import Sequential

from components.gesture_dataset import gesture_data
from components.search_space import search_space
from components.colors import colors
from components.custom_train import train_model, eval_model
from flops.flops_calculator import analyze_model

import config as cfg


# ----------------------------- Utilities -------------------------------------

def _ensure_dirs() -> None:
    """Create expected experiment folders if they don't exist."""
    for sub in ("Model", "Weights", "log_folder/logs", "dashboard/model"):
        os.makedirs(os.path.join(cfg.NAME_EXP, sub), exist_ok=True)


def _optimizer_from_name(name: str, lr: float) -> Optimizer:
    """
    Build a Keras optimizer instance from a name string safely (no eval).
    Accepts canonical names like 'Adam', 'RMSprop', 'SGD' etc.
    """
    # tf.keras.optimizers.get accepts either a string or a config dict.
    try:
        # Prefer a config so we can set LR explicitly regardless of defaults.
        return tf.keras.optimizers.get({"class_name": name, "config": {"learning_rate": lr}})
    except Exception:
        # Fallback: try to resolve by attribute (case sensitive)
        opt_cls = getattr(tf.keras.optimizers, name, None)
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer '{name}'.")
        return opt_cls(learning_rate=lr)


# ----------------------- Layer-wise LR wrapper -------------------------------

class LayerWiseLR(Optimizer):
    """
    Lightweight wrapper that scales gradients per-layer by a provided multiplier map.

    Notes:
      - Keeps base optimizer's LR synced with internal `_learning_rate` to honor callbacks
        like ReduceLROnPlateau.
      - Compatible with pre/post TF 2.11 Optimizer APIs.
    """

    def __init__(self, optimizer: Optimizer, multiplier: Dict[str, float],
                 learning_rate: float = 0.001, name: str = "LWLR", **kwargs) -> None:
        # New-style API check (TF 2.11+)
        if hasattr(Optimizer, "_HAS_AGGREGATE_GRAD"):
            super().__init__(name, **kwargs)
            self._set_hyper("learning_rate", learning_rate)
        else:
            super().__init__(learning_rate, name, **kwargs)

        self._learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        self._optimizer = optimizer
        self._multiplier = dict(multiplier or {})

    def apply_gradients(self, grads_and_vars, name: Optional[str] = None,
                        experimental_aggregate_gradients: Optional[bool] = True):
        """
        Scale each gradient by a layer-specific multiplier before delegating to the base optimizer.
        """
        updated_grads_and_vars = []
        for grad, var in grads_and_vars:
            if grad is not None:
                # Resolve "layer name" robustly across TF versions
                layer_name = (getattr(var, "path", None) or var.name).split("/")[0]
                scale = self._multiplier.get(layer_name, 1.0)
                updated_grads_and_vars.append((grad * scale, var))
            else:
                updated_grads_and_vars.append((grad, var))

        # Keep the wrapped optimizer's LR synced (e.g., to honor ReduceLROnPlateau updates)
        if hasattr(self._optimizer, "learning_rate"):
            self._optimizer.learning_rate.assign(self._learning_rate)
        return self._optimizer.apply_gradients(updated_grads_and_vars)

    # Delegate required slots/config to the base optimizer
    def _create_slots(self, var_list):
        if hasattr(self._optimizer, "_create_slots"):
            self._optimizer._create_slots(var_list)

    def get_config(self):
        # Minimal config; serialize base optimizer too if needed.
        base_cfg = {}
        if hasattr(self._optimizer, "get_config"):
            base_cfg = self._optimizer.get_config()
        return {"name": self._name, "learning_rate": float(self._learning_rate.numpy()),
                "base_optimizer": base_cfg}


# --------------------------- Main network class ------------------------------

class neural_network:
    """
    Manage the CNN architecture and training lifecycle:
      - Build network from params (and optionally continue from last saved model).
      - Train/evaluate with callbacks, optional data augmentation,
        and optional layer-wise learning rate scaling.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        n_classes: int,
    ) -> None:
        # Store dataset (ensure dtype for Keras)
        self.train_data = X_train.astype("float32")
        self.train_labels = Y_train
        self.test_data = X_test.astype("float32")
        self.test_labels = Y_test

        self.n_classes = n_classes
        self.epochs = cfg.EPOCHS

        # Legacy flags (kept for API compatibility)
        self.last_dense = 0
        self.counter_fc = 0
        self.counter_conv = 0
        self.rgl = False
        self.dense = False
        self.conv = False

        # Populated during training
        self.last_model_id: Optional[str] = None
        self.flops: Optional[float] = None

    # --------------------------- Build the model -----------------------------

    def build_network(self, params: Dict[str, Any]) -> Model:
        """
        Define (or reload) the network architecture from params.

        Strategy:
          1) Try to resume from latest dashboard model if present.
          2) Otherwise build a new model based on `params`.
        """
        _ensure_dirs()

        # 1) Resume if possible
        try:
            model_path = f"{cfg.NAME_EXP}/dashboard/model/model.keras"
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                print("Loaded previous model from dashboard/model.")
                return model
        except Exception:
            # Fall through to rebuild a fresh model if loading fails
            print("Previous model could not be loaded; building a new one.")

        # 2) Build a new CNN
        if (cfg.MODE in ("fwdPass", "hybrid")) and cfg.DATA_NAME == "gesture":
            input_shape = self.train_data.shape[2:]  # (H, W, C) for gesture pipeline
        else:
            input_shape = self.train_data.shape[1:]  # generic (H, W, C)

        reg_layer = reg.l2(params["reg"]) if "reg" in params else None

        inputs = Input(input_shape)
        x = Conv2D(params["unit_c1"], (3, 3), padding="same", kernel_regularizer=reg_layer)(inputs)
        x = Activation(params["activation"])(x)
        x = Conv2D(params["unit_c1"], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
        x = Activation(params["activation"])(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(params["dr1_2"])(x)

        x = Conv2D(params["unit_c2"], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
        x = Activation(params["activation"])(x)
        x = Conv2D(params["unit_c2"], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
        x = Activation(params["activation"])(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(params["dr1_2"])(x)

        # Dynamically added conv blocks (conv -> act -> conv -> act -> pool -> dropout)
        added_convs = [k for k in params if re.match(r"new_conv_\d+$", k)]
        for layer_key in sorted(added_convs, key=lambda s: int(s.split("_")[-1])):  # stable order
            x = Conv2D(params[layer_key], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
            x = Activation(params["activation"])(x)
            x = Conv2D(params[layer_key], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
            x = Activation(params["activation"])(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(params["dr1_2"])(x)

        x = Flatten()(x)
        x = Dense(params["unit_d"])(x)
        x = Activation(params["activation"])(x)
        x = Dropout(params["dr_f"])(x)

        # Dynamically added FC layers
        added_fcs = [k for k in params if re.match(r"new_fc_\d+$", k)]
        for layer_key in sorted(added_fcs, key=lambda s: int(s.split("_")[-1])):  # stable order
            x = Dense(params[layer_key])(x)
            x = Activation(params["activation"])(x)
            x = Dropout(params["dr_f"])(x)

        outputs = Dense(self.n_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if "flops_module" in cfg.MOD_LIST:
            # Compute FLOPs (approximate; counts MACs as 2 FLOPs)
            try:
                self.flops = analyze_model(model)
            except Exception:
                self.flops = None  # FLOPs computation failed; ignore 
        
        return model

    # ------------------------------ Training ---------------------------------

    def training(self, params: Dict[str, Any], da: Optional[bool]) -> Tuple[List[float], Dict[str, List[float]], Model]:
        """
        Compile and train the model.

        Args:
            params: Hyperparameters (expects keys like unit_c1, unit_c2, unit_d, activation,
                    dr1_2, dr_f, optimizer (str), learning_rate (float), batch_size (int), [reg]).
            da:     If True, apply lightweight on-the-fly data augmentation.

        Returns:
            (score, history, model) where:
              - score: [loss, accuracy] from evaluation
              - history: Keras-like history dict
              - model: trained (and reloaded) Keras model
        """
        _ensure_dirs()
        self.model = self.build_network(params)

        # Unique id for artifacts of this run
        model_name_id = datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")
        self.last_model_id = model_name_id

        # Persist the (fresh) architecture JSON for later reloading
        model_json_path = f"{cfg.NAME_EXP}/Model/model-{model_name_id}.json"
        with open(model_json_path, "w") as json_file:
            json_file.write(self.model.to_json())

        # Try loading previous weights if available (fine-tune / warm start)
        try:
            prev_weights = f"{cfg.NAME_EXP}/Weights/weights.h5"
            if os.path.exists(prev_weights):
                self.model.load_weights(prev_weights)
        except Exception:
            pass  # ignore if incompatible

        # TensorBoard callback
        tb_log_dir = f"{cfg.NAME_EXP}/log_folder/logs/{model_name_id}"
        tensorboard = TensorBoard(log_dir=tb_log_dir)

        # --- Optimizer (safe construction, then wrap with LayerWiseLR) ---
        base_opt = _optimizer_from_name(params["optimizer"], float(params["learning_rate"]))

        # Build per-layer multipliers (decay by sqrt(2) across Conv2D feature hierarchy)
        multiplier: Dict[str, float] = {}
        trainable_names = [
            (getattr(var, "path", None) or var.name).split("/")[0]
            for var in self.model.trainable_variables
        ]

        current_mul = 1.0
        lr_factor = 1.414213  # sqrt(2)
        for layer_name in trainable_names[::2]:  # skip bias variables (kernel/bias pairs)
            layer_type = self.model.get_layer(layer_name).__class__.__name__
            if layer_type in {"Conv2D"}:
                multiplier[layer_name] = current_mul
                current_mul /= lr_factor

        opt = LayerWiseLR(base_opt, multiplier, learning_rate=float(params["learning_rate"]))

        # --- Callbacks ---
        es1 = EarlyStopping(monitor="val_loss", min_delta=0.005, patience=30, verbose=1,
                            mode="min", restore_best_weights=True)
        es2 = EarlyStopping(monitor="val_accuracy", min_delta=0.005, patience=30, verbose=1,
                            mode="max", restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=20,
                                      verbose=1, min_lr=1e-4)

        # --- Optional data augmentation ---
        # We prepend a small augmentation pipeline. Wrapping with Sequential is fine here
        # since the base model is purely sequential in topology (functional API).
        if da:
            aug = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="nearest"),
            ])
            self.model = tf.keras.Sequential([aug, *self.model.layers])

        # --- Compile ---
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        # --- Train ---
        if (cfg.MODE in ("fwdPass", "hybrid")) and cfg.DATA_NAME == "gesture":
            if "debug" in cfg.NAME_EXP:
                # Simulated history for debug mode
                history = {k: [] for k in ["loss", "accuracy", "val_loss", "val_accuracy"]}
                for _ in range(cfg.EPOCHS):
                    history["val_loss"].append(random.uniform(0.1, 0.5))
                    history["val_accuracy"].append(random.uniform(0.1, 1.0))
                    history["loss"].append(random.uniform(0.1, 0.5))
                    history["accuracy"].append(random.uniform(0.1, 1.0))
            else:
                history = train_model(
                    self.model, opt,
                    self.train_data, self.train_labels,
                    self.test_data, self.test_labels,
                    self.epochs, params,
                    [tensorboard, reduce_lr, es1, es2]
                )
        else:
            if "debug" in cfg.NAME_EXP:
                history = {k: [] for k in ["loss", "accuracy", "val_loss", "val_accuracy"]}
                for _ in range(cfg.EPOCHS):
                    history["val_loss"].append(random.uniform(0.1, 0.5))
                    history["val_accuracy"].append(random.uniform(0.1, 1.0))
                    history["loss"].append(random.uniform(0.1, 0.5))
                    history["accuracy"].append(random.uniform(0.1, 1.0))
            else:
                history = self.model.fit(
                    self.train_data, self.train_labels,
                    epochs=self.epochs,
                    batch_size=int(params["batch_size"]),
                    verbose=2,
                    validation_data=(self.test_data, self.test_labels),
                    callbacks=[tensorboard, reduce_lr, es1, es2],
                ).history

        # --- Evaluate ---
        if (cfg.MODE in ("fwdPass", "hybrid")) and cfg.DATA_NAME == "gesture":
            if "debug" in cfg.NAME_EXP:
                score = [random.uniform(0.1, 0.5), random.uniform(0.1, 1.0)]
            else:
                score = eval_model(self.model, self.test_data, self.test_labels)
        else:
            if "debug" in cfg.NAME_EXP:
                score = [random.uniform(0.1, 0.5), random.uniform(0.1, 1.0)]
            else:
                score = self.model.evaluate(self.test_data, self.test_labels, verbose=2)

        # --- Save weights and canonical dashboard model ---
        weights_tmp = f"{cfg.NAME_EXP}/Weights/weights-{model_name_id}.weights.h5"
        self.model.save_weights(weights_tmp)

        dash_model_path = f"{cfg.NAME_EXP}/dashboard/model/model.keras"
        try:
            self.model.save(dash_model_path)
        except Exception:
            # If the wrapper/augmentation created a non-serializable wrapper, we still
            # preserve weights and architecture JSON; downstream code can rebuild.
            pass

        # --- Rebuild model from JSON + reload weights (cleans the graph/session coupling) ---
        with open(model_json_path, "r") as f:
            mj = json.load(f)
        self.model = tf.keras.models.model_from_json(json.dumps(mj))
        self.model.load_weights(weights_tmp)

        # Cleanup temp artifacts
        try:
            os.remove(weights_tmp)
        except OSError:
            pass
        try:
            os.remove(model_json_path)
        except OSError:
            pass

        return score, history, self.model


# ------------------------------ Standalone test ------------------------------

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test, n_classes = gesture_data()

    # Example: evaluate a saved best model (if present)
    try:
        model = tf.keras.models.load_model(
            f"{'25_03_04_12_35_fwdPass_gesture_accuracy_module_100_20'}/Model/best-model.keras"
        )
        print(model.summary())
        start = datetime.now()
        score = eval_model(model, X_test, Y_test)
        print("Time fwdPass: ", datetime.now() - start)
        print("Accuracy: ", score[1], "Loss: ", score[0])
    except Exception as e:
        print("Could not load first best model:", e)

    print("----------------------------------------------------")

    try:
        model = tf.keras.models.load_model(
            f"{'25_03_04_12_03_depth_gesture_accuracy_module_100_20'}/Model/best-model.keras"
        )
        start = datetime.now()
        score = model.evaluate(X_test, Y_test)
        print("Time depth: ", datetime.now() - start)
        print("Accuracy: ", score[1], "Loss: ", score[0])
    except Exception as e:
        print("Could not load second best model:", e)
