from __future__ import annotations

import os
import re
import json
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np  # for type hints / array-like
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import regularizers as reg
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    BatchNormalization,
    Flatten,
    MaxPooling2D,
    Dropout,
    Input,
    Add
)
from tensorflow.keras.optimizers import Optimizer
try:
    # Keras 3
    from keras.saving import serialize_keras_object, deserialize_keras_object
except Exception:
    # TF/Keras 2.x
    from tensorflow.keras.utils import serialize_keras_object, deserialize_keras_object

from components.search_space import search_space
from components.colors import colors
from flops.flops_calculator import analyze_model
# from components.monitor_model import MonitoredModel, GradientMonitor

from exp_config import load_cfg


# ----------------------------- Utilities -------------------------------------



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
    def __init__(self, optimizer, multiplier, learning_rate=0.001, name="LWLR", **kwargs):
        """
        class wrapper used to add the functionality of the layer wise learning rate
        """
        # checks for the presence of the _HAS_AGGREGATE_GRAD attribute,
        # present since version 2.11 with the introduction of the new optimizer APIs,
        # to determine how to initialize the wrapper instance
        if hasattr(Optimizer, "_HAS_AGGREGATE_GRAD"):
            # wrapper initialization with new APIs, with learning rate stored in an internal slot
            super().__init__(name, **kwargs)
            self._set_hyper("learning_rate", learning_rate)
        else:
            # wrapper initialization with the old API, with the learning rate as an argument
            super().__init__(name=name, **kwargs)

        # storage of the attributes in the wrapper instance
        self._learning_rate = learning_rate #tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        self._optimizer = optimizer
        self._multiplier = multiplier

        self.last_grad_global_norm = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.last_grad_max_norm    = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.last_grad_mean_norm   = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def apply_gradients(self, grads_and_vars, name: str = None, experimental_aggregate_gradients: bool = True):
        updated = []
        per_var_norms = []
        clean_grads = []

        for grad, var in grads_and_vars:
            if grad is not None:
                layer_name = (getattr(var, "path", None) or var.name).split("/")[0]
                scale = self._multiplier.get(layer_name, 1.0)
                g = grad * scale
                updated.append((g, var))
                clean_grads.append(g)
                per_var_norms.append(tf.norm(g))
            else:
                updated.append((grad, var))

        # Compute and save graph
        if clean_grads:
            gnorm = tf.linalg.global_norm(clean_grads)
            gmax  = tf.reduce_max(per_var_norms)
            gmean = tf.reduce_mean(per_var_norms)
            self.last_grad_global_norm.assign(gnorm)
            self.last_grad_max_norm.assign(gmax)
            self.last_grad_mean_norm.assign(gmean)
        else:
            self.last_grad_global_norm.assign(0.0)
            self.last_grad_max_norm.assign(0.0)
            self.last_grad_mean_norm.assign(0.0)

        self._optimizer.learning_rate.assign(self._learning_rate)

        return self._optimizer.apply_gradients(updated)

    # Delegate required slots/config to the base optimizer
    def _create_slots(self, var_list):
        if hasattr(self._optimizer, "_create_slots"):
            self._optimizer._create_slots(var_list)

    def get_config(self):
        # Config del parent (include clipnorm/clipvalue, ecc.)
        cfg = super().get_config()

        # serializza l'optimizer interno in modo Keras-friendly
        base_opt_cfg = serialize_keras_object(self._optimizer)

        # multiplier deve essere JSON-serializzabile
        mult_cfg = {str(k): float(v) if hasattr(v, "__float__") else v
                    for k, v in self._multiplier.items()}

        cfg.update({
            "name": getattr(self, "name", self.__class__.__name__),
            "optimizer": base_opt_cfg,
            "multiplier": mult_cfg
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        # Estrai ciò che serve alla __init__
        name = config.pop("name", "LWLR")

        opt_cfg = config.pop("optimizer", None)
        optimizer = deserialize_keras_object(opt_cfg) if isinstance(opt_cfg, dict) else opt_cfg
        mult = config.pop("multiplier", None)

        lr = config.get("learning_rate", None)

        return cls(optimizer=optimizer,
                   multiplier=mult,
                   learning_rate=lr,
                   name=name,
                   **config)

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
        da: bool,
        reg: bool,
    ) -> None:
        # Store dataset (ensure dtype for Keras)
        # Handle ROI datasets: X may be an object array of dicts with "data" and "pos" keys
        self.train_data = X_train.astype("float32")

        self.test_data = X_test.astype("float32")

        self.train_labels = Y_train
        self.test_labels = Y_test
        
        # Validate labels for None values
        if self.train_labels is None:
            raise ValueError("train_labels is None")
        if self.test_labels is None:
            raise ValueError("test_labels is None")

        self.cfg = load_cfg()
        self.n_classes = n_classes
        self.epochs = self.cfg.epochs

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
        self.nparams: Optional[float] = None

        self.da = da
        self.reg = reg
        self.model = None
    # --------------------------- Build the model -----------------------------


    def _validate_labels(self, train_labels, test_labels):
        """
        Validate that labels are not None and contain valid data.
        Critical for catching issues in ROI depth mode.
        """
        if train_labels is None:
            raise ValueError("train_labels is None - data loading failed in ROI depth mode")
        if test_labels is None:
            raise ValueError("test_labels is None - data loading failed in ROI depth mode")
        
        # Check for object arrays containing None
        if isinstance(train_labels, np.ndarray) and train_labels.dtype == object:
            if any(x is None for x in train_labels.flat):
                raise ValueError("train_labels contains None values")
        
        if isinstance(test_labels, np.ndarray) and test_labels.dtype == object:
            if any(x is None for x in test_labels.flat):
                raise ValueError("test_labels contains None values")

    def build_network(self, params: Dict[str, Any], layer_x_block=2) -> None:
        """
        Define (or reload) the network architecture from params.

        Strategy:
          1) Try to resume from latest dashboard model if present.
          2) Otherwise build a new model based on `params`.
        """
        # 1) clear session
        tf.keras.backend.clear_session()
        batch = self.cfg.dataset == 'tinyimagenet'
        self.model = None
        # 2) Build a new CNN
        input_shape = self.train_data.shape[1:]

        reg_layer = reg.l2() if self.reg else None

        inputs = Input(input_shape)
        if self.da:
            inputs = tf.keras.layers.RandomFlip("horizontal")(inputs)
            inputs = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="nearest")(inputs)

        x = Conv2D(params["unit_c1"] * params['num_neurons'], (3, 3), padding="same")(inputs)
        x = Activation(params["activation"])(x)
        x = BatchNormalization()(x) if batch else x
        for _ in range(1, layer_x_block-1):
            x = Conv2D(params["unit_c1"] * params['num_neurons'], (3, 3), padding="same")(x)
            x = Activation(params["activation"])(x)

        x = Conv2D(params["unit_c1"] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
        x = Activation(params["activation"])(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        for _ in range(layer_x_block-1):
            x = Conv2D(params["unit_c2"] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
            x = Activation(params["activation"])(x)

        x = Conv2D(params["unit_c2"] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
        x = Activation(params["activation"])(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)


        # Dynamically added conv blocks (conv -> act -> conv -> act -> pool -> dropout)
        added_convs = [k for k in params if re.match(r"new_conv_\d+$", k) and params[k] > 0]
        for layer_key in sorted(added_convs, key=lambda s: int(s.split("_")[-1])):  # stable order
            shortcut = x
            for _ in range(layer_x_block-1):
                x = Conv2D(params[layer_key] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
                x = Activation(params["activation"])(x)
                x = BatchNormalization()(x) if batch else x

            x = Conv2D(params[layer_key] * params['num_neurons'], (3, 3), padding="same", kernel_regularizer=reg_layer)(x)
            x = Activation(params["activation"])(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)

        # Dynamically added FC layers
        added_fcs = [k for k in params if re.match(r"new_fc_\d+$", k) and params[k] > 0]
        for layer_key in sorted(added_fcs, key=lambda s: int(s.split("_")[-1])):  # stable order
            x = Dense(params[layer_key] * params['num_neurons'], kernel_regularizer=reg_layer)(x)
            x = Activation(params["activation"])(x)
            x = Dropout(params["dr_f"])(x)

        outputs = Dense(self.n_classes, kernel_regularizer=reg_layer, activation="softmax")(x)

        # Build model with appropriate inputs
        self.model = Model(inputs=inputs, outputs=outputs)
        if self.cfg.verbose > 1:
            self.model.summary()
        if "flops_module" in self.cfg.mod_list:
            # Compute FLOPs (approximate; counts MACs as 2 FLOPs)
            try:
                self.flops = analyze_model(self.model)[0].total_float_ops
                trainableParams = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])
                nonTrainableParams = np.sum([np.prod(v.shape) for v in self.model.non_trainable_weights])
                self.nparams = trainableParams + nonTrainableParams
            except Exception:
                self.flops = None  # FLOPs computation failed; ignore
                self.nparams = None

    # ------------------------------ Training ---------------------------------

    def training(self, params: Dict[str, Any]) -> Tuple[List[float], Dict[str, List[float]], Model]:
        """
        Compile and train the model.

        Args:
            params: Hyperparameters (expects keys like unit_c1, unit_c2, unit_d, activation,
                    dr1_2, dr_f, optimizer (str), learning_rate (float), batch_size (int), [reg]).

        Returns:
            (score, history, model) where:
              - score: [loss, accuracy] from evaluation
              - history: Keras-like history dict
              - model: trained (and reloaded) Keras model
        """
        if self.model is None:
            print("Error: Model is not built.")
            exit(1)
        # Unique id for artifacts of this run
        model_name_id = datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")
        self.last_model_id = model_name_id

        # Persist the (fresh) architecture JSON for later reloading
        model_json_path = f"{self.cfg.name}/Model/model-{model_name_id}.json"
        with open(model_json_path, "w") as json_file:
            json_file.write(self.model.to_json())

        # Try loading previous weights if available (fine-tune / warm start)
        try:
            prev_weights = f"{self.cfg.name}/Weights/weights.h5"
            if os.path.exists(prev_weights):
                self.model.load_weights(prev_weights)
        except Exception:
            pass  # ignore if incompatible

        # TensorBoard callback
        tb_log_dir = f"{self.cfg.name}/log_folder/logs/{model_name_id}"
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
        es = EarlyStopping(monitor="val_accuracy", min_delta=0.005, patience=20, verbose=1,
                            mode="max", restore_best_weights=True)

        # --- Compile ---
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        # --- Train ---
        if "debug" in self.cfg.name:
            history = {k: [] for k in ["loss", "accuracy", "val_loss", "val_accuracy"]}
            for _ in range(self.cfg.epochs):
                history["val_loss"].append(random.uniform(0.1, 0.5))
                history["val_accuracy"].append(random.uniform(0.1, 1.0))
                history["loss"].append(random.uniform(0.1, 0.5))
                history["accuracy"].append(random.uniform(0.1, 1.0))
        else:
            # For ROI datasets, pass pos data as second input
            train_input = self.train_data
            test_input = self.test_data

            # Validate labels before model.fit() - critical for ROI depth mode
            self._validate_labels(self.train_labels, self.test_labels)

            history = self.model.fit(
                train_input, self.train_labels,
                epochs=self.epochs,
                batch_size=int(params["batch_size"]),
                verbose=2,
                validation_data=(test_input, self.test_labels),
                callbacks=[tensorboard, es],
            ).history
        # --- Evaluate ---
        score = self.eval_model()
        # --- Save weights and canonical dashboard model ---
        # weights_tmp = f"{self.cfg.name}/Weights/weights-{model_name_id}.weights.h5"
        # self.model.save_weights(weights_tmp)

        try:
            dash_model_path = f"{self.cfg.name}/dashboard/model/model.keras"
            self.model.save(dash_model_path)

            if not getattr(self.model, "built", False) or self.model.inputs is None:
                self.model = tf.keras.models.load_model(dash_model_path, custom_objects={"LayerWiseLR": LayerWiseLR})
        except:
            pass

        try:
            os.remove(model_json_path)
        except OSError:
            pass

        return score, history, self.model
    
    def eval_model(self):

        if "debug" in self.cfg.name:
            score = [random.uniform(0.1, 0.5), random.uniform(0.1, 1.0)]
        else:
            # For ROI datasets, pass pos data as second input
            test_input = self.test_data
            score = self.model.evaluate(test_input, self.test_labels, verbose=2)
        return score


# ------------------------------ Standalone test ------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from exp_config import create_config_file, set_active_config, load_cfg
    from itertools import product
    
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
        parser.add_argument("--epochs", type=int, default=50,
                            help="Epochs for training")
        parser.add_argument(
            "--mod_list", nargs="+", default=[],
            help="List of active modules (e.g., hardware_module flops_module)"
        )
        parser.add_argument("--dataset", type=str, default="tinyimagenet",
                            help="Dataset name")
        parser.add_argument("--name", type=str, default="tinyimagenet_exp",
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
            "--opt", type=str, default="RS_ruled",
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


    def create_experiment_folders() -> None:
        import sys
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
            
            sys.exit(1)
    
    args = parse_args()
    exp_dir = Path(args.name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = create_config_file(exp_dir, overrides=args.__dict__)

    set_active_config(cfg_path)
    cfg = load_cfg(force=True)
    create_experiment_folders()
    
    X_train, Y_train, X_test, Y_test, n_classes = get_datasets('tinyimagenet')

    params = {'unit_c1': 64, 
              'dr1_2': 0.3, 
              'unit_c2': 64, 
              'unit_d': 2048, 
              'dr_f': 0.5, 
              'learning_rate': 0.001, 
              'batch_size': 64, 
              'optimizer': 'Adamax', 
              'activation': 'swish'}
    
    for da, regl2 in product([False, True], repeat=2):
        print(f"\n=== Testing configuration: da={da}, reg={regl2}===")
        nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes, da=da, reg=regl2)
    
        nn.build_network(params, layer_x_block=2)
        score, history, model = nn.training(params)
        print(f"Test loss: {score[0]}")
        print(f"Test accuracy: {score[1]}")
        del nn
        del model
        del history
        K.clear_session()
