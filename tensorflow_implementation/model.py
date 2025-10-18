import json
from random import random
from time import time
from typing import List, Dict, Any

import tensorflow as tf
from keras import layers, Layer, models, Model, regularizers, callbacks, preprocessing
from keras.optimizers import *
from components.colors import colors
from components.neural_network import neural_network
from components.model_interface import LayerSpec, TunerModel, LayerTypes, InsertPosition, Params


class TFModel(TunerModel):

    _from_type_map = {
        LayerTypes.InputLayer: layers.InputLayer,
        LayerTypes.Conv2D: layers.Conv2D,
        LayerTypes.MaxPooling2D: layers.MaxPooling2D,
        LayerTypes.Dropout: layers.Dropout,
        LayerTypes.Dense: layers.Dense,
        LayerTypes.BatchNormalization: layers.BatchNormalization,
        LayerTypes.Flatten: layers.Flatten,
        LayerTypes.ELU: "elu",
        LayerTypes.ReLU: "relu",
        LayerTypes.SeLU: "selu",
        LayerTypes.SiLU: "swish",
        LayerTypes.Softmax: "softmax"
    }
    _to_type_map = {v: k for k, v in _from_type_map.items()}
    _to_type_map[layers.Activation] = "activation"

    def __init__(self, input_shape, params, n_classes, activation_function):
        super(TFModel, self).__init__()
        
        self.input_shape = input_shape
        self.params = params
        self.n_classes = n_classes
        self.activation = activation_function

        inputs = layers.Input(shape=input_shape, batch_size=1)
        x = layers.Conv2D(params['unit_c1'], (3, 3), padding='same')(inputs)
        x = layers.Activation(params['activation'])(x)
        x = layers.Conv2D(params['unit_c1'], (3, 3))(x)
        x = layers.Activation(params['activation'])(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(params['dr1_2'])(x)

        x = layers.Conv2D(params['unit_c2'], (3, 3), padding='same')(x)
        x = layers.Activation(params['activation'])(x)
        x = layers.Conv2D(params['unit_c2'], (3, 3))(x)
        x = layers.Activation(params['activation'])(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(params['dr1_2'])(x)

        x = layers.Flatten()(x)
        x = layers.Dense(params['unit_d'])(x)
        x = layers.Activation(params['activation'])(x)
        x = layers.Dropout(params['dr_f'])(x)
        x = layers.Dense(n_classes)(x)
        x = layers.Activation('softmax')(x)

        self.model = Model(inputs=inputs, outputs=x)
        self.model.build(input_shape=(None, *input_shape))
        
        self.create_specs()
   
    def create_specs(self):
        self.layers = {}
        for layer in self.model.layers:
            self.layers[layer.name] = self.to_spec(layer)

    def get_input_shape(self):
        # fix for keras >= 3.4, the input layer is saved
        # in lists of lists, instead of a single list
        if isinstance(self.model.input, list):
            return self.model.input[0]

        return self.model.input

    def add_layers(self, layers: List[LayerSpec], targets: List[LayerTypes], position: InsertPosition):
        # boolean used to identify which layers in the new architecture can use the old weights
        reuse_weights = True

        x = input_shape = self.get_input_shape()

        # iterate over each layer
        for layer in self.model.layers:
            if self.to_type(layer) == LayerTypes.InputLayer:
                continue

            # if the target matches the searched class or the searched layer name:
            if self.to_type(layer) in targets and position != InsertPosition.After:
                reuse_weights = False

                # add all the layers of the section to the final architecture
                self.add_names(layers)
                for layer_spec in layers:
                    x = self.from_spec(layer_spec)(x)

                if position == InsertPosition.Replace:
                    continue

            if reuse_weights:
                x = layer(x)
            else:
                spec = self.layers[layer.name]
                x = self.from_spec(spec)(x)

            if self.to_type(layer) in targets and position == InsertPosition.After:
                reuse_weights = False

                self.add_names(layers)
                for layer_spec in layers:
                    x = self.from_spec(layer_spec)(x)

        self.model = Model(inputs=input_shape, outputs=x)
        self.create_specs()

    def add_names(self, layer_specs: List[LayerSpec]):
        for layer_spec in layer_specs:
            layer_spec.name = f"{layer_spec.name}_{time()}_{random()}"

    def remove_layers(self, target: LayerSpec, linked_layers: List[LayerTypes], delimiter: bool, first_found: bool):
        removed_names = []
        inside_section = False
        found_section = False
        reuse_weights = True

        x = input_shape = self.get_input_shape()

        for layer in self.model.layers:
            if self.to_type(layer) == LayerTypes.InputLayer:
                continue

            layer_spec = self.layers[layer.name]

            if not inside_section and not found_section:
                # Check if this layer starts a section
                if layer.name == target.name:
                    inside_section = True
                    reuse_weights = False
                    removed_names.append(layer.name)
                    continue

            if inside_section:
                if layer_spec.type not in linked_layers:
                    # End of section
                    inside_section = False
                    found_section = first_found

                    if delimiter:
                        layer_spec = self.layers[layer.name]
                        x = self.from_spec(layer_spec)(x)
                    else:
                        removed_names.append(layer_spec.name)

                    continue

                # Still inside section
                removed_names.append(layer_spec.name)
                continue

            # Normal case: keep this layer
            if reuse_weights:
                x = layer(x)
            else:
                spec = self.layers[layer.name]
                x = self.from_spec(spec)(x)

        print("\n#### removed ####")
        for name in removed_names:
            print(name)

        self.model = Model(inputs=input_shape, outputs=x)
        self.create_specs()

    def from_type(self, layer_type: LayerTypes):
        return self._from_type_map[layer_type]

    def to_type(self, layer: Any):
        if isinstance(layer, layers.Activation):
            return self._to_type_map[layer.get_config()['activation']]
        elif isinstance(layer, layers.Layer):
            return self._to_type_map[layer.__class__]

        return self._to_type_map[layer]

    def to_spec(self, layer: Layer):
        layer_type = self.to_type(layer.__class__)
        if layer_type == LayerTypes.Conv2D:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer,
                params={
                    Params.IN_CHANNELS: layer.input.shape[3],
                    Params.IN_HEIGHT: layer.input.shape[1],
                    Params.IN_WIDTH: layer.input.shape[2],
                    Params.OUT_CHANNELS: layer.output.shape[3],
                    Params.OUT_HEIGHT: layer.output.shape[1],
                    Params.OUT_WIDTH: layer.output.shape[2],
                    Params.KERNEL_SIZE: layer.kernel_size,
                    Params.STRIDE: layer.strides,
                    Params.PADDING: layer.padding,
                    Params.BIAS: layer.use_bias,
                }
            )
        elif layer_type == LayerTypes.Dense:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer,
                params={
                    Params.IN_FEATURES: layer.input.shape[1],
                    Params.OUT_FEATURES: layer.output.shape[1],
                }
            )
        elif layer_type == LayerTypes.MaxPooling2D:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer,
                params={
                    Params.KERNEL_SIZE: layer.pool_size,
                    Params.STRIDE: layer.strides,
                }
            )
        elif layer_type == LayerTypes.Dropout:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer,
                params={
                    Params.DROPOUT_RATE: layer.rate
                }
            )
        elif layer_type == LayerTypes.Flatten:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer
            )
        elif layer_type == "activation":
            activation_function = layer.get_config()["activation"]
            return LayerSpec(
                name=layer.name,
                type=self.to_type(activation_function),
                module=layer,
                is_activation=True,
                params={
                    "activation_function": activation_function
                }
            )
        elif layer_type == LayerTypes.InputLayer:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer,
                params={
                    "input_shape": layer.batch_shape,
                }
            )
        elif layer_type == LayerTypes.BatchNormalization:
            return LayerSpec(
                name=layer.name,
                type=layer_type,
                module=layer
            )
        else:
            raise Exception("Missing LayerSpec for layer of type " + layer.__class__.__name__)

    def from_spec(self, layer_spec: LayerSpec):
        if layer_spec.type == LayerTypes.Conv2D:
            return layers.Conv2D(
                name=layer_spec.name,
                filters=layer_spec.get(Params.OUT_CHANNELS),
                kernel_size=layer_spec.get(Params.KERNEL_SIZE),
                padding="same" if layer_spec.get(Params.PADDING) == 1 else "valid",
                use_bias=layer_spec.get(Params.BIAS)
            )
        elif layer_spec.type == LayerTypes.Dense:
            return layers.Dense(
                name=layer_spec.name,
                units=layer_spec.get(Params.OUT_FEATURES)
            )
        elif layer_spec.type == LayerTypes.MaxPooling2D:
            return layers.MaxPooling2D(
                name=layer_spec.name,
                pool_size=layer_spec.get(Params.KERNEL_SIZE),
            )
        elif layer_spec.type == LayerTypes.Dropout:
            return layers.Dropout(
                name=layer_spec.name,
                rate=layer_spec.get(Params.DROPOUT_RATE)
            )
        elif layer_spec.type == LayerTypes.Flatten:
            return layers.Flatten(
                name=layer_spec.name
            )
        elif layer_spec.type in [LayerTypes.ELU, LayerTypes.Softmax]:
            return layers.Activation(
                name=layer_spec.name,
                activation=self.from_type(layer_spec.type),
            )
        elif layer_spec.type == LayerTypes.InputLayer:
            return layers.InputLayer(
                name=layer_spec.name,
                input_shape=layer_spec.get("input_shape")[1:],
                batch_size=1
            )
        elif layer_spec.type == LayerTypes.BatchNormalization:
            return layers.BatchNormalization(
                name=layer_spec.name,
            )
        else:
            return self.from_type(layer_spec.type)()
