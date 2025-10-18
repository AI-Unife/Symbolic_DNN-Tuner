from typing import List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from time import time
from random import random
import importlib

from keras.applications.vgg16 import VGG16

from components.model_interface import TunerModel, LayerTypes, LayerSpec, InsertPosition

class DynamicNet:

    def remove_section(self, tuner_model: TunerModel, target: LayerSpec, linked_layers: List[LayerTypes], delimiter: bool, first_found: bool):
        model.remove_layers(target, linked_layers, delimiter, first_found)
        return model

    def insert_section(self, tuner_model: TunerModel, n_section: int, new_section: List[LayerSpec], position: InsertPosition,
                       targets: List[LayerTypes]):
        """
        Insert a new section into the model.

        :param tuner_model: The original nn.Sequential model
        :param n_section: Number of times to replicate the new_section
        :param new_section: List of layers to insert
        :param position: Where to insert the new section ('before', 'after', or 'replace')
        :param targets: The name or class of the layer to target
        :return: Modified nn.Sequential model
        """

        extended_new_section = []
        for _ in range(n_section):
            extended_new_section.extend(new_section)

        tuner_model.add_layers(extended_new_section, targets, position)
        return tuner_model

    def get_last_section(self, model: TunerModel, layer_type: LayerTypes):
        """
        Method to find the index of the first layer in the last section of a specific type.
        :param model: PyTorch nn.Sequential model to search
        :param type_class: Class of the layer type to search for (e.g., torch.nn.Conv2d)
        :return: Index of the layer where the last section begins
        """
        target_spec = None

        for layer_spec in reversed(model.layers.values()):
            if layer_type == layer_spec.type:
                target_spec = layer_spec
            elif target_spec is not None:
                break

        return target_spec

    def count_layer_type(self, model: TunerModel, layer_type: LayerTypes):
        """
        Count how many layers of a certain type are in the PyTorch model.
        :param model: PyTorch model in which to count the number of 'layer_type' layers.
        :param layer_type: The layer type (e.g., nn.Conv2d, nn.Linear) to count.
        :return: Number of layers of the specified type.
        """
        return sum(1 for layer_spec in model.layers.values() if layer_spec.type == layer_type)


if __name__ == '__main__':

    # instantiate the class
    dynamicNet = DynamicNet()

    # load VGG16
    model = VGG16(weights='imagenet')
    model.summary()

    # remove al layers
    linked_list = ['Conv2D', 'MaxPooling2D', 'Flatten', 'Dense']
    model = dynamicNet.remove_section(model, 'Conv2D', linked_list, False, False)
    # add 5 maxpool
    model = dynamicNet.insert_section(model, 5, [MaxPooling2D()], 'after', model.layers[0].name)
    # add two conv before each maxpool
    new_section = [Conv2D(256, (2,2), padding="same"), Activation('relu')]
    model = dynamicNet.insert_section(model, 2, new_section, 'before', 'MaxPooling2D')
    # add after last maxpool dense section
    last_max = dynamicNet.get_last_section(model, 'MaxPooling2D')
    new_section = [Flatten(), Dense(256), Dense(10)]
    model = dynamicNet.insert_section(model, 1, new_section, 'after', last_max)
    model.summary()

    print(dynamicNet.any_batch(model))
    new_section = [BatchNormalization(), AveragePooling2D((2,2))]
    model = dynamicNet.insert_section(model, 1, new_section, 'replace', 'MaxPooling2D')
    model.summary()

    # remove last convolutional section
    last_conv = dynamicNet.get_last_section(model, 'Conv2D')
    linked_list = ['Conv2D', 'Activation', 'MaxPooling2D']
    model = dynamicNet.remove_section(model, last_conv, linked_list, True, False)
    last_conv = dynamicNet.get_last_section(model, 'Conv2D')
    model = dynamicNet.remove_section(model, last_conv, linked_list, True, False)
    model.summary()
    quit()

    new_section = [Conv2D(256, (2,2), padding="same"), Activation('relu')]
    model = dynamicNet.insert_section(model, 2, new_section, 'after', 'Input')
    last_act = dynamicNet.get_last_section(model, 'Activation')
    model = dynamicNet.insert_section(model, 1, [MaxPooling2D()], 'after', last_act)
    model.summary()
    quit()

    # replace all MaxPool layers with BatchNorm and AveragePool
    new_section = [BatchNormalization(), AveragePooling2D((2,2))]
    model = dynamicNet.insert_section(model, 2, new_section, 'replace', 'MaxPooling2D')
    model.summary()

    # remove all Dense layers
    model = dynamicNet.remove_section(model, 'Dense', [], False, False)
    model.summary()

    # add a new section after flatten layer made up of a Dense layer, an activation and Dropout
    new_section = [Dense(50), Activation('relu'), Dropout(0.1)]
    model = dynamicNet.insert_section(model, 1, new_section, 'after', 'Flatten')
    model.summary()

    # add a new convolutional section before the last one
    new_section = [Conv2D(1024, (2,2)), Conv2D(1024, (2,2)), MaxPooling2D()]
    last_conv = dynamicNet.get_last_section(model, 'Conv2D')
    model = dynamicNet.insert_section(model, 1, new_section, 'before', last_conv)
    model.summary()

    # remove a section that starts from 'block5_conv1' with all associated layers in linked_section
    linked_section = ['Conv2D', 'BatchNormalization', 'AveragePooling2D']
    model = dynamicNet.remove_section(model, 'block5_conv1', linked_section, True, False)
    model.summary()