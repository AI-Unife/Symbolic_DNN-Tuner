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
        tuner_model.remove_layers(target, linked_layers, delimiter, first_found)
        return tuner_model

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
