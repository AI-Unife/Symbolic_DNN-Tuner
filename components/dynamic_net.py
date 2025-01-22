import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from time import time
from random import random
import importlib

from keras.applications.vgg16 import VGG16

from abc import ABC, abstractmethod

class dynamic_net(ABC):

    @abstractmethod
    def remove_section(self, model, target, linked_layers, delimiter, first_found):
        raise NotImplementedError

    @abstractmethod
    def insert_section(self, model, n_section, new_section, position, target):
        raise NotImplementedError

    @abstractmethod
    def build_model(self, model, model_dict):
        raise NotImplementedError

    @abstractmethod
    def model_from_dict(self, model, model_dict):
        raise NotImplementedError

    @abstractmethod
    def add_names(self, layer_list):
        raise NotImplementedError

    @abstractmethod
    def get_last_section(self, model, type_class):
        raise NotImplementedError

    @abstractmethod
    def all_layers(self, layer_list):
        raise NotImplementedError

    @abstractmethod
    def count_layer_type(self, model, type):
        raise NotImplementedError

    @abstractmethod
    def any_batch(self, model):
        raise NotImplementedError

if __name__ == '__main__':

    # instantiate the class
    dynamicNet = dynamic_net()

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