import re
import os
from time import time
from typing import Any

import numpy as np
from pytest import param
import json
from components.colors import colors

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers as reg
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input,
                                     BatchNormalization)
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from components.dataset import cifar_data
from components.LOLR import Lolr
from components.model_interface import LayerTypes, Params, LayerSpec, InsertPosition, TunerModel
from components.search_space import search_space
from components.dynamic_net import DynamicNet

from abc import ABC, abstractmethod
     
class neural_network (ABC):
    """
    class used for the management of the neural network architecture,
    offering methods for training the dnn and adding and removing convolutional layers
    """
    def __init__(self, X_train, Y_train, X_test, Y_test, n_classes):
        """
        initialized the attributes of the class.
        first part is used for storing the examples of the dataset,
        the second part to keep track of the number of the various parts of the dnn,
        for example the number of convolutional or dense layers.
        """
        self.train_data = X_train
        self.train_labels = Y_train
        self.test_data = X_test
        self.test_labels = Y_test
        self.train_data = self.train_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
        self.train_data /= 255
        self.test_data /= 255
        self.n_classes = n_classes
        self.epochs = 2
        self.last_dense = 0
        self.counter_fc = 0
        self.counter_conv = 0
        self.tot_fc = 3
        self.tot_conv = 6
        self.rgl = False
        self.dense = False
        self.conv = False
        self.dnet = DynamicNet()

    @abstractmethod
    def from_checkpoint(checkpoint):
        """
        Receives the json object loaded from file
        """
        raise NotImplementedError
    
    @abstractmethod
    def from_scratch(input_shape, n_classes, params):
        raise NotImplementedError

    def insert_conv_section(self, model: TunerModel, params: Any, n_conv: int):
        """
        Inserts a convolutional section into a PyTorch model.
        :param model: PyTorch model to modify.
        :param params: Dictionary of parameters for the new section.
        :param n_conv: Position at which to insert the new convolutional section.
        :return: Modified model with the new conv section added.
        """
        # Count the current number of convolutional layers in the model
        current_conv_count = self.dnet.count_layer_type(model, LayerTypes.Conv2D)

        # Initialize the counter of new convolutions and the list of new layers
        new_conv_count = 0
        new_section = []

        # Cycle to add at most two convolutions in the new conv section
        for i in range(2):
            # Increase the counter to add a new convolutional layer
            new_conv_count += 1

            # If the sum of current and new convolutions exceeds the limit, stop adding layers
            if (new_conv_count + current_conv_count) > self.tot_conv:
                break

            # Add a convolutional layer and its activation
            new_section.extend([
                LayerSpec(
                    type=LayerTypes.Conv2D,
                    params={
                        Params.IN_CHANNELS: params['unit_c2'],
                        Params.OUT_CHANNELS: params['unit_c2'],
                        Params.KERNEL_SIZE: 3,
                        Params.PADDING: 1
                    }
                ),
                LayerSpec(
                    type=model.to_type(self.activation_map[params['activation']]),
                    is_activation=True,
                    params={
                        Params.ACTIVATION: self.activation_map[params['activation']]
                    }
                )
            ])

            # If batch normalization exists in the model, add it to the new convolutional section
            if self.dnet.count_layer_type(model, LayerTypes.BatchNormalization2D):
                new_section.append(
                LayerSpec(
                    type=LayerTypes.BatchNormalization,
                    params={
                        Params.NUM_FEATURES: params['unit_c2']
                    }
                )
            )

        # If the new section is empty, return the original model
        if not new_section:
            return model

        # Add max pooling and dropout to the convolutional section
        new_section.extend([
            LayerSpec(
                type=LayerTypes.MaxPooling2D,
                params={
                    Params.KERNEL_SIZE: 2
                }
            ),
            LayerSpec(
                type=LayerTypes.Dropout,
                params={
                    Params.DROPOUT_RATE: params['dr_f']
                }
            )
        ])

        # Insert the new section before the Flatten layer
        return self.dnet.insert_section(model, n_conv, new_section, InsertPosition.Before, [LayerTypes.Flatten])


    @abstractmethod
    def insert_batch(self, model, params):
        raise NotImplementedError

    def insert_fc_section(self, model, params, n_fc):
        """
        Method for inserting a fully connected section into a PyTorch model.

        :param model: The PyTorch model to modify.
        :param params: Dictionary with the parameters for the new section, including:
                       - 'new_fc': Number of units in the new fully connected layer.
                       - 'activation': Activation function as a string (e.g., 'relu', 'sigmoid').
                       - 'dr_f': Dropout rate (e.g., 0.5 for 50% dropout).
        :param n_fc: Position index where the new section should be inserted.
        :return: Modified model with the new fully connected section.
        """
        # Check if the number of dense layers in the model exceeds or equals the maximum allowed
        if self.dnet.count_layer_type(model, LayerTypes.Dense) >= self.tot_fc:
            return model

        # Build the new fully connected section
        activation = self.activation_map[params['activation']]
        new_section = [
            LayerSpec(
                type=LayerTypes.Dense,
                params={
                    Params.IN_FEATURES: params['new_fc'],
                    Params.OUT_FEATURES: params['new_fc'],
                }
            ),
            LayerSpec(
                type=model.to_type(activation),
                is_activation=True,
                params={
                    Params.ACTIVATION: activation
                }
            )
        ]

        # If batch normalization is already in the model, add it to the new section
        if (self.dnet.count_layer_type(model, LayerTypes.BatchNormalization1D) > 1 or
                self.dnet.count_layer_type(model, LayerTypes.BatchNormalization2D) > 1):
            new_section.append(
                LayerSpec(
                    type=LayerTypes.BatchNormalization,
                    params={
                        Params.NUM_FEATURES: params['new_fc']
                    }
                )
            )

        # Add dropout layer
        new_section.append(
            LayerSpec(
                type=LayerTypes.Dropout,
                params={
                    Params.DROPOUT_RATE: params['dr_f']
                }
            )
        )

        # Insert the new section at the specified position
        return self.dnet.insert_section(model, n_fc, new_section, InsertPosition.After, [LayerTypes.Flatten])

    def remove_conv_section(self, model):
        """
        Method to remove a convolutional section from a PyTorch model.
        :param model: PyTorch model from which to remove the convolutional section
        :return: model without the convolutional section
        """
        # Check the number of Conv2D layers
        current_conv_count = self.dnet.count_layer_type(model, LayerTypes.Conv2D)
        if current_conv_count <= 1:
            return model

        # Get the first layer of the last convolutional section
        layer_spec = self.dnet.get_last_section(model, LayerTypes.Conv2D)

        # Define the layers associated with the convolutional section
        linked_section = [LayerTypes.Conv2D, LayerTypes.ELU,
                          LayerTypes.BatchNormalization2D]  # TODO: activation should be the one in params

        # If the number of convolutions is odd, include additional layers to remove
        if (current_conv_count % 2) == 1:
            linked_section += [LayerTypes.MaxPooling2D, LayerTypes.Dropout]

        # Use the helper function to remove the section
        model.remove_layers(layer_spec, linked_section, True, True)
        return model

    def remove_fc_section(self, model):
        """
        Method used for removing a dense (fully connected) section.
        :param model: model from which to remove the dense section
        :return: model without the dense section
        """

        # If the number of dense (Linear) layers is less than or equal to 2,
        # specifically a dense layer after the flatten and the output,
        # don't remove any dense layer and return the model
        if self.dnet.count_layer_type(model, LayerTypes.Dense) <= 1:
            return model

        # Remove the first dense section in the model and all associated layers in linked_section
        linked_section = [LayerTypes.ELU, LayerTypes.BatchNormalization1D, LayerTypes.Dropout] #TODO: activation should be the one in params

        target = None
        for layer_spec in model.layers.values():
            if layer_spec.type == LayerTypes.Dense:
                target = layer_spec
                break

        model.remove_layers(target, linked_section, True, True)
        return model
    
    def build_network(self, params, new):
        """
        Function to define the network structure
        :param params new: network layer parameters
        :return: built model
        """

        # TODO: This stays here until i fix loading from saved model
        return self.from_scratch(self.train_data.shape[1:], self.n_classes, params)
        
        try:
            checkpoints_dir = "Model"
            checkpoints = os.listdir(checkpoints_dir)
            checkpoints.sort()

            latest_checkpoint = os.path.join(checkpoints_dir, checkpoints[-1])

            with open(latest_checkpoint, 'r') as f:
                checkpoint = json.load(f)

            print("MODELLO PRECEDENTE")

            return self.from_checkpoint(checkpoint)

        except:
            return self.from_scratch(self.train_data.shape[1:], self.n_classes, params)

    @abstractmethod
    def training(self, params, new, new_fc, new_conv, rem_conv, rem_fc, da, space):
        """
        Function for compiling and running training
        :param params, new, new_fc, new_conv, rem_conv, da, space: parameters to indicate a possible operation on the network structure and hyperparameter search space
        :return: training history, trained model and and performance evaluation score 
        """
        raise NotImplementedError


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
    ss = search_space()
    space = ss.search_sp()

    # default_params = {'unit_c1': 57, 'dr1_2': 0.1256695669329692, 'unit_c2': 124, 'unit_d': 315,
    #                   'dr_f': 0.045717734023783346, 'learning_rate': 0.08359864897019328, 'batch_size': 252,
    #                   'optimizer': 'RMSProp', 'activation': 'elu', 'reg': 0.05497168445820486, 'new_fc': 497}

    default_params = {'unit_c1': 32, 'dr1_2': 0.1256695669329692, 'unit_c2': 64, 'unit_d': 315,
                      'dr_f': 0.045717734023783346, 'learning_rate': 0.08359864897019328, 'batch_size': 252,
                      'optimizer': 'RMSProp', 'activation': 'elu', 'reg': 0.05497168445820486, 'new_fc': 497}

    n = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
    
    model = n.build_network(default_params, None)
    model.summary()

    n.rgl = True
    model = n.insert_batch(model, default_params)
    model.summary()

    model = n.insert_conv_section(model, default_params, 1)
    model.summary()

    model = n.remove_conv_section(model)
    model.summary()

    model = n.insert_fc_section(model, default_params, 1)
    model.summary()

    model = n.remove_fc_section(model)
    model.summary()
    quit()

    new_model = n.remove_conv_layer(model, default_params)
    model_name_id = time()
    model_json = new_model.to_json()
    model_name = "Model/model-{}.json".format(model_name_id)
    with open(model_name, 'w') as json_file:
                json_file.write(model_json)
    
    print(new_model.summary())
    
    # model2 = n.build_network(default_params, None)
    # print(model2.summary())
    
    n.conv = True
    new_model3 = n.insert_layer(
        new_model, '.*flatten.*', default_params, num_cv=1, position='before')

    print(new_model3.summary())


    # score, history, model = n.training(default_params, True, [True, 1], None, None, space)

    # f2 = open("algorithm_logs/history.txt", "w")
    # f2.write(str(history))
    # f2.close()