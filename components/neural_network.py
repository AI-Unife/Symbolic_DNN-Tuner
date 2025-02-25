import re
import os
from time import time
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
from components.search_space import search_space
from components.dynamic_net import dynamic_net

from abc import ABC, abstractmethod
     
class neural_network (ABC):
    """
    class used for the management of the neural network architecture,
    offering methods for training the dnn and adding and removing convolutional layers
    """
    def __init__(self, dynamic_net_class, X_train, Y_train, X_test, Y_test, n_classes):
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
        self.epochs = 5
        self.last_dense = 0
        self.counter_fc = 0
        self.counter_conv = 0
        self.tot_fc = 3
        self.tot_conv = 6
        self.rgl = False
        self.dense = False
        self.conv = False
        self.dnet = dynamic_net_class()

    @abstractmethod
    def from_checkpoint(checkpoint):
        """
        Receives the json object loaded from file
        """
        raise NotImplementedError
    
    # TODO: add comments for following methods and double check methods in other files for comments
    @abstractmethod
    def from_scratch(input_shape, n_classes, params):
        raise NotImplementedError

    @abstractmethod
    def insert_conv_section(self, model, params, n_conv):
        raise NotImplementedError

    @abstractmethod
    def insert_batch(self, model, params):
        raise NotImplementedError

    @abstractmethod
    def insert_fc_section(self, model, params, n_fc):
        raise NotImplementedError

    @abstractmethod
    def remove_conv_section(self, model):
        raise NotImplementedError

    @abstractmethod
    def remove_fc_section(self, model):
        raise NotImplementedError
    
    # TODO: fix this: currently returns either pytorch_implementation.model.Model or keras.Model
    def build_network(self, params, new):
        """
        Function to define the network structure
        :param params new: network layer parameters
        :return: built model
        """
        
        try:
            checkpoints_dir = "Model"
            checkpoints = os.listdir(checkpoints_dir)
            checkpoints.sort()

            latest_checkpoint = os.path.join(checkpoints_dir, checkpoints[-1])

            with open(latest_checkpoint, 'r') as f:
                checkpoint = json.load(f)

            print("MODELLO PRECEDENTE")

            model = self.from_checkpoint(checkpoint)

        except:
            model = self.from_scratch(self.train_data.shape[1:], self.n_classes, params)

        return model

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