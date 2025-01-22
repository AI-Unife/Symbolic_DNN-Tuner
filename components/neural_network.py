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

# class wrapper used to add the functionality of the layer wise learning rate
class LayerWiseLR(Optimizer):
    def __init__(self, optimizer, multiplier, learning_rate=0.001, name="LWLR", **kwargs):
        if hasattr(optimizer, 'update_step'):
            super().__init__(learning_rate, **kwargs)
        else:
            super().__init__(name, **kwargs)
            self._set_hyper("learning_rate", learning_rate)
        self._optimizer = optimizer
        self._multiplier = multiplier

    # function used to apply a multiplier to a specific layer
    def mul_param(self, param, var):
        # get layer name
        layer_key = var.name.split('/')[0]      
        # if there's a multiplier value associated with the layer, apply it to the parameter
        if layer_key in self._multiplier:
            param *= self._multiplier[layer_key]
        return param
            
    # update step used in keras 3.X
    def update_step(self, grad, var, learning_rate):
        new_lr = self.mul_param(learning_rate, var)
        self._optimizer.update_step(grad, var, new_lr)

    def build(self, var_list):
        super().build(var_list)
        self._optimizer.build(var_list)
        
    # update step used in keras 2.X
    @tf.function
    def _resource_apply_dense(self, grad, var):       
        new_lr = K.eval(self._get_hyper("learning_rate"))
        new_lr = self.mul_param(new_lr, var)
        self._optimizer.learning_rate.assign(new_lr)
        self._optimizer._resource_apply_dense(grad, var)

    def _create_slots(self, var_list):
        super()._create_slots(var_list)
        self._optimizer._create_slots(var_list)
     
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
        self.epochs = 5
        self.last_dense = 0
        self.counter_fc = 0
        self.counter_conv = 0
        self.tot_fc = 3
        self.tot_conv = 6
        self.rgl = False
        self.dense = False
        self.conv = False
        self.dnet = dynamic_net()

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
            return self.from_scratch(self.train_data.shape[1:], self.n_classes, params)

        return model


    def training(self, params, new, new_fc, new_conv, rem_conv, rem_fc, da, space):
        """
        Function for compiling and running training
        :param params, new, new_fc, new_conv, rem_conv, da, space: parameters to indicate a possible operation on the network structure and hyperparameter search space
        :return: training history, trained model and and performance evaluation score 
        """
        # build neural network
        model = self.build_network(params, new)
        try:
            # try adding or removing a layer in the neural network based on the anomalies diagnosis
            if new or new_fc or new_conv or rem_conv:
                # if the flag for the addition of a dense layer is true
                if new_fc:
                    if new_fc[0]:
                        self.dense = True
                        model = self.dnet.insert_fc_section(model, params, new_fc[1])
                # if the flag for the addition of regularization is true
                if new:
                    self.rgl = True
                    self.dense = False
                    model = self.dnet.insert_batch(model, params)
                # if the flag for the addition of a convolutional layer
                if new_conv:
                    if new_conv[0]:
                        self.conv = True
                        self.dense = False
                        self.rgl = False
                        model = self.dnet.insert_conv_section(model, params, new_conv[1])
                # if the flag for the removal of a convolutional layer is true
                if rem_conv:
                    self.conv = False
                    self.dense = False
                    self.rgl = False
                    model = self.dnet.remove_conv_section(model)
                # if the flag for the removal of a dense layer is true
                if rem_fc:
                    self.conv = False
                    self.dense = False
                    self.rgl = False
                    model = self.dnet.remove_fc_section(model)

        except Exception as e:
            print(colors.FAIL, e, colors.ENDC)
        
        # print the structure of the neural network and save it in a json file,
        # using the current time as identifier of the model
        print(model.summary())
        model_name_id = time()
        model_json = model.to_json()
        model_name = "Model/model-{}.json".format(model_name_id)
        with open(model_name, 'w') as json_file:
            json_file.write(model_json)

        # save the id of the model in an attribute, so that it can be used later to save an associated db with the same id
        self.last_model_id = model_name_id

        # try to load a set of weights 
        try:
            model.load_weights("Weights/weights.h5")
        except:
            print("Restart\n")

        # tensorboard logs
        tensorboard = TensorBoard(
            log_dir="log_folder/logs/{}".format(model_name_id))

        # losses, lrs = self.lolr_checking(model, space, params['learning_rate'], params['batch_size'], self.train_data,
        #                                  self.train_labels, self.test_data, self.test_labels)

        # compiling and training
        _opt = params['optimizer'] + "(learning_rate=" + str(params['learning_rate']) + ")"
        opt = eval(_opt)

        # create a dictionary with which to associate model layers with a multiplier (layer wise learning rate)
        multiplier = {}
        # trainable variables names
        new_keras = hasattr(opt, 'update_step')
        trainable = [(layer.path if new_keras else layer.name).split('/')[0] for layer in model.trainable_variables]
        # for each successive variable, i'll have a reduction by a factor of sqrt(2)
        current_mul = 1
        lr_factor = 1.414213
        # iterate over each trainable layer, skipping one (kernel and bias pairs)
        for layer in trainable[::2]:
            # get layer class name
            layer_type = model.get_layer(layer).__class__.__name__
            # if the current layer is a type on which we want to apply a multiplier
            if layer_type in ['Conv2D']:
                multiplier |= {layer : current_mul}
                current_mul /= lr_factor

        opt = LayerWiseLR(opt, multiplier, learning_rate=params['learning_rate'])

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        es1 = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=15, verbose=1, mode='min')
        es2 = EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=15, verbose=1, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-4)

        # if the flag of data augmentation is true
        if da:
            # define a generator in which are present the values of the data augmentation parameters
            datagen = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=True)
            datagen.fit(self.train_data)

            # train the model
            history = model.fit(
                datagen.flow(self.train_data, self.train_labels, batch_size=params['batch_size']), epochs=self.epochs,
                verbose=1, validation_data=(self.test_data, self.test_labels),
                callbacks=[tensorboard, reduce_lr, es1, es2]).history
        else:
            # train the network without data augmentation
            history = model.fit(self.train_data, self.train_labels, epochs=self.epochs, batch_size=params['batch_size'],
                                verbose=1,
                                validation_data=(self.test_data, self.test_labels),
                                callbacks=[tensorboard, reduce_lr, es1, es2]).history

        # evaluates model performance on test data
        score = model.evaluate(self.test_data, self.test_labels)

        # save the neural network weights and then reload them from the same json file you just saved
        # this avoids errors because of the changes in the network structure before training
        weights_name = "Weights/weights-{}.weights.h5".format(model_name_id)
        model.save_weights(weights_name)

        model.save("dashboard/model/model.keras")
        
        f = open("Model/model-{}.json".format(model_name_id))
        mj = json.load(f)
        model_json = json.dumps(mj)
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights("Weights/weights-{}.weights.h5".format(model_name_id))
        return score, history, model


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