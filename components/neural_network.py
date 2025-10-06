import re
import os
from time import time
from datetime import datetime
import numpy as np
from pytest import param
import json

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import regularizers as reg
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input,
                                     BatchNormalization)
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

from components.gesture_dataset import gesture_data
from components.search_space import search_space
from components.colors import colors
from components.custom_train import train_model, eval_model

import random

import config as cfg

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
            super().__init__(learning_rate, name, **kwargs)

        # storage of the attributes in the wrapper instance
        self._learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32) 
        self._optimizer = optimizer
        self._multiplier = multiplier

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        # initialization of the list containing the new variable-gradient pairs
        updated_grads_and_vars = []
        # iterates over each pair
        for grad, var in grads_and_vars:
            if grad is not None:
                # get the current layer name, separating it from its type based on keras version
                layer_name = (var.path if hasattr(var, 'path') else var.name).split('/')[0]
                # apply the multiplier to gradient based on current layer
                # if no multiplier is associated, applies 1 as default value
                scaled_grad = grad * self._multiplier.get(layer_name, 1.0)
                # add the updated pair to the list
                updated_grads_and_vars.append((scaled_grad, var))
            else:  
                updated_grads_and_vars.append((grad, var))
        # synchronize the learning rate of the base optimizer with that of the wrapper,
        # in order to apply changes due to callbacks, and then applies gradients
        self._optimizer.learning_rate.assign(self._learning_rate)
        return self._optimizer.apply_gradients(updated_grads_and_vars)
        
    def _create_slots(self, var_list):
        # call the creation of the internal slots of the base optimizer
        self._optimizer._create_slots(var_list)

    def get_config(self):
        # get the configuration containing the properties of the base optimizer
        return self._optimizer.get_config()
      
class neural_network:
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
        self.n_classes = n_classes
        self.epochs = cfg.EPOCHS
        self.last_dense = 0
        self.counter_fc = 0
        self.counter_conv = 0
        self.rgl = False
        self.dense = False
        self.conv = False

    def build_network(self, params):
        """
        Function to define the network structure
        :param params new: network layer parameters
        :return: built model
        """
        # try to resume the training of the last network, otherwise create a new one
        try:
            # sorts the models in the dedicated directory and try to open the last one created
            # if this is successful, read the model in json format and deserialize it in keras format
            model = tf.keras.models.load_model("{}/dashboard/model/model.keras".format(cfg.NAME_EXP))
            print("MODELLO PRECEDENTE")
        except:
            # if the last trained model cannot be loaded, create a new neural network
            if (cfg.MODE == 'fwdPass' or cfg.MODE == 'hybrid') and cfg.DATA_NAME == 'gesture':
                input_shape = self.train_data.shape[2:]
            else:
                input_shape = self.train_data.shape[1:]
            if 'reg' in params:
                reg_layer = reg.l2(params['reg'])
            else:
                reg_layer = None
            inputs = Input((input_shape))
            x = Conv2D(params['unit_c1'], (3, 3), padding='same', kernel_regularizer=reg_layer)(inputs)
            x = Activation(params['activation'])(x)
            x = Conv2D(params['unit_c1'], (3, 3), kernel_regularizer=reg_layer)(x)
            x = Activation(params['activation'])(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(params['dr1_2'])(x)

            x = Conv2D(params['unit_c2'], (3, 3), padding='same', kernel_regularizer=reg_layer)(x)
            x = Activation(params['activation'])(x)
            x = Conv2D(params['unit_c2'], (3, 3), kernel_regularizer=reg_layer)(x)
            x = Activation(params['activation'])(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(params['dr1_2'])(x)
            
            added_convs = [k for k in params if re.match(r'new_conv_\d+$', k)]
            for layer in added_convs:
                x = Conv2D(params[layer], (3, 3), padding='same', kernel_regularizer=reg_layer)(x)
                x = Activation(params['activation'])(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(params['dr1_2'])(x)

            x = Flatten()(x)
            x = Dense(params['unit_d'])(x)
            x = Activation(params['activation'])(x)
            x = Dropout(params['dr_f'])(x)
            added_fcs = [k for k in params if re.match(r'new_fc_\d+$', k)]
            for layer in added_fcs:
                x = Dense(params[layer])(x)
                x = Activation(params['activation'])(x)
                x = Dropout(params['dr_f'])(x)
            
            x = Dense(self.n_classes)(x)
            x = Activation('softmax')(x)

            model = Model(inputs=inputs, outputs=x)

        return model

    def training(self, params, da):
        """
        Function for compiling and running training
        :param params, new, new_fc, new_conv, rem_conv, da: parameters to indicate a possible operation on the network structure and hyperparameter search space
        :return: training history, trained model and and performance evaluation score 
        """
        # build neural network
        model = self.build_network(params)
        # model.summary()
        
        model_name_id = datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f") #time()
        model_json = model.to_json()
        model_name = "{}/Model/model-{}.json".format(cfg.NAME_EXP,model_name_id)
        with open(model_name, 'w') as json_file:
            json_file.write(model_json)

        # save the id of the model in an attribute, so that it can be used later to save an associated db with the same id
        self.last_model_id = model_name_id

        # try to load a set of weights 
        try:
            model.load_weights("{}/Weights/weights.h5".format(cfg.NAME_EXP))
        except:
            pass

        # tensorboard logs
        tensorboard = TensorBoard(
            log_dir="{}/log_folder/logs/{}".format(cfg.NAME_EXP, model_name_id))

        # losses, lrs = self.lolr_checking(model, space, params['learning_rate'], params['batch_size'], self.train_data,
        #                                  self.train_labels, self.test_data, self.test_labels)

        # compiling and training
        _opt = params['optimizer'] + "(learning_rate=" + str(params['learning_rate']) + ")"
        opt = eval(_opt)

        # create a dictionary with which to associate model layers with a multiplier (layer wise learning rate)
        multiplier = {}
        # trainable variables names
        trainable = [(layer.path if hasattr(layer, 'path') else layer.name).split('/')[0] for layer in model.trainable_variables]
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

        # if cfg.MODE == 'depth':
        es1 = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=300, verbose=1, mode='min', restore_best_weights=True)
        es2 = EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=300, verbose=1, mode='max', restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50, verbose=1, min_lr=1e-4)

        # if the flag of data augmentation is true
        if da:
            data_augmentation = tf.keras.models.Sequential([
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest'),
                ])
            model = tf.keras.Sequential([
                data_augmentation,  # Add data augmentation before the model
                *model.layers       
            ])
        # train the network 
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        if (cfg.MODE == 'fwdPass' or cfg.MODE == 'hybrid') and cfg.DATA_NAME == 'gesture':
            if "debug" in cfg.NAME_EXP:
                history = {key: [] for key in ["loss", "accuracy", "val_loss", "val_accuracy"]}
                for _ in range(cfg.EPOCHS):
                    history["val_loss"].append(random.uniform(0.1, 0.5))
                    history["val_accuracy"].append(random.uniform(0.1, 1.0))
                    history["loss"].append(random.uniform(0.1, 0.5))
                    history["accuracy"].append(random.uniform(0.1, 1.0))
            else:
                history = train_model(model, opt, self.train_data, self.train_labels, 
                                    self.test_data, self.test_labels, self.epochs,params, [tensorboard, reduce_lr, es1, es2])
        else:
            if "debug" in cfg.NAME_EXP:
                history = {key: [] for key in ["loss", "accuracy", "val_loss", "val_accuracy"]}
                for _ in range(cfg.EPOCHS):
                    history["val_loss"].append(random.uniform(0.1, 0.5))
                    history["val_accuracy"].append(random.uniform(0.1, 1.0))
                    history["loss"].append(random.uniform(0.1, 0.5))
                    history["accuracy"].append(random.uniform(0.1, 1.0))
            else:
                history = model.fit(self.train_data, self.train_labels, epochs=self.epochs, batch_size=params['batch_size'],
                            verbose=2,
                            validation_data=(self.test_data, self.test_labels),
                            callbacks=[tensorboard, reduce_lr, es1, es2]).history


        # evaluates model performance on test data
        if (cfg.MODE == 'fwdPass' or cfg.MODE == 'hybrid') and cfg.DATA_NAME == 'gesture':
            if "debug" in cfg.NAME_EXP:
                score = [random.uniform(0.1, 0.5), random.uniform(0.1, 1.0)]  # Simulated score for depth mode
            else:
                score = eval_model(model, self.test_data, self.test_labels)
        else:
            if "debug" in cfg.NAME_EXP:
                score = [random.uniform(0.1, 0.5), random.uniform(0.1, 1.0)]  # Simulated score for depth mode
            else:
                score = model.evaluate(self.test_data, self.test_labels, verbose=2)

        # save the neural network weights and then reload them from the same json file you just saved
        # this avoids errors because of the changes in the network structure before training
        weights_name = "{}/Weights/weights-{}.weights.h5".format(cfg.NAME_EXP,model_name_id)
        model.save_weights(weights_name)

        model.save("{}/dashboard/model/model.keras".format(cfg.NAME_EXP))
        
        f = open("{}/Model/model-{}.json".format(cfg.NAME_EXP,model_name_id))
        mj = json.load(f)
        model_json = json.dumps(mj)
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights("{}/Weights/weights-{}.weights.h5".format(cfg.NAME_EXP, model_name_id))
        
        os.system("rm {}/Weights/weights-{}.weights.h5".format(cfg.NAME_EXP, model_name_id))
        os.system("rm {}/Model/model-{}.json".format(cfg.NAME_EXP, model_name_id))
        
        return score, history, model

    



if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, n_classes = gesture_data()
    model = tf.keras.models.load_model("{}/Model/best-model.keras".format('25_03_04_12_35_fwdPass_gesture_accuracy_module_100_20'))
    print(model.summary())
    start = datetime.now()
    score = eval_model(model, X_test, Y_test)
    print("Time fwdPass: ", datetime.now() - start)
    print("Accuracy: ", score[1], "Loss: ", score[0])
    
    print(model.summary())
    start = datetime.now()
    model = tf.keras.models.load_model("{}/Model/best-model.keras".format('25_03_04_12_03_depth_gesture_accuracy_module_100_20'))
    score = model.evaluate(X_test, Y_test)
    print("Time depth: ", datetime.now() - start)
    print("Accuracy: ", score[1], "Loss: ", score[0])
    
    # ss = search_space()
    # space = ss.search_sp()

    # default_params = {'unit_c1': 57, 'dr1_2': 0.1256695669329692, 'unit_c2': 124, 'unit_d': 315,
    #                   'dr_f': 0.045717734023783346, 'learning_rate': 0.08359864897019328, 'batch_size': 252,
    #                   'optimizer': 'RMSProp', 'activation': 'elu', 'reg': 0.05497168445820486, 'new_fc': 497}

    # default_params = {'unit_c1': 58, 'dr1_2': 0.24057804119568613, 'unit_c2': 86, 'unit_d': 344, 'dr_f': 0.09032792140581808, 
    #                   'learning_rate': 0.0001861606710751586, 'batch_size': 63, 'optimizer': 'Adadelta', 'activation': 'relu', 'reg': 0.08651374042577238}

    
    # new = None
    # new_fc = None
    # new_conv = None
    # rem_conv = None
    # rem_fc = None
    # da = True

    # nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
    # score, history, model = nn.training(default_params, new, new_fc, new_conv, rem_conv, rem_fc, da, space)
    # n = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
    
    # model = n.build_network(default_params, None)
    # model.summary()

    # n.rgl = True
    # model = n.insert_batch(model, default_params)
    # model.summary()

    # model = n.insert_conv_section(model, default_params, 1)
    # model.summary()

    # model = n.remove_conv_section(model)
    # model.summary()

    # model = n.insert_fc_section(model, default_params, 1)
    # model.summary()

    # model = n.remove_fc_section(model)
    # model.summary()
    # quit()

    # new_model = n.remove_conv_layer(model, default_params)
    # model_name_id = time()
    # model_json = new_model.to_json()
    # model_name = "{}/Model/model-{}.json".format(cfg.NAME_EXP, model_name_id)
    # with open(model_name, 'w') as json_file:
    #             json_file.write(model_json)
    
    # print(new_model.summary())
    
    # # model2 = n.build_network(default_params, None)
    # # print(model2.summary())
    
    # n.conv = True
    # new_model3 = n.insert_layer(
    #     new_model, '.*flatten.*', default_params, num_cv=1, position='before')

    # print(new_model3.summary())


    # score, history, model = n.training(default_params, True, [True, 1], None, None, space)

    # f2 = open("algorithm_logs/history.txt", "w")
    # f2.write(str(history))
    # f2.close()