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
from components.dataset import cifar_data
from components.LOLR import Lolr
from components.search_space import search_space
from components.dynamic_net import dynamic_net
from components.colors import colors
from components.custom_train import train_model, eval_model

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
    def __init__(self, X_train, Y_train, X_test, Y_test, n_classes, best_score=0):
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
        # self.train_data /= 255
        # self.test_data /= 255
        self.n_classes = n_classes
        self.epochs = cfg.EPOCHS
        self.last_dense = 0
        self.counter_fc = 0
        self.counter_conv = 0
        self.tot_fc = 3
        self.tot_conv = 6
        self.rgl = False
        self.dense = False
        self.conv = False
        self.dnet = dynamic_net()
        self.best_score = best_score

    def build_network(self, params, new):
        """
        Function to define the network structure
        :param params new: network layer parameters
        :return: built model
        """
        # try to resume the training of the last network, otherwise create a new one
        try:
            # sorts the models in the dedicated directory and try to open the last one created
            # if this is successful, read the model in json format and deserialize it in keras format
            # list_ckpt = os.listdir("Model")
            # list_ckpt.sort()
            # f = open("{}/Model/".format(cfg.NAME_EXP) + list_ckpt[len(list_ckpt)-1])
            # mj = json.load(f)
            # f.close()
            # model_json = json.dumps(mj)
            # model = tf.keras.models.model_from_json(model_json)
            model = tf.keras.models.load_model("{}/dashboard/model/model.keras".format(cfg.NAME_EXP))
            print("MODELLO PRECEDENTE")
        except:
            # if the last trained model cannot be loaded, create a new neural network
            # exit()
            if cfg.MODE == 'fwdPass' or cfg.MODE == 'hybrid':
                input_shape = self.train_data.shape[2:]
            else:
                input_shape = self.train_data.shape[1:]
            inputs = Input((input_shape))
            x = Conv2D(params['unit_c1'], (3, 3), padding='same')(inputs)
            x = Activation(params['activation'])(x)
            x = Conv2D(params['unit_c1'], (3, 3))(x)
            x = Activation(params['activation'])(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(params['dr1_2'])(x)

            x = Conv2D(params['unit_c2'], (3, 3), padding='same')(x)
            x = Activation(params['activation'])(x)
            x = Conv2D(params['unit_c2'], (3, 3))(x)
            x = Activation(params['activation'])(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(params['dr1_2'])(x)

            x = Flatten()(x)
            x = Dense(params['unit_d'])(x)
            x = Activation(params['activation'])(x)
            x = Dropout(params['dr_f'])(x)
            x = Dense(self.n_classes)(x)
            x = Activation('softmax')(x)

            model = Model(inputs=inputs, outputs=x)
            
            ## provvisorio
            # model_name_id = time()
            # model_json = model.to_json()
            # model_name = "Model/model-{}.json".format(model_name_id)
            # with open(model_name, 'w') as json_file:
            #     json_file.write(model_json)

        return model

    def insert_conv_section(self, model, params, n_conv):
        """
        method used for inserting a convolutional section
        :param model params n_conv: insert in 'model' a number of 'n_conv' sections with 'params' parameters
        :return: model with new conv section
        """

        # counts the current number of convolutional layers in the model
        current_conv_count = self.dnet.count_layer_type(model, 'Conv2D')

        # build the new convolutional section, consisting of two convolutions
        # with their activations, a max pooling and dropout
        # initialize the counter of new convolutions to zero and
        # the list of layers of the new section to an empty list
        new_conv_count = 0
        new_section = []

        # cycle to add at most two convolutions in the new conv section
        for i in range(2):
            # increase the counter to add a new convolutional layer
            new_conv_count += 1
            # if the sum of the model convolutions and the new ones is greater than the limit,
            # then don't add any more layers
            if (new_conv_count + current_conv_count) > self.tot_conv:
                break
            # otherwise add a convolutional layer and its activation
            new_section += [Conv2D(params['unit_c2'], (3,3)), Activation(params['activation'])]
            
            # if batchNormalization is already in the model, add it to the new convolutional section
            if self.dnet.any_batch(model):
                new_section += [BatchNormalization()]
                             
        # if the new section is empty, because no more layers can be added,
        # then return the original model
        if not new_section:
            return model
        
        # add max pooling and dropout to the convolutional section
        new_section += [MaxPooling2D(pool_size=(2, 2)), Dropout(params['dr_f'])]

        # insert the new section before the flatten, so after the last convolutional section
        return self.dnet.insert_section(model, n_conv, new_section, 'before', 'Flatten')

    def insert_batch(self, model, params):
        """
        method used for inserting batchNormalization operations
        :param model params n_conv: insert in 'model' batchNormalization and regularization with 'params' parameters
        :return: model with batchNormalization and regularization
        """

        # if batchnormalization is already in the model, then return it
        if self.dnet.any_batch(model):
            return model

        # otherwise add regularization to each convolutional layer
        for layer in model.layers:
            if self.rgl:
                if 'Conv2D' in layer.__class__.__name__:
                    layer.kernel_regularizer = reg.l2(params['reg'])

        # iterating over all the layers, search all the activations to which to add the batchNormalization
        activation_list = []
        for layer in model.layers:
            # get the name of the activation function based on the version of keras
            activation_name = layer.activation.__name__ if hasattr(layer, 'activation') else layer.output.name
            layer_class = layer.__class__.__name__
            # if the activation is not softmax, so the last layer of the network, then save the layer in the list
            if layer_class == 'Activation' and activation_name != 'softmax':
                activation_list += [layer.name]

        # apply batchNormalization to all saved activations
        return self.dnet.insert_section(model, 1, [BatchNormalization()], 'after', activation_list)

    def insert_fc_section(self, model, params, n_fc):
        """
        method used for inserting a dense section
        :param model params n_fc: insert in 'model' a number of 'n_fc' sections with 'params' parameters
        :return: model with new dense section
        """

        # if the number of dense layers is greater than or equal to the max, then return the model
        if self.dnet.count_layer_type(model, 'Dense') >= self.tot_fc:
            return model 

        # build the new dense section, consisting of a dense layer and its activation
        new_section = [Dense(params['new_fc']),
                       Activation(params['activation'])]

        # if batchNormalization is already in the model, add it to the new dense section
        if self.dnet.any_batch(model):
            new_section += [BatchNormalization()]

        # add dropout to the dense section
        new_section += [Dropout(params['dr_f'])]
        
        # insert the new section after the flatten, so a the beggining of the dense section
        return self.dnet.insert_section(model, n_fc, new_section, 'after', 'Flatten')

    def remove_conv_section(self, model):
        """
        method used for removing a convolutional section
        :param model: model from which to remove the convolutional section
        :return: model without convolutional section
        """
        # if the number of convolutions is less than or equal to 1,
        # don't remove any convolution and return the model
        current_conv_count = self.dnet.count_layer_type(model, 'Conv2D')
        if current_conv_count <= 1:
            return model

        # get the name of the first layer of the last convolutional section
        last_conv_start = self.dnet.get_last_section(model, 'Conv2D')
        
        # remove the convolutional section starting from the convolution found earlier
        # and all associated layers in linked_section
        linked_section = ['Conv2D', 'Activation', 'BatchNormalization']
        
        # If the number of convolutions is odd, it means that one of the two conv in
        # the last convolutional block has already been eliminated
        # it's then necessary to remove all layers of the block, including maxpool and dropout
        if (current_conv_count % 2) == 1:
            linked_section += ['MaxPooling2D', 'Dropout']
        
        return self.dnet.remove_section(model, last_conv_start, linked_section, True, True)

    def remove_fc_section(self, model):
        """
        method used for removing a dense section
        :param model: model from which to remove the dense section
        :return: model without dense section
        """

        # if the number of dense layers is less than or equal to 2,
        # specifically a dense layer after the flatten and the output,
        # don't remove any dense layer and return the model
        if self.dnet.count_layer_type(model, 'Dense') <= 2:
            return model

        # remove the first dense section in the model and all associated layers in linked_section
        linked_section = ['Activation', 'BatchNormalization', 'Dropout']
        return self.dnet.remove_section(model, 'Dense', linked_section, True, True)
      
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
                        model = self.insert_fc_section(model, params, new_fc[1])
                # if the flag for the addition of regularization is true
                if new:
                    self.rgl = True
                    self.dense = False
                    model = self.insert_batch(model, params)
                # if the flag for the addition of a convolutional layer
                if new_conv:
                    if new_conv[0]:
                        self.conv = True
                        self.dense = False
                        self.rgl = False
                        model = self.insert_conv_section(model, params, new_conv[1])
                # if the flag for the removal of a convolutional layer is true
                if rem_conv:
                    self.conv = False
                    self.dense = False
                    self.rgl = False
                    model = self.remove_conv_section(model)
                # if the flag for the removal of a dense layer is true
                if rem_fc:
                    self.conv = False
                    self.dense = False
                    self.rgl = False
                    model = self.remove_fc_section(model)

        except Exception as e:
            print(colors.FAIL, e, colors.ENDC)
        
        # print the structure of the neural network and save it in a json file,
        # using the current time as identifier of the model
        # model.build()
        print(model.summary())
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
            print("Restart\n")

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

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        es1 = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=15, verbose=0, mode='min')
        es2 = EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=15, verbose=0, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=0, min_lr=1e-4)

        # if the flag of data augmentation is true
        if da:
            # define a generator in which are present the values of the data augmentation parameters
            # datagen = ImageDataGenerator(
            #     width_shift_range=0.1,
            #     height_shift_range=0.1,
            #     fill_mode='nearest',
            #     cval=0.,
            #     horizontal_flip=True)
            # datagen.fit(self.train_data)
            
            data_augmentation = tf.keras.models.Sequential([
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest'),
                ])
            model = tf.keras.Sequential([
                data_augmentation,  # Aggiunto come primo strato
                *model.layers       # Importa tutti gli strati esistenti
            ])
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        
        if cfg.MODE == 'fwdPass' or cfg.MODE == 'hybrid':
            history = train_model(model, opt, self.train_data, self.train_labels, 
                                    self.test_data, self.test_labels, self.epochs,params, [tensorboard, reduce_lr, es1, es2])
        else:
            history = model.fit(self.train_data, self.train_labels, epochs=self.epochs, batch_size=params['batch_size'],
                            verbose=0,
                            validation_data=(self.test_data, self.test_labels),
                            callbacks=[tensorboard, reduce_lr, es1, es2]).history

        # evaluates model performance on test data
        if cfg.MODE == 'fwdPass' or cfg.MODE == 'hybrid':
            score = eval_model(model, self.test_data, self.test_labels)
        else:
            score = model.evaluate(self.test_data, self.test_labels)

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
        print("acc: ", score[1], "loss: ", score[0], "best: ", self.best_score)
        if score[1] > self.best_score:
            print("SAVE MODEL - Best score: ", score[1])
            self.best_score = score[1]
            model.save("{}/Model/best-model.keras".format(cfg.NAME_EXP))
            # os.system("mv {}/Weights/weights-{}.weights.h5 {}/Weights/best-weights.h5".format(cfg.NAME_EXP, model_name_id, cfg.NAME_EXP))
            # os.system("mv {}/Model/model-{}.json {}/Model/best-model.json".format(cfg.NAME_EXP, model_name_id, cfg.NAME_EXP))
        
        os.system("rm {}/Weights/weights-{}.weights.h5".format(cfg.NAME_EXP, model_name_id))
        os.system("rm {}/Model/model-{}.json".format(cfg.NAME_EXP, model_name_id))
        return score, history, model, self.best_score




def evaluate_net(path, mode, frames=32, channels=2, pol=2):
    from flops import flops_calculator as fc
    # print(f"-------------------------------- {mode} --------------------------------")
    cfg.MODE = mode
    cfg.FRAMES = int(frames)
    cfg.NUM_CHANNELS = int(channels)
    cfg.POLARITY = "both" if int(pol) == 2 else "sum" 
    # print("cfg.FRAMES: ", cfg.FRAMES, "cfg.CHANNELS: ", cfg.NUM_CHANNELS, "cfg.POLARITY: ", cfg.POLARITY,)
    X_train, X_test, Y_train, Y_test, n_classes = gesture_data() # gesture_data() cifar_data()
    model = tf.keras.models.load_model(path) #"{}/Model/best-model.keras".format('25_03_04_12_03_depth_gesture_accuracy_module_100_20'))
    # print(model.summary())
    import io
    summary_str = io.StringIO()
    model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
    # print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    if mode == "depth":
        score = model.evaluate(X_test, Y_test, verbose=0)
    else:
        score = eval_model(model, X_test, Y_test)
    gpus = tf.config.list_physical_devices('GPU')
    time_gpu = "0:00:00.0"
    if len(gpus) > 0:
        start = datetime.now()
        if mode == "depth":
            score = model.evaluate(X_test, Y_test, verbose=0)
        else:
            score = eval_model(model, X_test, Y_test)
        # print("Time: ", datetime.now() - start)
        # print("Accuracy: ", score[1], "Loss: ", score[0])
        time_gpu = (datetime.now() - start) / X_test.shape[0]
    with tf.device('/CPU:0'):
        start = datetime.now()
        if mode == "depth":
            score = model.evaluate(X_test, Y_test, verbose=0)
        else:
            score = eval_model(model, X_test, Y_test)
        # print("Time: ", datetime.now() - start)
        # print("Accuracy: ", score[1], "Loss: ", score[0])
        time_cpu = (datetime.now() - start) / X_test.shape[0]
    flops, r_dict = fc.analyze_model(model)
    trainableParams = np.sum([np.prod(v.shape)for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.shape)for v in model.non_trainable_weights])
    nparams = trainableParams + nonTrainableParams
    return [time_gpu, time_cpu], score, flops.total_float_ops, nparams
    # print(f"FLOPS: {flops.total_float_ops}")
    # print(f"PARAMS: {nparams}")
    

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, n_classes = cifar_data() #gesture_data()
    ss = search_space()
    space = ss.search_sp()

    # default_params = {'unit_c1': 57, 'dr1_2': 0.1256695669329692, 'unit_c2': 124, 'unit_d': 315,
    #                   'dr_f': 0.045717734023783346, 'learning_rate': 0.08359864897019328, 'batch_size': 252,
    #                   'optimizer': 'RMSProp', 'activation': 'elu', 'reg': 0.05497168445820486, 'new_fc': 497}

    default_params = {'unit_c1': 58, 'dr1_2': 0.24057804119568613, 'unit_c2': 86, 'unit_d': 344, 'dr_f': 0.09032792140581808, 
                      'learning_rate': 0.0001861606710751586, 'batch_size': 63, 'optimizer': 'Adadelta', 'activation': 'relu', 'reg': 0.08651374042577238}

    
    new = None
    new_fc = None
    new_conv = None
    rem_conv = None
    rem_fc = None
    da = True

    nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
    score, history, model = nn.training(default_params, new, new_fc, new_conv, rem_conv, rem_fc, da, space)
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