import re
import os
from time import time
import numpy as np
from pytest import param
import json
from colors import colors

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers as reg
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input,
                                     BatchNormalization)
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dataset import cifar_data
from LOLR import Lolr
from search_space import search_space

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
        self.train_data /= 255
        self.test_data /= 255
        self.n_classes = n_classes
        self.epochs = 200
        self.last_dense = 0
        self.counter_fc = 0
        self.counter_conv = 0
        self.tot_fc = 3
        self.tot_conv = 6
        self.rgl = False
        self.dense = False
        self.conv = False

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
            list_ckpt = os.listdir("Model")
            list_ckpt.sort()
            f = open("Model/" + list_ckpt[len(list_ckpt)-1])
            mj = json.load(f)
            f.close()
            model_json = json.dumps(mj)
            model = tf.keras.models.model_from_json(model_json)
            print("MODELLO PRECEDENTE")
        except:
            # if the last trained model cannot be loaded, create a new neural network
            print(self.train_data.shape)

            inputs = Input((self.train_data.shape[1:]))
            x = Conv2D(params['unit_c1'], (3, 3), padding='same')(inputs)
            x = Activation(params['activation'])(x)
            x = Conv2D(params['unit_c1'], (3, 3))(x)
            x = Activation(params['activation'])(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            #x = Dropout(params['dr1_2'])(x)

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

    def insert_layer(self, model, layer_regex, params, num_fc=0, num_cv=0, position='after'):
        check = True
        # Auxiliary dictionary to describe the network graph
        K.clear_session()
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
        # set the input layers of each layer
        for layer in model.layers:
            # if the flag indicating the addition of batch normalization is true
            # and the current layer is a convolutional layer
            if self.rgl:
                if 'conv' in layer.name:
                    # add to layer regularization the batch normalization, getting parameters from the search space
                    layer.kernel_regularizer = reg.l2(params['reg'])
            # iterate over each output layer of the current layer
            for node in layer._outbound_nodes:
                # if the layer has multiple output
                if len(layer._outbound_nodes) > 1:
                    # if the current layer is dense and the flag for adding a new fully connected layer is false
                    if 'dense' in layer.name and not self.dense and check:
                        check = False
                        pass
                    else:
                        # get the name of the layer depending on the version of keras
                        # and if that name is not already a key, then add it to the dictionary
                        # as a list with the input as the only element
                        layer_name = node.operation.name if hasattr(node, 'operation') else node.outbound_layer.name
                        if layer_name not in network_dict['input_layers_of']:
                            network_dict['input_layers_of'].update(
                                {layer_name: [layer.name]})
                        else:
                            # otherwise add it to the input layer list of the current layer
                            network_dict['input_layers_of'][layer_name].append(layer.name)
                        check = True
                        break
                else:
                    # get the name of the layer depending on the version of keras
                    # and if that name is not already a key, then add it to the dictionary
                    # as a list with the input as the only element
                    layer_name = node.operation.name if hasattr(node, 'operation') else node.outbound_layer.name
                    if layer_name not in network_dict['input_layers_of']:
                        network_dict['input_layers_of'].update(
                            {layer_name: [layer.name]})
                    else:
                        # otherwise add it to the input layer list of the current layer
                        network_dict['input_layers_of'][layer_name].append(layer.name)

        # set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

        # get the input of each layer
        # teh key is the name of a layer, while the value is the name of its input layer
        d = network_dict['input_layers_of']
        k = []
        _d = {}
        # iterate over each layer and put all the layers name in the k list
        for i in d.keys():
            k.append(i)
        __d = d.copy()

        # iterate over each pair of layer name and its content
        for i, j in zip(d, k):
            # if a layer has mutiple inputs
            if len(d[i]) > 1:
                # insert the name of the second input into the layer name list
                # and insert into the dictionary the first input layer
                k.insert(k.index(i), d[i][1])
                __d[d[i][1]] = [d[i][0]]
                __d[i].pop(0)

        # copy the elements of the dict used for mmultiple inputs search into
        # the dictionary that maps the structure of the network
        for i in k:
            _d[i] = __d[i]
        network_dict.pop('input_layers_of')
        network_dict['input_layers_of'] = _d

        # iterate over all layers after the input and build the new network
        for layer in model.layers[1:]:
            # use 'dense_1' instead of 'final' for continuity with dense layer network names
            if layer.name == 'final':
                name = "dense_1"
            else:
                name = layer.name
            # create list of input layers iterating over the dict, searching for the
            # value of each layer of new_output_tensor_of in input_layers_of
            layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                           for layer_aux in network_dict['input_layers_of'][name]]

            # get activation name if layer has that attribute
            # this is used to get the name of the output layer based on the version of keras
            layer_output_name = layer.activation.__name__ if hasattr(layer, 'activation') else layer.output.name

            if len(layer_input) == 1:
                # fix error keras >= 3.4
                # 'input layer' is stored in list of lists [[...]] instead of a single list [...]
                # if the first element is a list, extract it
                if type(layer_input[0]) is list:
                    layer_input = layer_input[0]
                layer_input = layer_input[0]

            # insert layer if name matches the regular expression and doesn't have the same size as the dnn output
            if re.match(layer_regex, layer.name) and layer.output.shape[1] != 10:
                if position == 'replace':
                    x = layer_input
                elif position == 'after':
                    x = layer(layer_input)
                elif position == 'before':
                    pass
                else:
                    raise ValueError('position must be: before, after or replace')

                # check if the current layer isn't not the last layer with a softmax activation
                if not 'Softmax' in layer_output_name or not 'softm' in layer_output_name:

                    # if the flag indcatiding the addition of batch normalization is true
                    # and there's no batch normalization layers already, add it in the network
                    if self.rgl and not any(['batch' in i.name for i in model.layers]):
                        naming = '{}'.format(time())
                        x = BatchNormalization(name="batch_norm_{}".format(naming))(x)
                    # if the flag indcatiding the addition of dense layer is true
                    # and if the current number of dense layers is less than the total maximum number
                    elif self.dense and self.counter_fc < self.tot_fc:
                        # if i need to add at least one dense layer, but that this number is greater than
                        # the maximum number of dense layers, then set that parameter to 0
                        if num_fc > 0:
                            if num_fc > self.tot_fc:
                                num_fc = 0

                            # add a number of dense layers equal to those given in the parameter
                            for _ in range(num_fc):
                                self.counter_fc += 1
                                x = Dense(params['new_fc'], name='dense_{}'.format(time()))(x)
                        else:
                            # add a dense layer
                            x = Dense(params['new_fc'], name='dense_{}'.format(time()))(x)
                        # store the size of the last dense layer
                        self.last_dense = x.shape[1]

                    # if the flag indicating the addition of a convolutional layer is true
                    # and if the current number of conv layers is less than the total maximum number
                    elif self.conv and self.counter_conv < self.tot_conv:
                        # add a convolutional layer and an activation function layer
                        self.counter_conv += 1
                        naming = '{}'.format(time())
                        x = Conv2D(params['unit_c2'], (3, 3), padding='same', name='conv_{}'.format(naming))(x)
                        x = Activation(params['activation'], name='activation_{}'.format(naming))(x)
                        # x = Conv2D(params['unit_c2'], (3, 3), name='conv_2_{}'.format(naming))(x)
                        # x = Activation(params['activation'], name='activation_2_{}'.format(naming))(x)
                        # x = MaxPooling2D(pool_size=(2, 2), name='maxpooling_{}'.format(naming))(x)
                        # x = Dropout(params['dr1_2'], name='dropout_{}'.format(naming))(x)

                # if the position to insert the layer is before
                if position == 'before':
                    x = layer(x)
            else:
                # if the size of the output matches the number of classes, it matches the regular expression,
                # but it doesn't have a softmax activation function
                if layer.output.shape[1] == 10 and re.match(layer_regex, layer.name) and not 'softmax' in \
                                                                                             layer_output_name:
                    # add a dense layer with the same size called 'final'
                    x = Dense(layer.output.shape[1], name='final')(x) 
                else:
                    # if the layer is dense and the output size is equal classes number,
                    # then it's the last layer in the network
                    if self.conv and 'dense' in layer.name and layer.output.shape[1] == 10:
                        x = Dense(layer.output.shape[1], name='final')(x)
                    elif self.conv and 'dense' in layer.name:
                        # otherwise if it's a generic dense layer instead, add it
                        x = Dense(params['unit_d'])(x)
                    else:
                        # if it's not a dense layer and doesn't match regular expressions, then it's the input
                        x = layer(layer_input)

            # set new output tensor (the original one, or the one of the inserted layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})

        input = model.inputs
        return Model(inputs=input, outputs=x)

    def remove_conv_layer(self, model, params):
        """
        Remove convolutional layer
        :param model (_type_): keras model
        :return: new_model : new updated keras model
        """
        # check if there's at least one batch normalization operation in the network
        btc = any(['batch' in i.name for i in model.layers])

        # count the number of convolutional layers in the network and if there is only one,
        # return the model without removing further layers
        c = 0
        layers_list = model.layers
        for i in layers_list:
            if 'conv' in i.name:
                c += 1
        if c == 1:
            return model

        # get layer of input
        new_input = model.get_layer(layers_list[0].name)

        # reverse list of layers of the neural network
        reverse_layers_list = layers_list[::-1]

        # list used to map the final network structure after the removal of the conv layer
        buffer_rev = reverse_layers_list

        # init layers accumulators to empty lists
        reused_layers, to_delete, head = [], [], []

        # iterate over each layer of the dnn
        for layer in reverse_layers_list:
            # if the current layer is a convolutional layer
            if 'conv' in layer.name:
                # if there's at least one batch normalization in the network
                if btc:
                    # last part of the network will start with flatten
                    # go back four steps, because activation, batch and max pooling layers
                    to_delete.append(buffer_rev[reverse_layers_list.index(layer)-3])
                    head_start = buffer_rev[reverse_layers_list.index(layer)-4]
                    buffer_rev.remove(buffer_rev[reverse_layers_list.index(layer)-3])
                else:
                    # last part of the network will start with flatten
                    # go back three steps, because activation and max pooling layers
                    head_start = buffer_rev[reverse_layers_list.index(layer)-3]
 
                # insert layers that need to be deleted into to_delete list
                # and remove them from the buffer_rev that maps the dnn architeture
                # specifically the convolutional layer, the activation and the batch norm
                to_delete.append(buffer_rev[reverse_layers_list.index(layer)-2])
                to_delete.append(buffer_rev[reverse_layers_list.index(layer)-1])
                to_delete.append(buffer_rev[reverse_layers_list.index(layer)])
                buffer_rev.remove(buffer_rev[reverse_layers_list.index(layer)-2])
                buffer_rev.remove(buffer_rev[reverse_layers_list.index(layer)-1])
                buffer_rev.remove(buffer_rev[reverse_layers_list.index(layer)])
                break
        
        # iterate over each layer
        for layer in buffer_rev[:-1]:
            # if the name of the first layer of the final part of the dnn is present in the buffer
            if head_start.name in layer.name:
                # put in the head part all layers that have been reused and the layer itself
                head.extend(reused_layers)
                reused_layers = []
                head.append(buffer_rev[reverse_layers_list.index(layer)])
            else:
                # otherwise accumulate the layers in the list of reused layers
                reused_layers.append(buffer_rev[reverse_layers_list.index(layer)])
        
        # return the model if there's only one conv layer in the neural network
        if c == 1:
            return model

        # reverse layers accumulators in order to build the new dnn architeture
        head.reverse()
        to_delete.reverse()
        reused_layers.reverse()
        
        # print result of the dnn architeture analysis, showing the head of the dnn and the reused layers
        print("\n\n### head ###")
        for i in head:
            print(i.name)
        print("\n\n### reuse ###")
        for i in reused_layers:
            print(i.name)

        # generates the neural network by iterating over the previously constructed lists, starting from the input 
        x = new_input.output
        buff = None
        for e, i in enumerate(reused_layers):
            if e == len(reused_layers)-1:
                if i.__class__ == buff.__class__:
                    if 'max_pool' in i.name or 'activation' in i.name or 'dropout' in i.name:
                        pass
                else:
                    # get current layer, the last one of the reused layer,
                    # and put its output as the last of the second deleted element (activation)
                    # this allows continuity of output size once the conv layer is removed
                    _x = model.get_layer(i.name)
                    _x._outbound_nodes[0] = to_delete[1]._outbound_nodes[0]
                    x = _x(x)
            else:
                if i.__class__ == buff.__class__:
                    if 'max_pool' in i.name or 'activation' in i.name or 'dropout' in i.name or 'batch' in i.name:
                        pass
                else:
                    # concatenates the various layers based on reused layer list
                    x = model.get_layer(i.name)(x)
            buff = i
    
        # add the last part to the neural network architecture
        for e, i in enumerate(head):
            # based on the layer name, add a a layer with the same type
            if 'dense' in i.name:
                x = Dense(i.units, name='dense_{}'.format(time()))(x)
            elif 'dropout' in i.name:
                x = Dropout(i.rate, name='dropout_{}'.format(time()))(x)
            elif 'flatten' in i.name:
                x = Flatten()(x)
            elif 'max_p' in i.name:
                x = MaxPooling2D(pool_size=(
                    2, 2), name=i.name)(x)
            elif 'batch' in i.name:
                x = BatchNormalization(name=i.name)(x)
            else:
                # if the layer is an activation and the iteration 
                # reached the last element of the list, add the output layer with softmax
                # otherwise if it's a generic activation, add it to the network
                if 'activation' in i.name and e == len(head)-1:
                    x = Activation('softmax', name='activation_{}'.format(time()))(x)
                else:
                    x = Activation(params['activation'], name='activation_{}'.format(time()))(x)
        
        # depending on the keras version, it's necessary to determine where to find the input tensor
        new_input_model = new_input._input_tensor if hasattr(new_input, '_input_tensor') else new_input.input             
        return Model(inputs=new_input_model, outputs=x)

    def remove_fc(self, model):
        """
        Method used for removing a dense layer from the neural network
        :param model: neural network model from which a dense layer needs to be removed
        :return: model without a dense layer
        """
        # dense layer counter set to 0
        d = 0

        # get the layers of neural network and reverse the list
        # scan the architecture in reverse in order to find the dense layer to remove
        layers_list = model.layers
        layers_list = layers_list[::-1]

        # boolean to indicate if the layer to be removed was found
        first_dense = False

        # initialize dict containing the neural network layers after removal as empty
        removed = {}

        # initialize the name of the layer to be removed as an empty string
        removed_name = ""

        # iterate over each layer
        for i in layers_list:
  
            # get the name of the current layer class from which it's derived
            layer_name = i.__class__.__name__
            
            # boolean indicating if the current layer is dense
            dense_type = ('dense' in i.name or 'dense' in layer_name)

            # if the layer is dense and if it's not the output layer and 
            # i haven't found the layer to remove, then save the name of the layer to remove
            if dense_type and i.output.shape[1] != 10 and not first_dense:
                d+= 1
                first_dense = True
                removed_name = i.name
            else:
                # otherwise if it's a dense layer, increases the dense layer counter and
                # save the layer into the dict that maps the network architecture
                if dense_type:
                    d+=1
                removed |= {i.name : i}

        # if the total number of dense layers, not including the output one,
        # is less than or equal to 1, don't remove any layers and return the model
        if d <= 1:
            return model

        # reverse the dict that maps the architecture of the neural network
        removed_order = {}
        for i in removed.keys():
            removed_order = {i : removed[i]} | removed_order
            
        x = None
        new_inputs = None

        # build a new neural network, based on the previously saved layers,
        # adding a specific layer based on the type of layer saved before
        for layer_key in removed_order.keys():
            layer = removed_order[layer_key]
            layer_name = layer.__class__.__name__
            if 'Input' in layer_name:
                new_input = layer._input_tensor if hasattr(layer, '_input_tensor') else layer.input
                if(new_input.shape[0] == None):
                    new_input = new_input.shape[1:]
                new_inputs = Input(new_input)
                x = new_inputs
            elif 'Conv' in layer_name:
                x = Conv2D(layer.kernel.shape[-1], layer.kernel_size, padding=layer.padding, name=layer_key)(x)
            elif 'Activation' in layer_name:
                activation_name = layer.activation.__name__ if hasattr(layer, 'activation') else layer.output.name
                x = Activation(activation_name, name=layer_key)(x)
            elif 'Flatten' in layer_name:
                x = Flatten()(x)
            elif 'Batch' in layer_name:
                x = BatchNormalization(name=layer_key)(x)
            elif 'Dense' in layer_name:
                x = Dense(layer.units, name=layer_key)(x)
            elif 'Dropout' in layer_name:
                x = Dropout(layer.rate, name=layer_key)(x)
            elif 'Max' in layer_name:
                x = MaxPooling2D(layer.pool_size, name=layer_key)(x)

        print(f"\n#### removed ####\n{removed_name}\n")

        return Model(inputs=new_inputs, outputs=x)
        
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
                        model = self.insert_layer(model, '.*dense.*', params, num_fc=new_fc[1])
                # if the flag for the addition of regularization is true
                if new:
                    self.rgl = True
                    self.dense = False
                    model = self.insert_layer(model, '.*activation.*', params)
                # if the flag for the addition of a convolutional layer
                if new_conv:
                    if new_conv[0]:
                        self.conv = True
                        self.dense = False
                        self.rgl = False
                        model = self.insert_layer(model, '.*flatten.*', params, num_cv=new_conv[1], position='before')
                # if the flag for the removal of a convolutional layer is true
                if rem_conv:
                    self.conv = False
                    self.dense = False
                    self.rgl = False
                    model = self.remove_conv_layer(model, params)
                # if the flag for the removal of a dense layer is true
                if rem_fc:
                    self.conv = False
                    self.dense = False
                    self.rgl = False
                    model = self.remove_fc(model)

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
    print(model.summary())
    
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