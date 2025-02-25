import json
from time import time
import tensorflow as tf
from keras import layers, models, Model, regularizers, callbacks, preprocessing
from keras.optimizers import *
from components.colors import colors
from components.neural_network import neural_network


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

class TensorflowNeuralNetwork (neural_network):

    def from_checkpoint(self, checkpoint):
        checkpoint_json = json.dumps(checkpoint)
        return models.model_from_json(checkpoint_json)

    def from_scratch(self, input_shape, n_classes, params):
        inputs = layers.Input((input_shape))
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

        return Model(inputs=inputs, outputs=x)
    
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
            new_section += [layers.Conv2D(params['unit_c2'], (3,3)), layers.Activation(params['activation'])]
            
            # if batchNormalization is already in the model, add it to the new convolutional section
            if self.dnet.any_batch(model):
                new_section += [layers.BatchNormalization()]
                             
        # if the new section is empty, because no more layers can be added,
        # then return the original model
        if not new_section:
            return model
        
        # add max pooling and dropout to the convolutional section
        new_section += [layers.MaxPooling2D(pool_size=(2, 2)), layers.Dropout(params['dr_f'])]

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
                    layer.kernel_regularizer = regularizers.l2(params['reg'])

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
        return self.dnet.insert_section(model, 1, [layers.BatchNormalization()], 'after', activation_list)
    
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
        new_section = [layers.Dense(params['new_fc']),
                       layers.Activation(params['activation'])]

        # if batchNormalization is already in the model, add it to the new dense section
        if self.dnet.any_batch(model):
            new_section += [layers.BatchNormalization()]

        # add dropout to the dense section
        new_section += [layers.Dropout(params['dr_f'])]
        
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
        tensorboard = callbacks.TensorBoard(
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
        es1 = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=15, verbose=1, mode='min')
        es2 = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=15, verbose=1, mode='max')
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-4)

        # if the flag of data augmentation is true
        if da:
            # define a generator in which are present the values of the data augmentation parameters
            datagen = preprocessing.image.ImageDataGenerator(
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
        model = models.model_from_json(model_json)
        model.load_weights("Weights/weights-{}.weights.h5".format(model_name_id))
        return score, history, model