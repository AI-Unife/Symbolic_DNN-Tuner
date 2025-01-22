import json
from keras import layers, models, Model, regularizers
from components.neural_network import neural_network


class TensorflowNeuralNetwork (neural_network):

    def from_checkpoint(checkpoint):
        checkpoint_json = json.dumps(checkpoint)
        return models.model_from_json(checkpoint_json)

    def from_scratch(input_shape, n_classes, params):
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