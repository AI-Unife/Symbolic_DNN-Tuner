import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from time import time
from random import random
import importlib

from keras.applications.vgg16 import VGG16

class dynamic_net:
    def remove_section(self, model, target, linked_layers, delimiter, first_found):
        """
        method used to remove a section of layers
        """

        # boolean used to identify which layers in the new architecture can use the old weights
        reused_weights = True

        # booleans used during the search of layers to be removed
        # the first indicates if you're inside section during the search
        # the second indicates if a section that can be removed was found
        fc_section, fc_found = False, False

        # initialize dict containing the neural network layers after removal as empty
        removed = {}

        # initialize the name of layers to be removed as an empty string
        removed_name = ""

        # get the layers of neural network
        layers_list = model.layers

        # iterate over each layer
        for i in layers_list:
  
            # get the name of the current layer class from which it's derived
            layer_class = i.__class__.__name__

            # if i'm not inside a section, but the target does match with the searched layer class or a layer name
            if not fc_section and not fc_found and (target in layer_class or target in i.name):
                fc_section = True
                removed_name += i.name + '\n'

                # can't use the old weights for subsequent layers, because the size of layers will change
                reused_weights = False
            elif fc_section:
                # if the current layer is not among those connected to
                # this section, it means that i've reached the end
                if layer_class not in linked_layers and i.name not in linked_layers:
                    fc_section = False
                    fc_found = first_found
                    if delimiter:
                        removed |= {i.name : [reused_weights, {'class_name' : layer_class}, i.get_config()]}
                    else:
                        removed_name += i.name + '\n'
                else:
                    removed_name += i.name + '\n'
            else:
                # add the current layer to the final archiecture
                removed |= {i.name : [reused_weights, {'class_name' : layer_class}, i.get_config()]}

        print(f"\n#### removed ####\n{removed_name}")

        return self.build_model(model, removed)

    def insert_section(self, model, n_section, new_section, position, target):
        """
        method used for inserting a new section of layers
        """

        if not self.all_layers(new_section):
            print("\n#### New section contains elements that are not layers ####\n")
            return model

        # boolean used to identify which layers in the new architecture can use the old weights
        reused_weights = True

        # initialize dict containing the neural network layers after addition as empty
        net_dense = {}

        # get the layers of neural network
        layers_list = model.layers

        # iterate over each layer
        for i in layers_list:
            # get the name of the current layer class from which it's derived
            layer_class = i.__class__.__name__

            section = {}
            replace_flag = False
            
            # if the target matches the searched class or the searched layer name:
            if (layer_class in target or i.name in target):
                reused_weights = False
                replace_flag = True

                # list containing the layers the new section
                fc_section = []

                # insert 'n_section' sections, adding an ID to each name to make them unique
                for _ in range(n_section):
                    fc_section += self.add_names(new_section)

                # add all the layers of the section to the final architecture
                for x in fc_section:
                    section |= {x.name : [reused_weights, {'class_name' : x.__class__.__name__}, x.get_config()]}

            current_layer = {i.name : [reused_weights, {'class_name' : layer_class}, i.get_config()]}

            if position == 'before':
                net_dense |= (section | current_layer)
            elif position == 'after':
                net_dense |= (current_layer | section)
            elif position == 'replace' and replace_flag:
               replace_flag = False
               net_dense |= section
            else:
               net_dense |= current_layer
        
        return self.model_from_dict(model, net_dense)
        #return self.build_model(model, net_dense)

    def build_model(self, model, model_dict):
        """
        Method used to build the network and handle any problems during its creation
        """
        try:
            return self.model_from_dict(model, model_dict)
        except:
            print("\n#### Error during model creation ####\n")
            return model

    def model_from_dict(self, model, model_dict):
        """
        method used to build the network based on a dict containing the configurations of each layer
        """

        # set variables containing respectively the new network archiecture and input layer to 'None'
        x = None
        new_inputs = None

        # names of all layers in the model
        name_list = [i.name for i in model.layers]

        # import the module containing all the keras layers
        module = importlib.import_module("tensorflow.keras.layers")

        # build a new neural network, based on the previously saved layers
        for layer_key in model_dict.keys():
            layer = model_dict[layer_key][2]
            layer_name = model_dict[layer_key][1]['class_name']

            if 'Input' in layer_name:
                input_shape = model.input
                if isinstance(input_shape, list):
                    input_shape = input_shape[0]

                new_inputs = Input(input_shape.shape[1:])
                x = new_inputs
            elif model_dict[layer_key][0] and layer_key in name_list:
                # if i can use old weights from the current layer, load them from the model
                x = model.get_layer(layer['name'])(x)
            else:
                # otherwise create a new layer based on the configuration of the previously saved one
                if layer_name in ['Conv2D', 'SeparableConv2D', 'Conv2DTranspose']:
                    layer_inst = getattr(module, layer_name)(1, 1)
                elif layer_name in ['ZeroPadding2D',
                                    'MaxPooling2D',
                                    'AveragePooling2D',
                                    'GlobalAveragePooling2D']:
                    layer_inst = getattr(module, layer_name)((2,2))
                elif layer_name in ['Dense']:
                    layer_inst = getattr(module, layer_name)(1)
                elif layer_name in ['Dropout', 'SpatialDropout2D']:
                    layer_inst = getattr(module, layer_name)(0.5)
                elif layer_name in ['Activation']:
                    layer_inst = getattr(module, layer_name)('relu')
                elif layer_name in ['Flatten','BatchNormalization', 'ReLU', 'Softmax']:
                    layer_inst = getattr(module, layer_name)()

                x = layer_inst.from_config(layer)(x)

        return Model(inputs=new_inputs, outputs=x)

    def add_names(self, layer_list):
        """
        method used to generate an id to be added to each new layer name
        """
        naming = "_{}".format(time() + random())
        new_list = []
        for layer in layer_list:
            layer_config = layer.get_config()
            layer_config['name'] += naming
            new_list += [layer.from_config(layer_config)]
        return new_list

    def get_last_section(self, model, type_class):
        """
        method used to find the the name of the first layer of a specific type section
        """
   
        # get the layers of the model and reverse the list
        layer_list = model.layers
        layer_list = layer_list[::-1]

        type_name = None
        type_flag = False

        # iterate over each layer in reverse order
        for layer in layer_list:
            layer_class = layer.__class__.__name__

            # if the class searched is the current layer class and
            # i haven't found yet the beginning of the section
            if type_class in layer_class and not type_flag:
                type_flag = True
                type_name = layer.name
            elif type_flag and type_class not in layer_class:
                # otherwise, if i'm in the section and the current layer type doesn't match,
                # stop the search, because i reached the end of the section
                break
            elif type_flag:
                # i'm inside the section, save the name of the current layer
                type_name = layer.name

        return type_name

    def all_layers(self, layer_list):
        """
        method used to check if all elements of a list are layers
        """
        return all(['.layers.' in str(type(i)) for i in layer_list])

    def count_layer_type(self, model, type):
        return [(i.__class__.__name__ == type) for i in model.layers].count(True)

    def any_batch(self, model):
        return any(['batch' in i.name for i in model.layers])

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

    last_conv = dynamicNet.get_last_section(model, 'Conv2D')
    linked_list = ['Conv2D', 'Activation', 'MaxPooling2D']
    model = dynamicNet.remove_section(model, last_conv, linked_list, True, False)
    last_conv = dynamicNet.get_last_section(model, 'Conv2D')
    model = dynamicNet.remove_section(model, last_conv, linked_list, True, False)
    model.summary()
    quit()

    new_section = [Conv2D(256, (2,2), padding="same"), Activation('relu')]
    model = dynamicNet.insert_section(model, 2, new_section, 'after', 'input')
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
