import json
from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks, preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
from components.colors import colors
from components.neural_network import NeuralNetwork
from components.dataset import TunerDataset
from components.model_interface import LayerTypes, TunerModel, LayerSpec, InsertPosition, Params
from tensorflow_implementation.model import TFModel

# class wrapper used to add the functionality of the layer wise learning rate
@keras.utils.register_keras_serializable()
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "optimizer": keras.optimizers.serialize(self._optimizer),
                "multiplier": self._multiplier,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        optimizer_config = config.pop("optimizer")
        multiplier = config.pop("multiplier")
        optimizer = keras.optimizers.deserialize(optimizer_config)
        return cls(optimizer=optimizer, multiplier=multiplier, **config)


class NeuralNetwork (NeuralNetwork):
    def __init__(self, dataset: TunerDataset):
        super().__init__(dataset)

        self.activation_map = {
            "relu": "relu",
            "elu": "elu",
            "selu": "selu",
            "swish": "swish"
        }

    def from_checkpoint(self, checkpoint):
        checkpoint_json = json.dumps(checkpoint)
        keras_model = models.model_from_json(checkpoint_json)
    
        tf_model = TFModel.__new__(TFModel)   # bypass __init__
        tf_model.model = keras_model
        tf_model.input_shape = keras_model.input_shape[1:]
        tf_model.n_classes = keras_model.output_shape[-1]
        # tf_model.params = checkpoint.get("params")
        # tf_model.activation = checkpoint.get("activation")
        tf_model.create_specs()

        return tf_model

    def from_scratch(self, input_shape, n_classes, params):
        return TFModel(input_shape, params, n_classes, params["activation"])

    def insert_batch(self, model: TunerModel, params):
        """
        method used for inserting batchNormalization operations
        :param model params n_conv: insert in 'model' batchNormalization and regularization with 'params' parameters
        :return: model with batchNormalization and regularization
        """

        # If BatchNormalization is already in the model, return it
        if (self.dnet.count_layer_type(model, LayerTypes.BatchNormalization1D) > 1 or
                self.dnet.count_layer_type(model, LayerTypes.BatchNormalization2D) > 1):
            return model

        # otherwise add regularization to each convolutional layer
        if self.rgl:
            for layer in model.model.layers:
                if 'Conv2D' in layer.__class__.__name__:
                    layer.kernel_regularizer = regularizers.l2(params['reg'])

        # Collect activations to which BatchNormalization should be added
        activation_list = [layer.type for layer in model.layers.values() if layer.is_activation and layer.type != LayerTypes.Softmax]

        new_section = [
            LayerSpec(
                type=LayerTypes.BatchNormalization
            )
        ]

        # apply batchNormalization to all saved activations
        return self.dnet.insert_section(model, 1, new_section, InsertPosition.After, activation_list)

    def training(self, params, new, new_fc, new_conv, rem_conv, rem_fc, da, space):
        """
        Function for compiling and running training
        :param params, new, new_fc, new_conv, rem_conv, da, space: parameters to indicate a possible operation on the network structure and hyperparameter search space
        :return: training history, trained model and and performance evaluation score 
        """
        # build neural network
        self.model = self.build_network(params, new)
        print("\n\nBuilt network with new:", new)
        print("\n\n| ------------- LayerSpecs ------------- |")
        for k in self.model.layers:
            print(self.model.layers[k])

        # print(self.model.model.summary())

        # self.model = self.insert_batch(self.model, params)
        # params['new_fc'] = 512
        # self.model = self.insert_fc_section(self.model, params, 1)
        # self.model = self.insert_conv_section(self.model, params, 1)
        # self.model = self.remove_conv_section(self.model)
        # self.model = self.remove_fc_section(self.model)

        # print(self.model.model.summary())
        # exit()
        
        try:
            # try adding or removing a layer in the neural network based on the anomalies diagnosis
            if new or new_fc or new_conv or rem_conv:
                # if the flag for the addition of a dense layer is true
                if new_fc:
                    if new_fc[0]:
                        self.dense = True
                        self.model = self.insert_fc_section(self.model, params, new_fc[1])
                # if the flag for the addition of regularization is true
                if new:
                    self.rgl = True
                    self.dense = False
                    self.model = self.insert_batch(self.model, params)
                # if the flag for the addition of a convolutional layer
                if new_conv:
                    if new_conv[0]:
                        self.conv = True
                        self.dense = False
                        self.rgl = False
                        self.model = self.insert_conv_section(self.model, params, new_conv[1])
                # if the flag for the removal of a convolutional layer is true
                if rem_conv:
                    self.conv = False
                    self.dense = False
                    self.rgl = False
                    self.model = self.remove_conv_section(self.model, params)
                # if the flag for the removal of a dense layer is true
                if rem_fc:
                    self.conv = False
                    self.dense = False
                    self.rgl = False
                    self.model = self.remove_fc_section(self.model, params)

        except Exception as e:
            print(colors.FAIL, "Error modifying network:", e, colors.ENDC)
        
        # print the structure of the neural network and save it in a json file,
        # using the current time as identifier of the model
        print(self.model.model.summary())

        model_name_id = time()
        model_json = self.model.model.to_json()
        model_name = "Model/model-{}.json".format(model_name_id)
        with open(model_name, 'w') as json_file:
            json_file.write(model_json)

        # save the id of the model in an attribute, so that it can be used later to save an associated db with the same id
        self.last_model_id = model_name_id

        # try to load a set of weights 
        try:
            self.model.model.load_weights("Weights/weights.h5")
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
        trainable = [(layer.path if new_keras else layer.name).split('/')[0] for layer in self.model.model.trainable_variables]
        # for each successive variable, i'll have a reduction by a factor of sqrt(2)
        current_mul = 1
        lr_factor = 1.414213
        # iterate over each trainable layer, skipping one (kernel and bias pairs)
        for layer in trainable[::2]:
            # get layer class name
            layer_type = self.model.model.get_layer(layer).__class__.__name__
            # if the current layer is a type on which we want to apply a multiplier
            if layer_type in ['Conv2D']:
                multiplier |= {layer : current_mul}
                current_mul /= lr_factor

        opt = LayerWiseLR(opt, multiplier, learning_rate=params['learning_rate'])

        self.model.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
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
            history = self.model.model.fit(
                datagen.flow(self.train_data, self.train_labels, batch_size=params['batch_size']), epochs=self.epochs,
                verbose=1, validation_data=(self.test_data, self.test_labels),
                callbacks=[tensorboard, reduce_lr, es1, es2]).history
        else:
            # train the network without data augmentation
            history = self.model.model.fit(self.train_data, self.train_labels, epochs=self.epochs, batch_size=params['batch_size'],
                                verbose=1,
                                validation_data=(self.test_data, self.test_labels),
                                callbacks=[tensorboard, reduce_lr, es1, es2]).history

        # evaluates model performance on test data
        score = self.model.model.evaluate(self.test_data, self.test_labels)

        # save the neural network weights and then reload them from the same json file you just saved
        # this avoids errors because of the changes in the network structure before training
        weights_name = "Weights/weights-{}.weights.h5".format(model_name_id)
        self.model.model.save_weights(weights_name)
        self.model.model.save("dashboard/model/model.keras")
        
        f = open("Model/model-{}.json".format(model_name_id))
        self.model = self.from_checkpoint(json.load(f))
        self.model.model.load_weights(weights_name)

        print("\n\n Reloaded model to avoid errors.\n\n")

        # score is [val_loss, accuracy]
        return score, history, self.model 
