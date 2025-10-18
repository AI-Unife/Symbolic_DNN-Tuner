from tensorflow.keras import layers, models
from components.backend_interface import BackendInterface

class ModuleBackend(BackendInterface):
    def build_lenet(self):
        model = models.Sequential()
        model.add(layers.Conv2D(6, 5, activation='tanh', padding="same", input_shape=(28, 28, 1)))
        model.add(layers.AveragePooling2D(2))
        model.add(layers.Activation('sigmoid'))
        model.add(layers.Conv2D(16, 5, activation='tanh'))
        model.add(layers.AveragePooling2D(2))
        model.add(layers.Activation('sigmoid'))
        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='tanh'))
        model.add(layers.Dense(84, activation='tanh'))
        model.add(layers.Dense(10, activation='softmax'))
        return model

    def get_layers(self, model):
        return model.layers

    def get_input_shape(self, layer):
        return layer.input.shape

    def get_output_shape(self, layer):
        return layer.output.shape

    def get_layer_info(self, layer):
        """Translate TensorFlow layers to common standard types"""
        tf_layer_type = layer.__class__.__name__

        if tf_layer_type == 'Conv2D':
            return 'conv2d', {
                'in_channels': layer.input.shape[-1],
                'out_channels': layer.output.shape[-1],
                'kernel_size': layer.kernel_size[0],
                'stride': layer.strides[0],
                'padding': 'valid' if layer.padding == 'valid' else 'same',
                'bias': layer.use_bias
            }
        elif tf_layer_type == 'Dense':
            return 'dense', {
                'in_features': layer.input.shape[-1],
                'out_features': layer.output.shape[-1],
                'bias': layer.use_bias
            }
        else:
            return 'other', {}  # unknown / ignored layers
        
    def get_flops(self, model, input_shapes):
        from tensorflow_implementation.flops import flops_calculator
        return flops_calculator.analyze_model(model)