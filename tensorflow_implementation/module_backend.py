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

    def get_flops(self, model, input_shapes):
        from tensorflow_implementation.flops import flops_calculator
        return flops_calculator.analyze_model(model)
