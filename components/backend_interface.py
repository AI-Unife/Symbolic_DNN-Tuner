class BackendInterface:
    def build_lenet(self):
        raise NotImplementedError

    def get_layers(self, model):
        raise NotImplementedError

    def get_input_shape(self, layer):
        raise NotImplementedError

    def get_output_shape(self, layer):
        raise NotImplementedError

    def get_layer_info(self, layer):
        """Returns (standard_layer_type: str, parameters: dict)"""
        raise NotImplementedError
    
    def get_flops(self, model, input_shapes):
        raise NotImplementedError