class BackendInterface:
    def build_lenet(self):
        raise NotImplementedError

    def get_flops(self, model, input_shapes):
        raise NotImplementedError
