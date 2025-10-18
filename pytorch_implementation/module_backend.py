import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from components.backend_interface import BackendInterface

class ModuleBackend(BackendInterface):
    def build_lenet(self):
        """Build LeNet model using Sequential"""
        model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Sigmoid(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )
        return model

    def get_layers(self, model):
        """For Sequential, just return the layers"""
        return list(model.modules())[1:]

    def get_input_shape(self, layer):
        """PyTorch doesn't store input shape."""
        return None

    def get_layer_info(self, layer):
        """Recognize layer types"""
        if isinstance(layer, nn.Conv2d):
            return 'conv2d', {
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size[0],
                'stride': layer.stride[0],
                'padding': 'valid' if layer.padding[0] == 0 else 'same',
                'bias': layer.bias is not None
            }
        elif isinstance(layer, nn.Linear):
            return 'dense', {
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'bias': layer.bias is not None
            }
        else:
            return 'other', {}
        
    def get_flops(self, model, input_shapes):
        istrain = model.training
        model.eval()
        
        input_shapes = (1, input_shapes[2], input_shapes[0], input_shapes[1])
        input_shapes = input_shapes if isinstance(input_shapes, torch.Tensor) else torch.randn(input_shapes)

        flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
        with flop_counter:
            model(input_shapes).sum().backward() # Forward and backward
            # model(input_shapes) # Forward only
            total_flops =  flop_counter.get_total_flops()

        if istrain:
            model.train()

        return total_flops