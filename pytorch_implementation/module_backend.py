import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from components.backend_interface import BackendInterface

class ModuleBackend(BackendInterface):

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
        
    def get_flops(self, base_model, input_shapes):
        """
        Calculate FLOPs and parameters for the model.
        Counts only forward pass to match TensorFlow implementation.
        """
        model = base_model
        istrain = model.training
        model.eval()
        
        input_shapes = torch.randn(1, *input_shapes)
        input_shapes = input_shapes if isinstance(input_shapes, torch.Tensor) else torch.randn(input_shapes)

        flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
        with flop_counter:
            # Forward only (to match TensorFlow's forward-only counting)
            model(input_shapes)
            total_flops = flop_counter.get_total_flops()
            total_params = sum(p.numel() for p in model.parameters())

        if istrain:
            model.train()

        return total_flops, total_params
    
    def get_latency(self, model, input_shapes):
        """Latency measurement can be implemented here if needed."""
        return None
    
    def get_params(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        return total_params