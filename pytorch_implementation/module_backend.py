

import torch
import torch.nn as nn
import tensorflow as tf
from torch.utils.flop_counter import FlopCounterMode

from components.backend_interface import BackendInterface
from components.colors import colors

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


def isConv2d(module) -> bool:
    """Check if the module is a Conv2d layer."""
    return isinstance(module, nn.Conv2d)
    

def get_dummy_input(self)->torch.Tensor:
    """Generate a dummy input tensor for profiling."""
    return torch.randint(0, 256, (1, 3, 32, 32))


def extract_conv_layers(self, model, input_shape):
    """
    Extract Conv2D layer input/output shapes.
    Returns list of (input_shape, output_shape).
    """

    conv_layers = []
    
    # PYTORCH
    if hasattr(model, "modules"):

        def hook_fn(module, input, output):
            if isinstance(module, nn.Conv2d):
                conv_layers.append((tuple(input[0].shape),
                                    tuple(output.shape)))

        hooks = []

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(hook_fn))

        try:
            B_eff = 1
            C_in, H_in, W_in = input_shape

            dummy_input = torch.randn(B_eff, C_in, H_in, W_in)

            model.eval()

            with torch.no_grad():
                model(dummy_input)

        finally:
            for h in hooks:
                h.remove()

    # TENSORFLOW / KERAS
    elif hasattr(model, "layers"):
        for layer in model.layers:

            if isinstance(layer, tf.keras.layers.Conv2D):

                try:
                    input_shape = layer.input_shape
                except:
                    input_shape = layer.input.shape

                try:
                    output_shape = layer.output_shape
                except:
                    output_shape = layer.output.shape

                conv_layers.append((input_shape, output_shape))

    else:
        raise TypeError("Unsupported model type")

    return conv_layers