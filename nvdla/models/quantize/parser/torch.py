
import os
import sys
import copy

import torch.nn as nn

# Path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import qtorch as qt
from parser.utils import replace_module

########################################
### Quantizer

class qconvert():
    def __init__(self, bitwidth):
        self.bitwidth = bitwidth

    ## Replacement function
    def conv(self, module, name):
        # Replaces a standard nn.Conv2d layer with a quantized version
        # Uses the same parameters as the original layer (in/out channels, kernel size, etc.).
        newmodule = qt.Conv2d(
            in_channels=module.in_channels,
            out_channels = module.out_channels,
            kernel_size = module.kernel_size,
            stride = module.stride,
            padding = module.padding,
            dilation = module.dilation,
            bias = (module.bias is not None),
            bitwidth = self.bitwidth)

        # Copies the learned weights using load_state_dict
        newmodule.load_state_dict(module.state_dict())

        return newmodule
    
    def linear(self, module, name):
        newmodule = qt.Linear(
            in_features=module.in_features,
            out_features = module.out_features,
            bias = (module.bias is not None),
            bitwidth = self.bitwidth)
        
        newmodule.load_state_dict(module.state_dict())

        return newmodule


    ## Full Net Replacements -- Internal
    def replaceConv2d(self, model):
        return replace_module(model, nn.Conv2d, self.conv)

    def replaceLinear(self, model):
        return replace_module(model, nn.Linear, self.linear)


    ## Full Net Quantization
    def quantize(self, model):
        newmodel = copy.deepcopy(model)
        newmodel = self.replaceConv2d(newmodel)
        newmodel = self.replaceLinear(newmodel)
        return newmodel

