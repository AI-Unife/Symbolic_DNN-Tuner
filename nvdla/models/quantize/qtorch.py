
import torch
import torch.nn as nn

from qlib.torch import MinMaxObserver, SymCalibration, Quantize, Dequantize

########################################
### Quantization Utils

def symQuantize(tensor, bitwidth, dtype=torch.int64):
    scale, offset = SymCalibration(tensor, MinMaxObserver, bitwidth)
    qTensor = Quantize(tensor, scale, offset, dtype)
    return qTensor, scale, offset


########################################
### Extension of Torch Layers

## Conv2d
class Conv2d(nn.Conv2d):
    """
    Conv2d with Bias layer that exploits the available nvdlasim layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, bitwidth=8):

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.stride     = stride
        self.padding    = padding
        self.dilation   = dilation
        self.groups     = groups
        self.bitwidth   = bitwidth

    def __str__(self):
        superstr = super().__str__()
        return f'[qint{self.bitwidth}]{superstr}'

    def __repr__(self):
        superstr = super().__repr__()
        return f'[qint{self.bitwidth}]{superstr}'

    def forward(self, input):            
        ftype = input.dtype
        qtype = torch.float32 # Use float backend for GPU

        qFT, fScale, fOff = symQuantize(input.detach(), self.bitwidth, qtype)
        qWT, wScale, wOff = symQuantize(self.weight.data, self.bitwidth, qtype)

        if(self.bias is None):
            qPsums = torch.nn.functional.conv2d(qFT, qWT, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            qB = Quantize(self.bias.data, (fScale*wScale), (fOff+wOff), qtype)
            qPsums = torch.nn.functional.conv2d(qFT, qWT, bias=qB, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return Dequantize(qPsums, (fScale*wScale), (fOff+wOff), dtype=ftype)


## Linear
class Linear(nn.Linear):
    """
    Linear with Bias layer that exploits the available nvdlasim layer
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, bitwidth=8):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.bitwidth   = bitwidth

    def __str__(self):
        superstr = super().__str__()
        return f'[qint{self.bitwidth}]{superstr}'

    def __repr__(self):
        superstr = super().__repr__()
        return f'[qint{self.bitwidth}]{superstr}'

    def forward(self, input):            
        ftype = input.dtype
        qtype = torch.float32 # Use float backend for GPU

        qFT, fScale, fOff = symQuantize(input.detach(), self.bitwidth, qtype)
        qWT, wScale, wOff = symQuantize(self.weight.data, self.bitwidth, qtype)

        if(self.bias is None):
            qPsums = torch.nn.functional.linear(qFT, qWT, bias=None)
        else:
            qB = Quantize(self.bias.data, (fScale*wScale), (fOff+wOff), qtype)
            qPsums = torch.nn.functional.linear(qFT, qWT, bias=qB)

        return Dequantize(qPsums, (fScale*wScale), (fOff+wOff), dtype=ftype)