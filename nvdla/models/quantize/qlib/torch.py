import torch

########################################
### Quantization Routine

def MinMaxObserver(array):
    min = array.min()
    max = array.max()

    return min, max

def SymCalibration(array, observer, bitwidth):
    alpha, beta = observer(array)
    upbound = max(abs(alpha), abs(beta))
    if (upbound == 0):
        scale = 1
        offset = 0
    else:
        Gamma = (2**(bitwidth-1))-1
        scale = Gamma / upbound
        offset = 0

    return scale, offset

def Quantize(array, scale, offset, dtype=torch.int32):
    return torch.floor(torch.multiply(torch.subtract(array, offset), scale)).to(dtype)


def Dequantize(array, scale, offset, dtype=torch.float32):
    return torch.add(torch.divide(array.to(dtype), scale), offset)

    

