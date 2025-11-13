import numpy as np

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

def Quantize(array, scale, offset, dtype=np.int32):
    return np.floor(np.multiply(np.subtract(array, offset), scale)).astype(dtype)


def Dequantize(array, scale, offset, dtype=np.float32):
    return np.add(np.divide(array.astype(dtype), scale), offset)

    

