import sys
import os

import numpy as np

import yaml
import math

##############################
## Profilation Routines

# CONV Operations
def getHWConvs(nvdla, datTensor, wtTensor):
    K,C,R,S = size(wtTensor)
    B,C2,H,W = size(datTensor)
    AtomicK = nvdla.config['cmac']['atomic-k']
    HWBatch = nvdla.config['cmac']['batch-size']
    return int(math.ceil(K/AtomicK)*math.ceil(B / HWBatch))

# Get Entries
def getEntries(nvdla, tensor, thpt):
    B, C, H, W = size(tensor)
    HWBatch = nvdla.config['cmac']['batch-size']
    return int(H*W*math.ceil(C/thpt)*min(B, HWBatch))

def getDATentries(nvdla, datTensor):
    B, C, H, W = size(datTensor)
    AtomicC = nvdla.config['cmac']['atomic-c']
    HWBatch = nvdla.config['cmac']['batch-size']
    return int(H*W*math.ceil(C/AtomicC)*min(B, HWBatch))

def getWTentries(nvdla, wtTensor):
    K, C, R, S = size(wtTensor)
    AtomicC = nvdla.config['cmac']['atomic-c']
    AtomicK = nvdla.config['cmac']['atomic-k']
    return int(R*S*math.ceil(C/AtomicC)*min(K, AtomicK))

def getOUTentries(nvdla, datTensor, wtTensor, stride, padding, dilatation):
    B, C, H, W  = size(datTensor)
    K, C2, R, S = size(wtTensor)
    if(C != C2):
        raise ValueError(f'-E: mismatching channel sizes: {C} != {C2}')

    S_ = S
    R_ = R
    C_ = K
    W_ = ((2*padding[0] +W -S_)/stride) +1
    H_ = ((2*padding[1] +H -R_)/stride) +1

    return W_ * H_


# Transactions
def getTransactions(nvdla, tensor):
    wordsize  = nvdla.config['primary-dbb']['wordsize']
    precision = nvdla.bits
    size      = math.prod(tensor)*precision
    return int(math.ceil(size/wordsize))


# BUF Usage
def cbufUsage(nvdla, datTensor, wtTensor, stride, padding, dilatation):
    datEntries  = getDATentries(nvdla, datTensor) +2*padding[0] +2*padding[1]
    wtEntries   = getWTentries(nvdla, wtTensor)
    entries     = datEntries + wtEntries
    cbufEntries = nvdla.config['cbuf']['banks'] * nvdla.config['cbuf']['bank-depth']
    usage       = 100*(entries / cbufEntries)
    return usage, int(entries), int(cbufEntries)

def caccUsage(nvdla, datTensor, wtTensor, stride, padding, dilatation):
    entries     = getOUTentries(nvdla, datTensor, wtTensor, stride, padding, dilatation)
    caccEntries = nvdla.config['cacc']['banks'] * nvdla.config['cacc']['bank-depth']
    usage       = 100*(entries / caccEntries)
    return usage, int(entries), int(caccEntries)


# Get Atomics
def getAtomics(nvdla, datTensor, wtTensor, stride, padding, dilatation):
    B, C, H, W  = size(datTensor)
    K, C2, R, S = size(wtTensor)
    if(C != C2):
        raise ValueError(f'-E: mismatching channel sizes: {C} != {C2}')

    AtomicC = nvdla.config['cmac']['atomic-c']

    S_ = S
    R_ = R
    C_ = K
    W_ = ((2*padding[0] +W -S_)/stride) +1
    H_ = ((2*padding[1] +H -R_)/stride) +1

    StripeSize = W_ * H_
    BlockSize  = StripeSize * S * R

    return int(BlockSize*math.ceil(C / AtomicC))


# Execution Time
def getCLKperiod(technology='xilinx', busWordSize=1):
    if(technology == 'xilinx'):    # Time period in NS
        clkPeriod = 4.0
        accesLatency = 33.5
    elif(technology == 'raw'):     # Just counts the raw values
        clkPeriod = 1
        accesLatency = 1
    else:
        raise ValueError(f'-E: {technology} is not a supported technology type')

    return (clkPeriod, accesLatency)

def getCONVTime(nvdla, datTensor, wtTensor, stride, padding, dilatation, backend='nvdla-v1', technology='xilinx'):
    convs = getHWConvs(nvdla, datTensor, wtTensor)
    convAtomics = getAtomics(nvdla, datTensor, wtTensor, stride, padding, dilatation)
    if(backend == 'nvdla-v1'):
        atomicCLKs = 7
    elif(backend == 'nvdla-v2'):
        atomicCLKs = 6
    else:
        raise ValueError(f'-E: {backend} is not a supported backend type')

    outEntries = getOUTentries(nvdla, datTensor, wtTensor, stride, padding, dilatation)
    return int(convs*(convAtomics*atomicCLKs + outEntries))*getCLKperiod(technology=technology)[0]

def getSDPTime(nvdla, tensor, backend='nvdla-v1', technology='xilinx'):
    if((not nvdla.config['sdp']['bs']['en']) and (not nvdla.config['sdp']['bn']['en']) and (not nvdla.config['sdp']['bn']['en'])):
        raise ValueError(f'-E: no units available in SDP')

    latency = 0
    if(backend == 'nvdla-v1'):
        xCLKs = 3
        yCLKs = 3
    elif(backend == 'nvdla-v2'):
        xCLKs = 3
        yCLKs = 3
    else:
        raise ValueError(f'-E: {backend} is not a supported backend type')

    if(nvdla.config['sdp']['bs']['en']):
        bsThpt = nvdla.config['sdp']['bs']['thpt']
        latency += getEntries(nvdla, tensor, bsThpt)*xCLKs

    if(nvdla.config['sdp']['bn']['en']):
        bnThpt = nvdla.config['sdp']['bn']['thpt']
        latency += getEntries(nvdla, tensor, bnThpt)*xCLKs

    if(nvdla.config['sdp']['ew']['en']):
        ewThpt = nvdla.config['sdp']['ew']['thpt']
        latency += getEntries(nvdla, tensor, ewThpt)*yCLKs

    return int(latency)*getCLKperiod(technology=technology)[0]

def getPDPTime(nvdla, tensor, backend='nvdla-v1', technology='xilinx'):
    if(not nvdla.config['pdp']['en']):
        raise ValueError(f'-E: no units available in SDP')

    if(backend == 'nvdla-v1'):
        pdpCLKs = 5
    elif(backend == 'nvdla-v2'):
        pdpCLKs = 5
    else:
        raise ValueError(f'-E: {backend} is not a supported backend type')

    return int(getEntries(nvdla, tensor, nvdla.config['pdp']['thpt']) *pdpCLKs)*getCLKperiod(technology=technology)[0]


# Function to get elements from a list
def size(value):
    if len(value) == 4:
        return value[0], value[1], value[2], value[3]
    elif len(value) == 2:
        return value[0], value[1]
    

##############################
## Config Wrapper


class nvdla:
    def __init__(self, config) -> None:
        with open(config) as stream:
            try:
               self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        if(self.config['dtype'] == 'int8'):
            self.bits = 8
            self.isFloat = False
        elif(self.config['dtype'] == 'int16'):
            self.bits = 16
            self.isFloat = False
        elif(self.config['dtype'] == 'int32'):
            self.bits = 32
            self.isFloat = False
        elif(self.config['dtype'] == 'fp16'):
            self.bits = 16
            self.isFloat = True
        elif(self.config['dtype'] == 'fp32'):
            self.bits = 32
            self.isFloat = True
        elif(self.config['dtype'] == 'bf16'):
            self.bits = 16
            self.isFloat = True
        else:
            raise ValueError(f'-E: {self.config["dtype"]} is not supported')

        #print(f'-I({__file__}): Loaded config: {self.config["name"]}')

##############################
## Extension of Torch Layers

## Conv2d
class Conv2d():
    """
    Conv2d Profiler
    """
    def __init__(self, nvdla, file, layerid, out, in_channels, out_channels, kernel_size, stride=1, padding=0, dilatation=1, bias=True, out_offset=0, out_rshift=0):
        #super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.logfile    = file
        self.nvdla      = nvdla
        self.stride     = stride
        self.padding    = padding if padding is tuple else (padding, padding)
        self.dilatation = dilatation
        self.biasTrue   = bias
        self.offset     = out_offset
        self.rshift     = out_rshift
        self.layerid    = layerid
        
        # new informations for tensorflow
        self.kernel = kernel_size
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.out = out
        self.bias = [out_channels]

    def forward(self, input):
        # FIXME: Feature now tupported
        B,C,H,W = size(input)

        if(B > 1):
            raise ValueError(f'-E: Batch Size > 1 is not currently supported')

        # CONV execution
        out = self.out
        self.weight = [self.out_ch, self.in_ch, self.kernel, self.kernel]

        # Profile HW Ops
        convs = getHWConvs(self.nvdla, input, self.weight)

        # Profile Latency
        latency = getCONVTime(self.nvdla, input, self.weight, self.stride, self.padding, self.dilatation)
        latency += getSDPTime(self.nvdla, out)
        
        # Profile Mem Transactions
        readT  = getTransactions(self.nvdla, input)
        readT += getTransactions(self.nvdla, self.weight)
        if(self.biasTrue):
            readT += getTransactions(self.nvdla, self.bias)
        writeT = getTransactions(self.nvdla, out)
        
        # Scale Mem Transactions
        readT  *= getCLKperiod(technology='xilinx')[1]
        writeT *= getCLKperiod(technology='xilinx')[1]

        # Profile BUF Usage
        cbuf = cbufUsage(self.nvdla, input, self.weight, self.stride, self.padding, self.dilatation)
        cacc = caccUsage(self.nvdla, input, self.weight, self.stride, self.padding, self.dilatation)
        
        return latency

## Linear
class Linear():
    """
    Linear Profiler
    """
    def __init__(self, nvdla, file, layerid, out, in_features, out_features, bias=True, out_offset=0, out_rshift=0):
        #super().__init__(in_features, out_features, bias=bias)
        self.logfile    = file
        self.nvdla      = nvdla
        self.biasTrue   = bias
        self.offset     = out_offset
        self.rshift     = out_rshift
        self.layerid    = layerid
        
        # new informations for tensorflow
        self.in_f = in_features
        self.out_f = out_features
        self.out = out
        self.bias = [out_features]

    def forward(self, input):
        # FIXME: Feature now tupported
        self.weight = [self.out_f, self.in_f]
        
        B,C  = size(input)
        K,C2 = size(self.weight)
        
        if(B > 1):
            raise ValueError(f'-E: Batch Size > 1 is not currently supported')

        # LINEAR execution
        out = self.out

        # Reshape
        inputT = [B, C] + [1, 1]
        weightT = [K, C2] + [1, 1]
        outT = [B, K] + [1, 1]

        # Profile HW Ops
        convs = getHWConvs(self.nvdla, inputT, weightT)

        # Profile Latency
        latency = getCONVTime(self.nvdla, inputT, weightT, 1, (0,0), 1)
        latency += getSDPTime(self.nvdla, outT)

        # Profile Mem Transactions
        readT  = getTransactions(self.nvdla, input)
        readT += getTransactions(self.nvdla, self.weight)
        if(self.biasTrue):
            readT += getTransactions(self.nvdla, self.bias)
        writeT = getTransactions(self.nvdla, out)

        # Scale Mem Transactions
        readT  *= getCLKperiod(technology='xilinx')[1]
        writeT *= getCLKperiod(technology='xilinx')[1]
        
        # Profile BUF Usage
        cbuf = cbufUsage(self.nvdla, inputT, weightT, 1, (0,0), 1)
        cacc = caccUsage(self.nvdla, inputT, weightT, 1, (0,0), 1)

        return latency

