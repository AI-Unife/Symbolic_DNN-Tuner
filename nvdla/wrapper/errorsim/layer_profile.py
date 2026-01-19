
import os
import sys
import argparse
import yaml

import numpy as np
import torch
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from wrapper import nvdlaWrap
from quantize.qlib.torch import MinMaxObserver, SymCalibration, Quantize, Dequantize

##############################

units=[
    'top',
    'top.conv',
    'top.conv.cdma',
    'top.conv.cbuf',
    'top.conv.cbuf.datbuf',
    'top.conv.cbuf.wtbuf',
    'top.conv.csb',
    'top.conv.csc',
    'top.conv.dl',
    'top.conv.wl',
    'top.conv.cmac',
    'top.conv.cacc',
    'top.conv.dbuf',
    'top.sdp',
    'top.sdp.csb',
    'top.sdp.core'
]

layers=[
    'conv',
    'convbias',
    'linear',
    'linearbias',
    'relu'
]

##############################

def cfgCheck(configfile):
    with open(configfile, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    
    if(config['dat-type'] == config['wt-type']):
        return True
    else:
        return False

def filterArray(batch, filter):
    matches = np.all(batch == filter, axis=tuple(range(1, batch.ndim)))
    return batch[~matches]


##############################

def dutRelu(nvdla, ftens, ktens=None, bias=None, stride=0, padding=0, dilation=0, 
            logfile=''):
    return nvdla.relu(ftens, logfile=logfile)

def dutConv(nvdla, ftens, ktens, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1),
            logfile=''):
    return nvdla.conv(ftens, ktens, stride=stride, padding=padding, dilation=dilation, logfile=logfile)

def dutConvBias(nvdla, ftens, ktens, bias, stride=(1,1), padding=(0,0), dilation=(1,1),
            logfile=''):
    return nvdla.convBias(ftens, ktens, bias, stride=stride, padding=padding, dilation=dilation, logfile=logfile)

def dutLinear(nvdla, ftens, ktens, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1),
            logfile=''):
    FT = nvdla.rollback(ftens)
    WT = nvdla.rollback(ktens)

    R = nvdla.conv(FT, WT, stride=(1,1), padding=(0,0), dilation=(1,1), logfile=logfile)

    return nvdla.unroll(R) 

def dutLinearBias(nvdla, ftens, ktens, bias, stride=(1,1), padding=(0,0), dilation=(1,1),
            logfile=''):
    FT = nvdla.rollback(ftens)
    WT = nvdla.rollback(ktens)

    R = nvdla.conv(FT, WT, bias, stride=(1,1), padding=(0,0), dilation=(1,1), slogfile=logfile)

    return nvdla.unroll(R) 


##############################

def load_layer(layercfg):
    with open(layercfg, 'r') as f:
        layer = yaml.load(f, Loader=yaml.SafeLoader)
    
    if(layer['optype'] == 'conv'):
        stride   = (layer['params']['stride-y'], layer['params']['stride-x'])
        padding  = (layer['params']['padding-y'], layer['params']['padding-x'])
        dilation = (layer['params']['dilation-y'], layer['params']['dilation-x'])

        Fmap = np.load(layer['inputs']['features'])
        Kmap = np.load(layer['inputs']['weights'])
        if ('bias' in layer['inputs']) and (layer['inputs']['bias'] != ''):
            Bias = np.load(layer['inputs']['bias'])
        else:
            Bias = None
        
    elif(layer['optype'] == 'linear'):
        stride = (1,1)
        padding = (0,0)
        dilation = (1,1)

        Fmap = np.load(layer['inputs']['features'])
        Kmap = np.load(layer['inputs']['weights'])
        if ('bias' in layer['inputs']) and (layer['inputs']['bias'] != ''):
            Bias = np.load(layer['inputs']['bias'])
        else:
            Bias = None
        
    elif(layer['optype'] == 'relu'):
        stride = (1,1)
        padding = (0,0)
        dilation = (1,1)

        Fmap = np.load(layer['inputs']['features'])

    else:
        raise ValueError('Unsupported optype') 

    return Fmap, Kmap, Bias, stride, padding, dilation


def dutlayer(layercfg):
    with open(layercfg, 'r') as f:
        layer = yaml.load(f, Loader=yaml.SafeLoader)
    
    if(layer['optype'] == 'conv'):
        if ('bias' in layer['inputs']) and (layer['inputs']['bias'] != ''):
            dut_interface = dutConvBias
        else:
            dut_interface = dutConv
        
    elif(layer['optype'] == 'linear'):
        if ('bias' in layer['inputs']) and (layer['inputs']['bias'] != ''):
            dut_interface = dutLinearBias
        else:
            dut_interface = dutLinear
        
    elif(layer['optype'] == 'relu'):
            dut_interface = dutRelu

    else:
        raise ValueError('Unsupported optype') 

    return dut_interface


def profile(nvdla, layercfg, logfile, verbose=False):

    print(f'-I: Target NVDLA configuration: {nvdla}')

    # Load Inputs
    print(f'-I: Loading input layerfile {layercfg}...')
    Fmap, Kmap, Bias, stride, padding, dilation = load_layer(layercfg)

    print(f'-I: Feature Tensor Shape {Fmap.shape}')
    print(f'-I: Weight Tensor Shape  {Kmap.shape}')
    if not Bias is None:
        print(f'-I: Bias Vector Shape    {Bias.shape}')
    print(f'-I: Stride               {stride}')
    print(f'-I: Padding              {padding}')
    print(f'-I: Dilation             {dilation}')

    # Get Layer
    dut_interface = dutlayer(layercfg)

    # Quantization
    qtype  = torch.int64

    Fscale, Foffset = SymCalibration(torch.from_numpy(Fmap), MinMaxObserver, nvdla.bitwidth)
    Kscale, Koffset = SymCalibration(torch.from_numpy(Kmap), MinMaxObserver, nvdla.bitwidth)

    qFmap  = Quantize(torch.from_numpy(Fmap), Fscale, Foffset, qtype).numpy()
    qKmap  = Quantize(torch.from_numpy(Kmap), Kscale, Koffset, qtype).numpy()
    if not Bias is None:
        qBias = Quantize(torch.from_numpy(Bias), (Fscale*Kscale), (Foffset+Koffset), qtype).numpy()
    else:
        qBias = None

    # SEU Campaign
    print(f'-I: Starting layer profilation...')

    start  = datetime.now()
    qPsums = dut_interface(nvdla, qFmap, qKmap, qBias, stride, padding, dilation, logfile)
    stop   = datetime.now()

    print(f'-I: Layer profiled (elapsed: {(stop-start).total_seconds()*1000} ms)')

    # Dequantization
    ftype  = torch.float32
    Psums = Dequantize(torch.from_numpy(qPsums), (Fscale*Kscale), (Foffset+Koffset), ftype).numpy()

    if verbose:
        print(f'-I: Output Tensor Shape  {Psums.shape}')


##############################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help='Config identifier file')
    parser.add_argument('--verbose', action='store_true', help='Enables additional log')
    parser.add_argument('--layer', type=str, required=True, help='Injected layer settings')
    parser.add_argument('--output', type=str, default='./profiler.yaml', help='Output save dir')

    args = parser.parse_args()

    # Init NVDLA
    nvdla = nvdlaWrap(args.config, profiler=True)
    nvdla.core.clear_faults()

    profile(nvdla, args.layer, args.output, args.verbose)
