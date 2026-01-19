
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

# def filterArray(batch, filter):
#     matches = np.all(batch == filter, axis=tuple(range(1, batch.ndim)))
#     return batch[~matches]


##############################

def dutRelu(nvdla, ftens, ktens=None, bias=None, stride=0, padding=0, dilation=0, 
            funit='top', ftile=(0,0), fstart=0, fstop=0, fnumber=0,
            logfile=''):
    return nvdla.relu(ftens, seu=True, ftile=ftile, fstart=fstart, fstop=fstop, fnumber=fnumber, funit=funit, logfile=logfile)

def dutConv(nvdla, ftens, ktens, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1),
            funit='top', ftile=(0,0), fstart=0, fstop=0, fnumber=0,
            logfile=''):
    return nvdla.conv(ftens, ktens, stride=stride, padding=padding, dilation=dilation, seu=True, ftile=ftile, fstart=fstart, fstop=fstop, fnumber=fnumber, funit=funit, logfile=logfile)

def dutConvBias(nvdla, ftens, ktens, bias, stride=(1,1), padding=(0,0), dilation=(1,1),
            funit='top', ftile=(0,0), fstart=0, fstop=0, fnumber=0,
            logfile=''):
    return nvdla.convBias(ftens, ktens, bias, stride=stride, padding=padding, dilation=dilation, seu=True, ftile=ftile, fstart=fstart, fstop=fstop, fnumber=fnumber, funit=funit, logfile=logfile)

def dutLinear(nvdla, ftens, ktens, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1),
            funit='top', ftile=(0,0), fstart=0, fstop=0, fnumber=0,
            logfile=''):
    FT = nvdla.rollback(ftens)
    WT = nvdla.rollback(ktens)

    R = nvdla.conv(FT, WT, stride=(1,1), padding=(0,0), dilation=(1,1), seu=True, ftile=ftile, fstart=fstart, fstop=fstop, fnumber=fnumber, funit=funit, logfile=logfile)

    return nvdla.unroll(R) 

def dutLinearBias(nvdla, ftens, ktens, bias, stride=(1,1), padding=(0,0), dilation=(1,1),
            funit='top', ftile=(0,0), fstart=0, fstop=0, fnumber=0,
            logfile=''):
    FT = nvdla.rollback(ftens)
    WT = nvdla.rollback(ktens)

    R = nvdla.convBias(FT, WT, bias, stride=(1,1), padding=(0,0), dilation=(1,1), seu=True, ftile=ftile, fstart=fstart, fstop=fstop, fnumber=fnumber, funit=funit, logfile=logfile)

    return nvdla.unroll(R) 


##############################

def create_config(cfgfile, seulogfile, workdir):
    workfile = os.path.join(workdir, 'work_specs.yaml')

    # Load config
    with open(cfgfile, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Set Timeout
    with open(seulogfile, 'r') as f:
        seulog = yaml.load(f, Loader=yaml.SafeLoader)
        maxtime = 0
        for tileid, tile in seulog.items():
            maxtime = max(maxtime, tile['time']['total'])

        config['sim-timeout'] = maxtime + ((20*maxtime)//100) # Add 20% on total simtime
    
    # FIXME
    # # Set Buffer Sizes
    # with open(seulogfile, 'r') as f:
    #     seulog = yaml.load(f, Loader=yaml.SafeLoader)
    #     datbufsize = 0
    #     wtbufsize  = 0
    #     outbufsize = 0
    #     for tileid, tile in seulog.items():
    #         datbufsize = max(datbufsize, tile['mems']['dat']['size'])
    #         wtbufsize  = max(wtbufsize,  tile['mems']['wt']['size'])
    #         outbufsize = max(outbufsize, tile['mems']['out']['size'])

    #     config['cbuf']['datbuf']['num-banks']  = 1
    #     config['cbuf']['datbuf']['bank-depth'] = datbufsize
    #     config['cbuf']['wtbuf']['num-banks']   = 1
    #     config['cbuf']['wtbuf']['bank-depth']  = wtbufsize
    #     config['cacc']['num-banks']            = 1
    #     config['cacc']['bank-depth']           = outbufsize

    # Save & Exit
    with open(workfile, 'w') as f:
        yaml.dump(config, f, Dumper=yaml.SafeDumper)

    return workfile


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
        stride   = None
        padding  = None
        dilation = None

        Fmap = np.load(layer['inputs']['features'])
        Kmap = np.load(layer['inputs']['weights'])
        if ('bias' in layer['inputs']) and (layer['inputs']['bias'] != ''):
            Bias = np.load(layer['inputs']['bias'])
        else:
            Bias = None
        
    elif(layer['optype'] == 'relu'):
        stride   = None
        padding  = None
        dilation = None

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


def gen_testlist(exposelog, funit, fnumber):

    testlist = []
    
    # Generate Labels and Probs vectors
    with open(exposelog, 'r') as f:
        tiles  = yaml.load(f, Loader=yaml.SafeLoader)
        labels = []
        probs  = []

        for tileid, tile in tiles.items():
            labels.append(tileid)
            probs.append(tile['time']['perc'])

    # Distribute experiments
    fdistribution = np.bincount(np.random.choice(len(probs), fnumber, p=probs), minlength=len(probs))

    # Generate tests
    for it in range(len(labels)):
        tile      = tiles[labels[it]]
        fnumber_i = fdistribution[it]
        ftile     = (tile['b-tile'], tile['c-tile'])
        fstart    = tile['units'][funit]['time']['start']
        fstop     = tile['units'][funit]['time']['stop']

        testlist.append((labels[it], ftile, fstart, fstop, fnumber_i))

    return testlist


def seu_campaign(nvdla, layercfg, unit, exposelog, fnumber, output, logfile, store_golden=None, verbose=False):

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

    # Quantization
    qtype  = torch.int64

    Fscale, Foffset = SymCalibration(torch.from_numpy(Fmap), MinMaxObserver, nvdla.bitwidth)
    Kscale, Koffset = SymCalibration(torch.from_numpy(Kmap), MinMaxObserver, nvdla.bitwidth)

    qFmap  = Quantize(torch.from_numpy(Fmap), Fscale, Foffset, qtype).numpy()
    qKmap  = Quantize(torch.from_numpy(Kmap), Kscale, Koffset, qtype).numpy()
    if not Bias is None:
        qBias  = Quantize(torch.from_numpy(Bias), (Fscale*Kscale), (Foffset+Koffset), qtype).numpy()
    else:
        qBias = None

    # Get Layer
    print('-I: Selecting layer interface...')
    dut_interface = dutlayer(layercfg)

    # Get Tests List
    print('-I: Generating test list...')
    testlist = gen_testlist(exposelog, unit, fnumber)

    print(f'-I: Running {len(testlist)} tests over {len(exposelog)} tiles')
    if verbose:
        print('-I: Printing test list...')
        for test in range(len(testlist)):
            tileid, _, _, _, fnumber_i = testlist[test]
            print(f'# Tile {tileid} with {fnumber_i} faults')

    # Golden run
    if not store_golden is None and (store_golden != ''):
        print(f'-I: Running Golden test')
        start   = datetime.now()
        qGolden  = dut_interface(nvdla, qFmap, qKmap, qBias, stride, padding, dilation)[0]
        stop    = datetime.now()
        print(f'-I: Run golden convolution')
        # Dequantization
        ftype  = torch.float32
        Golden = Dequantize(torch.from_numpy(qGolden), (Fscale*Kscale), (Foffset+Koffset), ftype).numpy()
        np.save(store_golden, Golden)

    # SEU Campaign
    qErrors = None
    with open(logfile, 'w') as f1:
        pass

    print(f'-I: Starting SEU campaign...')

    for it in range(len(testlist)):
        
        tileid, ftile, fstart, fstop, fnumber_i = testlist[it]

        if (fnumber_i > 0):
            print(f'-I: Injecting {fnumber_i} faults in tile {tileid}...')
        
            workdir   = os.path.dirname(os.path.realpath(output))
            logfile_i = os.path.join(workdir, f'seulog_test{it}.csv')

            start     = datetime.now()
            testout   = dut_interface(nvdla, qFmap, qKmap, qBias, stride, padding, dilation, unit, ftile, fstart, fstop, fnumber_i, logfile=logfile_i)
            stop      = datetime.now()
                
            print(f'-I: Test {it}/{len(testlist)} concluded (elapsed: {(stop-start).total_seconds()*1000} ms)')

            if(testout.shape[0] != fnumber_i):
                print(f'-E: Expected tensors with {fnumber_i} outputs, but received of {testout.shape[0]} outputs')
                sys.exit('-E: Internal Campaign Error')

            if qErrors is None:
                qErrors = testout
            else:
                qErrors = np.concatenate((qErrors, testout), axis=0)

            with open(logfile, 'a') as f1:
                with open(logfile_i) as f2:
                    f1.writelines(f2)
    
            os.remove(logfile_i)

    print(f'-I: All test run, SEU campaign finished')

    # Dequantization
    if(qErrors.size > 0):
        ftype  = torch.float32
        Errors = Dequantize(torch.from_numpy(qErrors), (Fscale*Kscale), (Foffset+Koffset), ftype).numpy()
    else:
        Errors = None

    # Exit
    return Errors


##############################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config',   type=str, required=True, help='Config identifier file')
    parser.add_argument('--layer',    type=str, required=True, help='Injected layer settings')
    parser.add_argument('--seulog',   type=str, required=True, help='SEU exposure logfile')
    parser.add_argument('--output',   type=str, default='seu_results.npy', help='Output save file')
    parser.add_argument('--logfile',  type=str, default='seulog.csv', help='Output log file')
    parser.add_argument('--fnumber',  type=int, required=True, help='Number of injected fault')
    parser.add_argument('--funit',    type=str, default='top', choices=units, help='Injected fault unit')
    parser.add_argument('--verbose',  action='store_true', help='Enables additional log')
    parser.add_argument('--golden',   type=str, help='Store golden result')

    args = parser.parse_args()

    # Gen Working Config
    print('-I: Generating work config...')
    config = create_config(args.config, args.seulog, os.path.dirname(os.path.realpath(args.output)))

    # Init NVDLA
    print('-I: Init DUT NVDLA...')
    nvdla = nvdlaWrap(config, profiler=False)
    nvdla.core.clear_faults()

    res = seu_campaign(nvdla, args.layer, args.funit, args.seulog, args.fnumber, args.output, args.logfile, args.golden, args.verbose)

    if res is None:
        with open(args.output, 'w') as f1:
            pass
    else:
        np.save(os.path.join(args.output), res)
