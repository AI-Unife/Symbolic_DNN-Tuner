
import os
import argparse
import torch
from datetime import datetime

from utils import test
from zoo import models_factory, datasets, networks

from quantize.parser.torch import qconvert as converter

###################################################

if __name__ == '__main__':

    #### Args Parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',          choices=networks, required=True)
    parser.add_argument('--dataset',        choices=datasets, required=True)
    parser.add_argument('--batchsize',      default=64, type=int)
    parser.add_argument('--bitwidth',       type=int, default=8, help='Quantization bitwidth')
    # parser.add_argument('--dp',             action='store_true',                help='use data parallel')
    # parser.add_argument('--dimhead',        default='512', type=int,            help='(for ViTs only)')
    # parser.add_argument('--convkernel',     default='8', type=int,              help='(for convmixers only)')

    args = parser.parse_args()

    #### Settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # use_amp = not args.noamp
    # aug = args.noaug
    aug = False

    # print('==> Preparing data..')
    # if args.model=='vit_timm':
    #     size = 384
    # else:
    ##     size = args.size
    #     size = 32


    #### Model Factory
    print('==> Building model..')
    model, testloader = models_factory(args.dataset, args.model, args.batchsize, aug)

    # Generate Converter
    quantizer = converter(args.bitwidth)

    # Quantize network
    model = quantizer.quantize(model)
    print(f'Quantized model: {model}')

    #### Testing
    print(f'-I({__file__}): Evaluating Testing Accuracy...')
    start  = datetime.now()
    test(testloader, model, device=device)
    stop   = datetime.now()
    print(f'-I({__file__}): elapsed: {(stop-start).total_seconds()} s')
