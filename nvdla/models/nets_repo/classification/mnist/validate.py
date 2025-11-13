
import os
from datetime import datetime
import argparse
import torch

from dataset import getMNIST
import models
from utils import test

###################################################

datasetdir = os.getenv('TORCH_DATASETPATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'))
traindir   = os.getenv('TORCH_TRAINPATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'))

###################################################

if __name__ == '__main__':

    networks = [
        'lenet',
        'alexnet',
        'vgg11'
    ]

    device   = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',      choices=networks, required=True)
    parser.add_argument('--batchsize',  type=int, default=64,   help='Batch Size')

    args = parser.parse_args()

    print(f'-I({__file__}): Running with device: {device}')

    ## Get Network
    print(f'-I({__file__}): Loading model...')

    if args.model=='lenet':
        model=models.LeNet()
    elif args.model=='alexnet':
        model=models.AlexNet()
    elif args.model=='vgg11':
        model=models.VGG11()
    else:
        raise ValueError('unsupported model')

    savefile = os.path.join(traindir, f'fp32_{args.model}_mnist.pth')
    model.load_state_dict(torch.load(savefile))
    print(f'-I({__file__}): Model loaded')

    ## Dataset
    img_size = (28, 28)
    trainloader, testloader = getMNIST(datasetdir, img_size, args.batchsize)

    print(f'-I({__file__}): Evaluating Training Accuracy...')
    start  = datetime.now()
    test(trainloader, model, device=device)
    stop   = datetime.now()
    print(f'-I({__file__}): elapsed: {(stop-start).total_seconds()} s')
    
    print(f'-I({__file__}): Evaluating Testing Accuracy...')
    start  = datetime.now()
    test(testloader, model, device=device)
    stop   = datetime.now()
    print(f'-I({__file__}): elapsed: {(stop-start).total_seconds()} s')
