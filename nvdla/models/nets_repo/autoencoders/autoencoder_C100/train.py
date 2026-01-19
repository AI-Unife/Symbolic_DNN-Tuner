import os
import argparse

import torch
from torch import optim
import torch.nn as nn

from cifar100 import getCIFAR
from autoencoder import AutoEncoder
import utils

###################################################

datasetdir = os.environ['TORCH_DATASETDIR']
traindir   = os.environ['TORCH_TRAINDIR']

###################################################

def train(model, savefile, epochs, dataloader):

    ## Prepare Model for Training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model = model.to(device)

    ## Train Settings
    learning_rate = 1e-4
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [epochs//4, epochs//2, epochs//1.25], gamma = 0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3,5,7], gamma = 0.1)

    ## Train Model
    print(f'-I({__file__}): Training model...')
    utils.train_model(model, dataloader, epochs, loss_fn, optimizer, scheduler, device)
    print(f'-I({__file__}): Network trained')

    ## Save Trained Network
    model = model.to('cpu')
    torch.save(model.state_dict(), savefile)
    print(f'-I({__file__}): Weights saved into {savefile}')

###################################################

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch Size')

    args = parser.parse_args()

    ## Get Network
    print(f'-I({__file__}): Loading model...')
    autoencoder = AutoEncoder().to(device)
    savefile = os.path.join(traindir, 'fp32_autoencoder_cifar100.pth')
    print(f'-I({__file__}): Model loaded')
    print(args.batchsize)

    ## Dataset
    trainloader, _ = getCIFAR(datasetdir, 32, batchsize= int(args.batchsize), device=device)

    train(autoencoder, savefile, args.epochs, trainloader)