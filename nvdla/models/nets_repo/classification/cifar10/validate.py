# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import argparse
import time

import models
from utils import progress_bar
from dataset import getCIFAR10

#####################################################################################################################
## System

datasetdir = os.getenv('TORCH_DATASETPATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'))
savedir    = os.getenv('TORCH_TRAINPATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'))

#####################################################################################################################
## Train Routine

# def train(epoch):
def test(net, testloader):

    ## Validation
    net.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (val_loss/(batch_idx+1), 100.*val_correct/val_total, val_correct, val_total))
    
    acc = 100.*val_correct/val_total

    return val_loss, acc

#####################################################################################################################
## Networks

if __name__ == '__main__':

    workdir  = os.path.dirname(os.path.realpath(__file__))

    networks = [
        'lenet',
        'alexnet',
        'vgg11',
        'vgg19',
        'res18',
        'res34',
        'res50',
        'res101',
        'convmixer',
        'mlpmixer',
        'vit_small',
        'vit_tiny',
        'simplevit',
        'vit',
        'cait',
        'cait_small',
        'swin']

    # Args Parser
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--net',            choices=networks, required=True,    help='Network to train')
    parser.add_argument('--batchsize',      default=512)
    parser.add_argument('--dp',             action='store_true',                help='use data parallel')
    parser.add_argument('--patch',          default='4', type=int,              help='network patch')
    parser.add_argument('--dimhead',        default='512', type=int,            help='(for ViTs only)')
    parser.add_argument('--convkernel',     default='8', type=int,              help='(for convmixers only)')

    args = parser.parse_args()

    batchsize = int(args.batchsize)
    aug = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    if args.net=='vit_timm':
        size = 384
    else:
        size = 32


    # Model factory..
    print('==> Building model..')

    if args.net=='lenet':
        net = models.LeNet(dropout_value=0.5)
    elif args.net=='alexnet':
        net = models.AlexNet()
    elif args.net=='vgg11':
        net = models.VGG11CIFAR10(dropout_value=.1)
    elif args.net=='vgg19':
        net = models.VGG('VGG19')
    elif args.net=='res18':
        net = models.ResNet18()
    elif args.net=='res34':
        net = models.ResNet34()
    elif args.net=='res50':
        net = models.ResNet50()
    elif args.net=='res101':
        net = models.ResNet101()
    elif args.net=='convmixer':
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = models.ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
    elif args.net=='mlpmixer':
        from models.mlpmixer import MLPMixer
        net = MLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = args.patch,
        dim = 512,
        depth = 6,
        num_classes = 10
    )
    elif args.net=='vit_small':
        # from models.vit_small import ViT
        net = models.ViT_small(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=='vit_tiny':
        # from models.vit_small import ViT
        net = models.ViT_small(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 4,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=='simplevit':
        # from models.simplevit import SimpleViT
        net = models.ViT_simple(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512
    )
    elif args.net=='vit':
        net = models.ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    # elif args.net=='vit_timm':
    #     import timm
    #     net = timm.create_model('vit_base_patch16_384', pretrained=True)
    #     net.head = nn.Linear(net.head.in_features, 10)
    elif args.net=='cait':
        # from models.cait import CaiT
        net = models.CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
    elif args.net=='cait_small':
        # from models.cait import CaiT
        net = models.CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
    elif args.net=='swin':
        # from models.swin import swin_t
        net = models.swin_t(window_size=args.patch,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1))
    else:
        raise ValueError('unsupported net')

    # For Multi-GPU
    if 'cuda' in device:
        print(device)
        if args.dp:
            print('using data parallel')
            net = torch.nn.DataParallel(net) # make parallel
            cudnn.benchmark = True


    #### Dataset
    trainloader, testloader, classes = getCIFAR10(datasetdir, size, args.batchsize, aug)


    #### Load Net
    print('==> Resuming from savefile..')
    net.load_state_dict(torch.load(os.path.join(savedir, f'fp32_{args.net}_cifar10.pth')))
    net.to(device)

    #### Testing
    # Criterion
    criterion = nn.CrossEntropyLoss()

    net.cuda()

    test(net, testloader)