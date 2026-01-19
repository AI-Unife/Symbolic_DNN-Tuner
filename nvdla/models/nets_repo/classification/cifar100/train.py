#-*- coding: utf-8 -*-
'''

Train CIFAR100 with PyTorch and Vision Transformers!
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
import models.simplevit
from utils import progress_bar
from dataset import getCIFAR100

#####################################################################################################################
## System

datasetdir = os.getenv('TORCH_DATASETPATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'))
savedir    = os.getenv('TORCH_TRAINPATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'))
# savedir    = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

#####################################################################################################################
## Train Routine

# def train(epoch):
def train(net, epoch, workdir, netname, patch, best_acc):
    print('\nEpoch: %d' % epoch)

    ## Train Step
    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)

        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*train_correct/train_total, train_correct, train_total))

    train_loss /= (batch_idx+1)

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

    # Save checkpoint.
    if acc > best_acc:
        print('Saving checkpoint..')
        state = {'model': net.state_dict(),
            'acc': acc,
            'epoch': epoch}

        checkpointsdir = os.path.join(workdir, 'checkpoint')
        if not os.path.isdir(checkpointsdir):
            os.mkdir(checkpointsdir)
        
        torch.save(state, os.path.join(checkpointsdir, netname + f'-{patch}-ckpt.t7'))
        best_acc = acc
    
    ## Write Logs
    logdir = os.path.join(workdir, 'log')
    os.makedirs(logdir, exist_ok=True)

    # .txt Log
    if epoch == 0:
        with open(os.path.join(logdir, f'log_{args.net}_patch{args.patch}.txt'), 'w') as f:
            pass

    content = f'{time.ctime()} , Epoch: {epoch}, lr: {optimizer.param_groups[0]['lr']:.7f}, val loss: {val_loss:.5f}, acc: {(acc):.5f}'
    print(content)

    with open(os.path.join(logdir, 'log_' + args.net + '_patch' + str(patch) + '.txt'), 'a') as appender:
        appender.write(content + '\n')

    # .csv Log
    if epoch == 0:
        with open(os.path.join(logdir, f'log_{args.net}_patch{args.patch}.csv'), 'w') as f:
            pass

    with open(os.path.join(logdir, f'log_{args.net}_patch{args.patch}.csv'), 'a') as f:
        print(f'{train_loss}, {val_loss}, {acc}', file=f)

    return train_loss, val_loss, acc

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

    optimizers = [
        'adam',
        'sgd'
    ]

    # Args Parser
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')

    parser.add_argument('--net',            choices=networks, required=True,    help='Network to train')
    parser.add_argument('--batchsize',      default=512)
    parser.add_argument('--n_epochs',       type=int, default='200',            help='training epochs (400 for ViTs, 200 otherwise)')
    parser.add_argument('--lr',             default=1e-3, type=float,           help='learning rate (recommended: 1e-4 for ViTs, 1e-3 otherwise)') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--opt',            default='adam', choices=optimizers)
    parser.add_argument('--resume', '-r',   action='store_true',                help='resume from checkpoint')
    parser.add_argument('--noaug',          action='store_false',               help='disable use randomaug')
    parser.add_argument('--noamp',          action='store_true',                help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--dp',             action='store_true',                help='use data parallel')
    parser.add_argument('--patch',          default='4', type=int,              help='network patch')
    parser.add_argument('--dimhead',        default='512', type=int,            help='(for ViTs only)')
    parser.add_argument('--convkernel',     default='8', type=int,              help='(for convmixers only)')
    parser.add_argument('--nosave',         action='store_false',               help='do not save the results')

    args = parser.parse_args()

    use_amp = not args.noamp
    aug = args.noaug

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
        net = models.VGG11CIFAR100(dropout_value=.1)
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
        net = models.ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=100)
    elif args.net=='mlpmixer':
        from models.mlpmixer import MLPMixer
        net = MLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = args.patch,
        dim = 512,
        depth = 6,
        num_classes = 100
    )
    elif args.net=='vit_small':
        # from models.vit_small import ViT
        net = models.ViT_small(
        image_size = size,
        patch_size = args.patch,
        num_classes = 100,
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
        num_classes = 100,
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
        num_classes = 100,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512
    )
    elif args.net=='vit':
        net = models.ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 100,
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
        num_classes = 100,
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
        num_classes = 100,
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
                    num_classes=100,
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
    trainloader, testloader = getCIFAR100(datasetdir, size, args.batchsize, aug)


    #### Resume
    if args.resume and os.path.isfile(os.path.join(workdir, 'checkpoint', f'{args.net}-{args.patch}-ckpt.t7')):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(os.path.join(workdir, 'checkpoint', f'{args.net}-{args.patch}-ckpt.t7'))
        net.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    #### Training
    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

    # Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    net.cuda()

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        train_loss, val_loss, acc = train(net, epoch, workdir, args.net, args.patch, best_acc)
        
        scheduler.step(epoch-1) # step cosine scheduling
        
        # print(f'{train_loss}, {val_loss}, {acc}')

    if args.nosave or (acc < best_acc):
        net.to('cpu') # Use always CPU device for storage
        print(f'saving results net...')
        os.makedirs(savedir, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(savedir, f'fp32_{args.net}_cifar100.pth'))
