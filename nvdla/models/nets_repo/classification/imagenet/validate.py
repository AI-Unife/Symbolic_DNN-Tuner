
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import argparse
import time


from utils import progress_bar
from dataset import imageNET

from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models import resnet50, resnet18, resnet34, resnet101, resnet152
import torch.utils.model_zoo
import torchvision.models as models

#####################################################################################################################
## System

datasetdir = os.getenv('TORCH_DATASETPATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'))
print(datasetdir)
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
                         % (val_loss/(batch_idx+1), (val_correct/val_total)*100, val_correct, val_total))

    acc = (val_correct/val_total)* 100

    return val_loss, acc

#####################################################################################################################
## Networks

if __name__ == '__main__':

    workdir  = os.path.dirname(os.path.realpath(__file__))

    networks = [
        'lenet',
        'alexnet',
        'mobilenetv2',
        'mobilenetv3',
        'googlenet',
        'vgg16',
        'vgg11',
        'vgg19',
        'res18',
        'res34',
        'res50',
        'res101',
        'res152',
        'squeezenet',
        'densenet',
        'resnext50',
        'swin_t',
        'convmixer',
        'mlpmixer',
        'vit_small',
        'vit_tiny',
        'simplevit',
        'vit',
        'cait',
        'cait_small',
       ]

    # Args Parser
    parser = argparse.ArgumentParser(description='PyTorch imagenet Training')

    parser.add_argument('--net',            choices=networks, required=True,    help='Network to train')
    parser.add_argument('--batchsize',      default=64)
    parser.add_argument('--dp',             action='store_true',                help='use data parallel')
    parser.add_argument('--patch',          default='4', type=int,              help='network patch')
    parser.add_argument('--dimhead',        default='512', type=int,            help='(for ViTs only)')
    parser.add_argument('--convkernel',     default='8', type=int,              help='(for convmixers only)')
    parser.add_argument('--size',           default='256', type=int,             help='Input image size')
    parser.add_argument('--noaug',          action='store_true',                help='Disable data augmentation')
    args = parser.parse_args()

    batchsize = int(args.batchsize)
    imsize = int(args.size)

    # use_amp = not args.noamp
    aug = args.noaug

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    if args.net=='vit_timm':
        size = 384
    else:
        size = imsize


    # Model factory..
    print('==> Building model..')

    if args.net=='lenet':
        net = models.LeNet(dropout_value=0.5)
    elif args.net=='alexnet': # 56.5
        weights = AlexNet_Weights.DEFAULT
        net = models.alexnet(weights=weights)
        net.eval()
    elif args.net == 'googlenet': # 69.7
        net = models.googlenet(pretrained=True)
    elif args.net=='mobilenetv3': # 74
        net = models.mobilenet_v3_large(pretrained=True)
    elif args.net=='mobilenetv2': # 71.8
        net = models.mobilenet_v2(pretrained=True)
    elif args.net == 'vgg11':
        net = models.vgg11(pretrained=True)
    elif args.net=='vgg16':
        net = models.vgg16(pretrained = True)
    elif args.net=='vgg19':
        net = models.vgg19(pretrained=True)
    elif args.net=='res18': # 69.7
        weights = ResNet18_Weights.DEFAULT
        net = resnet18(weights=weights)
        net.eval()
    elif args.net=='res34': # 73.2
        weights = ResNet34_Weights.DEFAULT
        net = resnet34(weights=weights)
        net.eval()
    elif args.net=='res50': # 80.3
        weights = ResNet50_Weights.DEFAULT
        net = resnet50(weights=weights)
        net.eval()
    elif args.net=='res101':# 81.6
        weights = ResNet101_Weights.DEFAULT
        net = resnet101(weights=weights)
        net.eval()
    elif args.net=='res152': # 82.3
        weights = ResNet152_Weights.DEFAULT
        net = resnet152(weights=weights)
        net.eval()
    # elif args.net=='convmixer':
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        # net = models.ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
    # elif args.net=='mlpmixer':
    #     from models.mlpmixer import MLPMixer
    #     net = MLPMixer(
    #         image_size = 32,
    #         channels = 3,
    #         patch_size = args.patch,
    #         dim = 512,
    #         depth = 6,
    #         num_classes = 1000
    #     )

    # elif args.net=='vit_small':
    #     from models.vit_small import ViT
    #
    #     net = models.ViT_small(
    #         image_size = size,
    #         patch_size = args.patch,
    #         num_classes = 1000,
    #         dim = int(args.dimhead),
    #         depth = 6,
    #         heads = 8,
    #         mlp_dim = 512,
    #         dropout = 0.1,
    #         emb_dropout = 0.1
    #     )
    # elif args.net=='vit_tiny':
    #     # from models.vit_small import ViT
    #     net = models.ViT_small(
    #         image_size = size,
    #         patch_size = args.patch,
    #         num_classes = 1000,
    #         dim = int(args.dimhead),
    #         depth = 4,
    #         heads = 6,
    #         mlp_dim = 256,
    #         dropout = 0.1,
    #         emb_dropout = 0.1
    #     )
    # elif args.net=='simplevit':
    #     # from models.simplevit import SimpleViT
    #     net = models.ViT_simple(
    #         image_size = size,
    #         patch_size = args.patch,
    #         num_classes = 1000,
    #         dim = int(args.dimhead),
    #         depth = 6,
    #         heads = 8,
    #         mlp_dim = 512
    #     )
    # elif args.net=='vit':
    #     net = models.ViT(
    #         image_size = size,
    #         patch_size = args.patch,
    #         num_classes = 1000,
    #         dim = int(args.dimhead),
    #         depth = 6,
    #         heads = 8,
    #         mlp_dim = 512,
    #         dropout = 0.1,
    #         emb_dropout = 0.1
    #     )
    # elif args.net=='vit_timm':
    #     import timm
    #     net = timm.create_model('vit_base_patch16_384', pretrained=True)
    #     net.head = nn.Linear(net.head.in_features, 10)
    # elif args.net=='cait':
    #     # from models.cait import CaiT
    #     net = models.CaiT(
    #         image_size = size,
    #         patch_size = args.patch,
    #         num_classes = 1000,
    #         dim = int(args.dimhead),
    #         depth = 6,   # depth of transformer for patch to patch attention only
    #         cls_depth=2, # depth of cross attention of CLS tokens to patch
    #         heads = 8,
    #         mlp_dim = 512,
    #         dropout = 0.1,
    #         emb_dropout = 0.1,
    #         layer_dropout = 0.05
    #     )
    # elif args.net=='cait_small':
    #     # from models.cait import CaiT
    #     net = models.CaiT(
    #         image_size = size,
    #         patch_size = args.patch,
    #         num_classes = 1000,
    #         dim = int(args.dimhead),
    #         depth = 6,   # depth of transformer for patch to patch attention only
    #         cls_depth=2, # depth of cross attention of CLS tokens to patch
    #         heads = 6,
    #         mlp_dim = 256,
    #         dropout = 0.1,
    #         emb_dropout = 0.1,
    #         layer_dropout = 0.05
    #     )
    elif args.net=='swin_t': # 81
        net = models.swin_t(pretrained=True)
    elif args.net == 'squeezenet': # 58.1
        net = models.squeezenet1_0(pretrained=True)
    elif args.net == 'densenet': # 77.1
        net = models.densenet161(pretrained=True)
    elif args.net == 'resnext50': # 77.6
        net = models.resnext50_32x4d(pretrained = True)
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
    testloader = imageNET(datasetdir, size, args.batchsize)


    #### Load Net
    print('==> Resuming from savefile..')
    # net.load_state_dict(torch.load(os.path.join(savedir, f'fp32_{args.net}_imagenet.pth')))
    net.to(device)

    #### Testing
    # Criterion
    criterion = nn.CrossEntropyLoss()

    net.cuda()

    test(net, testloader)