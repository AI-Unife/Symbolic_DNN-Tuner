
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MNIST
from nets_repo.classification.mnist.dataset import getMNIST
from nets_repo.classification.mnist.models import LeNet as LeNet_MNIST
from nets_repo.classification.mnist.models import AlexNet as AlexNet_MNIST
from nets_repo.classification.mnist.models import VGG11 as VGG11_MNIST

# CIFAR10
from nets_repo.classification.cifar10.dataset import getCIFAR10
from nets_repo.classification.cifar10.models import LeNet as LeNet_CIFAR10
from nets_repo.classification.cifar10.models import AlexNet as AlexNet_CIFAR10
from nets_repo.classification.cifar10.models import ResNet9 as ResNet9_CIFAR10
from nets_repo.classification.cifar10.models import ResNet18 as ResNet18_CIFAR10
from nets_repo.classification.cifar10.models import ResNet34 as ResNet34_CIFAR10
from nets_repo.classification.cifar10.models import ResNet50 as ResNet50_CIFAR10
from nets_repo.classification.cifar10.models import VGG11 as VGG11_CIFAR10

# CIFAR100
from nets_repo.classification.cifar100.dataset import getCIFAR100
from nets_repo.classification.cifar100.models import AlexNet as AlexNet_CIFAR100
from nets_repo.classification.cifar100.models import ResNet18 as ResNet18_CIFAR100
from nets_repo.classification.cifar100.models import ResNet34 as ResNet34_CIFAR100
from nets_repo.classification.cifar10.models import ResNet50 as ResNet50_CIFAR100

# GTSRB
from other_nets.classification.gtsrb.dataset import getGTSRB
from other_nets.classification.gtsrb.models.mobilenetv2 import get_mobilenetv2_model as MobileNet_GTSRB

# OXFORDPET
from other_nets.segmentation.oxfordpet.dataset import get_oxfordpet as getOXFORDPET
from other_nets.segmentation.oxfordpet.models.deeplabv3 import get_deeplabv3 as DeepLab_OXFORDPET

###################################################

# NETS REPO
datasetdir = os.environ['TORCH_DATASETPATH']
traindir   = os.environ['TORCH_TRAINPATH']

# # OTHER NETS
# cwd          = os.path.dirname(os.path.abspath(__file__))
# gtsrbdir     = os.path.join(cwd, 'other_nets', 'classification', 'gtsrb')
# oxfordpetdir = os.path.join(cwd, 'other_nets', 'segmentation', 'oxfordpet')


###################################################

datasets = [
    'mnist',
    'cifar10',
    'cifar100',
    #'coco',
    'gtsrb',
    'oxfordpet',
]

networks = [
    'lenet',
    'alexnet',
    'vgg11',
    'res9',
    'res18',
    'res34',
    'res50',
    #'yolov11',
    'mobilenetv2',
    'deeplabv3',
    ]

###################################################

def models_factory(dataset, modelname, batchsize, device='cpu', augment=False):

    if dataset=='mnist':
        if modelname=='lenet':
            model = LeNet_MNIST()
        elif modelname=='alexnet':
            model = AlexNet_MNIST()
        elif modelname=='vgg11':
            model = VGG11_MNIST()
        else:
            raise ValueError('unsupported net')

        _, testloader = getMNIST(datasetdir, (28,28), batchsize, device)

    elif dataset=='cifar10':
        if modelname=='lenet':
            model = LeNet_CIFAR10(dropout_value=0.5)
        elif modelname=='alexnet':
            model = AlexNet_CIFAR10()
        elif modelname=='vgg11':
            model = VGG11_CIFAR10(dropout_value=.1)
        # elif modelname=='vgg19':
        #     model = models.VGG('VGG19')
        elif modelname=='res9':
            model = ResNet9_CIFAR10()
        elif modelname=='res18':
            model = ResNet18_CIFAR10()
        elif modelname=='res34':
            model = ResNet34_CIFAR10()
        elif modelname=='res50':
            model = ResNet50_CIFAR10()
        # elif modelname=='res50':
        #     model = ResNet50_CIFAR10()
        # elif modelname=='res101':
        #     model = ResNet101_CIFAR10()
        # elif modelname=='convmixer':
        #     # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        #     model = models.ConvMixer(256, 16, kernel_size=convkernel, patch_size=1, n_classes=10)
        # elif modelname=='mlpmixer':
        #     from models.mlpmixer import MLPMixer
        #     model = MLPMixer(
        #     image_size = 32,
        #     channels = 3,
        #     patch_size = patch,
        #     dim = 512,
        #     depth = 6,
        #     num_classes = 10
        # )
        # elif modelname=='vit_small':
        #     # from models.vit_small import ViT
        #     model = models.ViT_small(
        #     image_size = size,
        #     patch_size = patch,
        #     num_classes = 10,
        #     dim = int(dimhead),
        #     depth = 6,
        #     heads = 8,
        #     mlp_dim = 512,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )
        # elif modelname=='vit_tiny':
        #     # from models.vit_small import ViT
        #     model = models.ViT_small(
        #     image_size = size,
        #     patch_size = patch,
        #     num_classes = 10,
        #     dim = int(dimhead),
        #     depth = 4,
        #     heads = 6,
        #     mlp_dim = 256,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )
        # elif modelname=='simplevit':
        #     # from models.simplevit import SimpleViT
        #     model = models.ViT_simple(
        #     image_size = size,
        #     patch_size = patch,
        #     num_classes = 10,
        #     dim = int(dimhead),
        #     depth = 6,
        #     heads = 8,
        #     mlp_dim = 512
        # )
        # elif modelname=='vit':
        #     model = models.ViT(
        #     image_size = size,
        #     patch_size = patch,
        #     num_classes = 10,
        #     dim = int(dimhead),
        #     depth = 6,
        #     heads = 8,
        #     mlp_dim = 512,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )
        # # elif modelname=='vit_timm':
        # #     import timm
        # #     model = timm.create_model('vit_base_patch16_384', pretrained=True)
        # #     net.head = nn.Linear(net.head.in_features, 10)
        # elif modelname=='cait':
        #     # from models.cait import CaiT
        #     model = models.CaiT(
        #     image_size = size,
        #     patch_size = patch,
        #     num_classes = 10,
        #     dim = int(dimhead),
        #     depth = 6,   # depth of transformer for patch to patch attention only
        #     cls_depth=2, # depth of cross attention of CLS tokens to patch
        #     heads = 8,
        #     mlp_dim = 512,
        #     dropout = 0.1,
        #     emb_dropout = 0.1,
        #     layer_dropout = 0.05
        # )
        # elif modelname=='cait_small':
        #     # from models.cait import CaiT
        #     model = models.CaiT(
        #     image_size = size,
        #     patch_size = patch,
        #     num_classes = 10,
        #     dim = int(dimhead),
        #     depth = 6,   # depth of transformer for patch to patch attention only
        #     cls_depth=2, # depth of cross attention of CLS tokens to patch
        #     heads = 6,
        #     mlp_dim = 256,
        #     dropout = 0.1,
        #     emb_dropout = 0.1,
        #     layer_dropout = 0.05
        # )
        # elif modelname=='swin':
        #     # from models.swin import swin_t
        #     model = models.swin_t(window_size=patch,
        #                 num_classes=10,
        #                 downscaling_factors=(2,2,2,1))
        else:
            raise ValueError('unsupported net')

        ## Get Network
        savefile = os.path.join(traindir, f'fp32_{modelname}_{dataset}.pth')
        print(f'-I({__file__}): Loading model...')
        model.load_state_dict(torch.load(savefile))
        print(f'-I({__file__}): Model loaded')
        print(f'-I({__file__}): printing loaded model:')
        print(model)

        ## Dataset
        _, testloader, _ = getCIFAR10(datasetdir, 32, batchsize, augment)

    elif dataset=='cifar100':
        if modelname=='alexnet':
            model = AlexNet_CIFAR100()
        # elif modelname=='vgg19':
        #     model = models.VGG('VGG19')
        elif modelname=='res18':
            model = ResNet18_CIFAR100()
        elif modelname=='res34':
            model = ResNet34_CIFAR100()
        elif modelname=='res50':
            model = ResNet50_CIFAR100()
        else:
            raise ValueError('unsupported net')

        ## Get Network
        savefile = os.path.join(traindir, f'fp32_{modelname}_{dataset}.pth')
        print(f'-I({__file__}): Loading model...')
        model.load_state_dict(torch.load(savefile))
        print(f'-I({__file__}): Model loaded')
        print(f'-I({__file__}): printing loaded model:')
        print(model)

        ## Dataset
        _, testloader = getCIFAR100(datasetdir, 32, batchsize, augment)

    # FIXME
    # elif dataset == 'coco':
    #     from other_nets.detection.coco.dataset import getCOCO
    #     testloader = getCOCO(datasetdir, batchsize)

    #     if modelname == 'yolov11':
    #         from other_nets.detection.coco.models.yolov11.yolov11 import get_yolov11
    #         print(f'-I({__file__}): Loading model...')
    #         model = get_yolov11(traindir)
    #         print(f'-I({__file__}): Model loaded')
    #         print(f'-I({__file__}): printing loaded model:')
    #         print(model)
    #     else:
    #         raise ValueError('unsupported net')

    elif dataset == 'gtsrb':
        # from other_nets.classification.gtsrb.dataset import getGTSRB
        testloader = getGTSRB(datasetdir, batchsize)
        # testloader = getGTSRB(os.path.join(gtsrbdir, 'gtsrb_data'), batchsize) # TORCH_DATASETPATH=models/other_nets/classification/gtsrb/gtsrb_data

        if modelname == 'mobilenetv2':
            # from other_nets.classification.gtsrb.models.mobilenetv2 import get_mobilenetv2_model
            print(f'-I({__file__}): Loading model...')
            model = MobileNet_GTSRB(43, os.path.join(traindir, 'fp32_mobilenet-v2_gtsrb.pth')) # TORCH_TRAINPATH=models/other_nets/classification/gtsrb/models/mobilenetv2_gtsrb_best.pth 
            # model = get_mobilenetv2_model(43, os.path.join(gtsrbdir, 'models', 'mobilenetv2_gtsrb_best.pth')) # TORCH_TRAINPATH=models/other_nets/classification/gtsrb/models/mobilenetv2_gtsrb_best.pth 
            print(f'-I({__file__}): Model loaded')
            print(f'-I({__file__}): printing loaded model:')
            print(model)
        else:
            raise ValueError('unsupported net')
 
    elif dataset == 'oxfordpet':
        # from other_nets.segmentation.oxfordpet.dataset import get_oxfordpet
        testloader = getOXFORDPET(datasetdir, batchsize, 4) # TORCH_DATASETPATH=models/other_nets/segmentation/oxfordpet/oxfordpet_data
        # testloader = get_oxfordpet(os.path.join(oxfordpetdir, 'oxfordpet_data'), batchsize, 4) # TORCH_DATASETPATH=models/other_nets/segmentation/oxfordpet/oxfordpet_data

        if modelname == 'deeplabv3':
            # from other_nets.segmentation.oxfordpet.models.deeplabv3 import get_deeplabv3
            print(f'-I({__file__}): Loading model...')
            model = DeepLab_OXFORDPET(os.path.join(traindir, 'fp32_deeplab-v3_oxfordpet.pt')) # TORCH_TRAINPATH=models/other_nets/segmentation/oxfordpet/models/deeplabv3_pet_0.7500.pt
            # model = get_deeplabv3(os.path.join(oxfordpetdir, 'models', 'deeplabv3_pet_0.7500.pt')) # TORCH_TRAINPATH=models/other_nets/segmentation/oxfordpet/models/deeplabv3_pet_0.7500.pt
            print(f'-I({__file__}): Model loaded')
            print(f'-I({__file__}): printing loaded model:')
            print(model)
        else:
            raise ValueError('unsupported net')

    else:
        raise ValueError('unsupported dataset')
    
    return model, testloader
