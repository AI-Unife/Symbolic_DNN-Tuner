import torchvision
import torch.nn as nn

from torchinfo import summary

def get_vgg16_model(n_classes, pretrained_weights=True, return_transforms=True):
    weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    if pretrained_weights:
        model = torchvision.models.vgg16(weights=weights)
    else:
        model = torchvision.models.vgg16()


    model.classifier[6] = nn.Linear(4096, n_classes, bias=True)
    for param in model.classifier[6].parameters():
        param.requires_grad = True

    model.eval()

    if return_transforms:
        return model, weights.transforms()
    else:
        return model
