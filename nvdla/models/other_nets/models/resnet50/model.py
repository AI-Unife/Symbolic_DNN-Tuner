import torchvision
import torch.nn as nn

from torchinfo import summary

def get_resnet50_model(n_classes, pretrained_weights=True, return_transforms=True):
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    if pretrained_weights:
        model = torchvision.models.resnet50(weights=weights)
    else:
        model = torchvision.models.resnet50()


    model.fc= nn.Linear(2048, n_classes, bias=True)
    for param in model.fc.parameters():
        param.requires_grad = True

    # summary(model, row_settings=('var_names',), input_size=(1,3,224,224), col_names=('input_size', 'output_size'), depth=5)
    model.eval()

    if return_transforms:
        return model, weights.transforms()
    else:
        return model
