import torchvision
import torch.nn as nn


def get_vit_model(n_classes, pretrained_weights=True, return_transforms=True):
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    if pretrained_weights:
        model = torchvision.models.vit_b_16(weights=weights)
    else:
        model = torchvision.models.vit_b_16()

    model.heads.head = nn.Linear(768, n_classes, bias=True)
    for param in model.heads.head.parameters():
        param.requires_grad = True

    model.eval()

    if return_transforms:
        return model, weights.transforms()
    else:
        return model
