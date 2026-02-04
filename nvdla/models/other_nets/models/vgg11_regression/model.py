import torch
import torch.nn as nn
import torchvision

from torchinfo import summary


def vgg_11_regression():
    weights = torchvision.models.VGG11_Weights.IMAGENET1K_V1

    vgg = torchvision.models.vgg11(weights=weights)
    vgg.classifier[6] = nn.Linear(4096, 1, bias=True)

    return vgg


if __name__ == "__main__":
    model = vgg_11_regression()
    summary(model, input_size=(1, 3, 224, 224), row_settings=("var_names",))
