import torch
import torch.nn as nn
import torchvision

from torchinfo import summary


def resnet18_regression():
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1

    resnet = torchvision.models.resnet18()

    resnet.fc = nn.Linear(512, 1, bias=True)

    return resnet


if __name__ == "__main__":
    model = resnet18_regression()
    summary(model, input_size=(1, 3, 224, 224), row_settings=("var_names",))
