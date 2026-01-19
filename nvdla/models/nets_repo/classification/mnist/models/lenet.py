
# Torch
import torch
from torch import nn

##############################
class LeNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super().__init__()
        # MNIST input
        self.conv1   = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2   = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=True)
        self.linear1 = nn.Linear(in_features=400, out_features=120, bias=True)
        self.linear2 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.linear3 = nn.Linear(in_features=84, out_features=n_classes, bias=True)
        self.relu1   = nn.ReLU()
        self.relu2   = nn.ReLU()
        self.relu3   = nn.ReLU()
        self.relu4   = nn.ReLU()
        self.pool    = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.linear3(x)
        return x

