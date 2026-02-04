import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, in_chans=1, img_size=28, **kwargs):
        super(AlexNet, self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Adapted to 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        # Fuse conv + relu in features
        for i in [0, 3, 6, 8, 10]:
            torch.quantization.fuse_modules(
                self.features, [str(i), str(i + 1)], inplace=True
            )
        # Fuse linear + relu in classifier
        torch.quantization.fuse_modules(self.classifier, ["0", "1"], inplace=True)
        torch.quantization.fuse_modules(self.classifier, ["2", "3"], inplace=True)
