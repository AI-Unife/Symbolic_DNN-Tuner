from typing import Callable
import torchvision.transforms as TVT


def train_transform() -> Callable:
    return TVT.Compose(
        [
            TVT.Resize(256),
            TVT.CenterCrop(224),
            TVT.AutoAugment(interpolation=TVT.InterpolationMode.BILINEAR),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def data_transform() -> Callable:
    return TVT.Compose(
        [
            TVT.Resize(256),
            TVT.CenterCrop(224),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
