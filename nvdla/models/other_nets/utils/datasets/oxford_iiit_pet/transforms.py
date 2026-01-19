import torch
import torchvision

import numpy as np


def create_target_transform(output_size=(128, 128)):
    def target_transform(target):
        tensor = np.asarray(target)
        tensor = np.copy(tensor)
        tensor = torch.from_numpy(tensor)
        tensor = tensor.unsqueeze(0)
        tensor = torchvision.transforms.functional.resize(
            tensor,
            output_size,
            torchvision.transforms.InterpolationMode.NEAREST,
            antialias=None,
        )
        tensor = tensor[0].long() - 1
        return tensor

    return target_transform


def create_image_transform(output_size=(128, 128)):
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(
                output_size,
                torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=None,
            ),
            # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return image_transform
