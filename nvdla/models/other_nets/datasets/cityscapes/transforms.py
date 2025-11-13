import torch
import torchvision
import torchvision.transforms.functional as TVF

import numpy as np

from vision_models_pytorch.datasets.cityscapes.classes import CITYSCAPE_ID_TO_TRAIN


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
        tensor = tensor[0].long()
        new_tensor = torch.zeros_like(tensor)
        for orig_id, dest_id in CITYSCAPE_ID_TO_TRAIN.items():
            if dest_id > 0:
                new_tensor += (tensor == orig_id) * dest_id

        return new_tensor

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


def create_valid_transform(output_size=(128, 128)):
    target_f = create_target_transform(output_size)
    image_f = create_image_transform(output_size)

    def train_transform(image, target):
        target = target_f(target)
        image = image_f(image)
        return image, target

    return train_transform


def create_train_transform(output_size=(128, 128)):
    target_f = create_target_transform(output_size)
    image_f = create_image_transform(output_size)
    hflip = create_random_flip(0.5)

    def train_transform(image, target):
        target = target_f(target)
        image = image_f(image)
        return hflip(image, target)

    return train_transform


def create_random_flip(p):
    def random_flip(image, target):
        if np.random.uniform() < p:
            image = TVF.hflip(image)
            target = TVF.hflip(target)
        return image, target

    return random_flip
