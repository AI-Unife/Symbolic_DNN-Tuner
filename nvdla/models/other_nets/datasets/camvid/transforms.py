import torchvision
import torchvision.transforms.functional as TVF

import random


def create_target_transform(output_size=(128, 128)):
    def target_transform(target):
        tensor = target.unsqueeze(0)
        tensor = TVF.resize(
            tensor,
            output_size,
            torchvision.transforms.InterpolationMode.NEAREST,
            antialias=None,
        )
        tensor = tensor[0]
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


def create_joint_transform(p_flip=0.5):
    def joint_transform(image, target):
        random_flip = random.random() < p_flip
        if random_flip:
            image = TVF.hflip(image)
            target = TVF.hflip(target)
        return image, target

    return joint_transform
