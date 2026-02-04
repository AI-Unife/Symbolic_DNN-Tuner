import torch
import torchvision
import numpy as np
from vision_models_pytorch.datasets.cityscapes.classes import (
    CITYSCAPE_CLASS_NAMES,
    CITYSCAPE_N_CLASSES,
)
from vision_models_pytorch.datasets.cityscapes.transforms import (
    create_train_transform,
    create_valid_transform,
)
from vision_models_pytorch.models.deeplabv3.train import train

from vision_models_pytorch.utils.augmentation import TransformableSubset


def main():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    valid_transform = create_valid_transform(output_size=(256, 512))
    train_transforms = create_train_transform(output_size=(256, 512))

    train_dataset = torchvision.datasets.Cityscapes(
        "downloaded_datasets/cityscapes",
        mode="fine",
        target_type="semantic",
        split="train",
        transforms=train_transforms,
    )

    valid_dataset = torchvision.datasets.Cityscapes(
        "downloaded_datasets/cityscapes",
        mode="fine",
        target_type="semantic",
        split="val",
        transforms=valid_transform,
    )

    print(f"Number of classes: {CITYSCAPE_N_CLASSES}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=16, pin_memory=True
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True
    )

    train(
        train_dataloader,
        valid_dataloader,
        classes=list(CITYSCAPE_CLASS_NAMES.values()),
        ignore_class=0,
        learning_rate=1e-4,
        weight_decay=0.0005,
        epochs=150,
        checkpoint_name="cityscapes_deeplabv3",
    )


if __name__ == "__main__":
    main()
