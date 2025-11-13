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

from vision_models_pytorch.models.unet.model import UNet
from vision_models_pytorch.models.unet.train import train_unet


batch_size = 16


def main():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    valid_transform = create_valid_transform(output_size=(128, 256))
    train_transforms = create_train_transform(output_size=(128, 256))

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
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    model = UNet(
        in_channels=3, n_classes=CITYSCAPE_N_CLASSES, padding=True, batch_norm=True
    )

    print(CITYSCAPE_N_CLASSES)
    print(list(CITYSCAPE_CLASS_NAMES.values()))

    train_unet(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        n_classes=CITYSCAPE_N_CLASSES,
        class_names=list(CITYSCAPE_CLASS_NAMES.values()),
        ignore_class=0,
        learning_rate=1e-4,
        weight_decay=0.005,
        checkpoint_name="unet_cityscapes",
    )


if __name__ == "__main__":
    main()
