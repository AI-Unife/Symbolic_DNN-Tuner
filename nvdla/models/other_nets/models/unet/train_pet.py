import torch
import torchvision
import numpy as np
from vision_models_pytorch.datasets.oxford_iiit_pet.transforms import (
    create_image_transform,
    create_target_transform,
)
from vision_models_pytorch.models.unet.model import UNet
from vision_models_pytorch.models.unet.train import train_unet

from vision_models_pytorch.utils.augmentation import TransformableSubset


def main():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    data = torchvision.datasets.OxfordIIITPet(
        "downloaded_datasets/oxford_iiit_pet",
        download=True,
        split="trainval",
        target_types="segmentation",
        target_transform=create_target_transform(output_size=(128, 128)),
    )

    train_size = int(np.round(0.7 * len(data)).item())
    valid_size = int(np.round(0.3 * len(data)).item())

    gen = torch.Generator().manual_seed(42)

    train_dataset, valid_dataset = torch.utils.data.random_split(
        data, [train_size, valid_size], generator=gen
    )

    image_transform = create_image_transform(output_size=(128, 128))

    train_dataset = TransformableSubset(train_dataset, data_transform=image_transform)
    valid_dataset = TransformableSubset(valid_dataset, data_transform=image_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )

    model = UNet(in_channels=3, n_classes=3, padding=True)

    train_unet(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        n_classes=3,
        learning_rate=1e-3,
        weight_decay=0,
    )


if __name__ == "__main__":
    main()
