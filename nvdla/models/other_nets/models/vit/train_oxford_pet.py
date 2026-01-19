import torch
import torchvision
import numpy as np
from vision_models_pytorch.models.vit.train import train_vit

from vision_models_pytorch.utils.augmentation import TransformableSubset

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    train_transforms = torchvision.transforms.Compose(
        [
            weights.transforms(),
            # torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # torchvision.transforms.RandomGrayscale(),
            torchvision.transforms.RandomHorizontalFlip(),
        ]
    )

    data = torchvision.datasets.OxfordIIITPet(
        "downloaded_datasets/oxford_iiit_pet", download=True
    )

    train_size = int(np.round(0.7 * len(data)).item())
    valid_size = int(np.round(0.3 * len(data)).item())

    gen = torch.Generator().manual_seed(42)

    train_dataset, valid_dataset = torch.utils.data.random_split(
        data, [train_size, valid_size], generator=gen
    )

    train_dataset = TransformableSubset(train_dataset, data_transform=train_transforms)
    valid_dataset = TransformableSubset(
        valid_dataset, data_transform=weights.transforms()
    )

    train_loader = torch.data.utils.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True
    )
    valid_loader = torch.data.utils.DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )

    train_vit(
        train_loader,
        valid_loader,
        len(data.classes),
        epochs=50,
        learning_rate=1e-3,
        lr_exp_decay=0.95,
        weight_decay=1e-4,
        device=device,
        weight_best_path="vit_pet_train_best.pth",
        weight_latest_path="vit_pet_train_latest.pth",
    )
