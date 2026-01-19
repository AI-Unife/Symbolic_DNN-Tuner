import torch
import torchvision
import numpy as np

from torch.utils.data import DataLoader

from vision_models_pytorch.datasets.gtsrb.transforms import data_transform, train_transform
from vision_models_pytorch.models.mobilenetv2.train import train_mobilenetv2
from vision_models_pytorch.models.vit.train import train_vit

from vision_models_pytorch.utils.augmentation import TransformableSubset

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    data = torchvision.datasets.GTSRB(
        "downloaded_datasets/gtsrb", split="train", download=True
    )

    n_classes = 43

    train_size = int(np.round(0.7 * len(data)).item())
    valid_size = int(np.round(0.3 * len(data)).item())

    gen = torch.Generator().manual_seed(42)

    train_dataset, valid_dataset = torch.utils.data.random_split(
        data, [train_size, valid_size], generator=gen
    )


    train_dataset = TransformableSubset(train_dataset, data_transform=train_transform())
    valid_dataset = TransformableSubset(
        valid_dataset, data_transform=data_transform()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=True
    )

    train_mobilenetv2(
        train_loader,
        valid_loader,
        n_classes,
        epochs=50,
        learning_rate=2e-3,
        lr_exp_decay=0.95,
        weight_decay=1e-4,
        device=device,
        weight_best_path="checkpoints/mobilenetv2_gtsrb_best.pth",
        weight_latest_path="checkpoints/mobilenetv2_gtsrb_latest.pth",
    )


if __name__ == '__main__':
    main()