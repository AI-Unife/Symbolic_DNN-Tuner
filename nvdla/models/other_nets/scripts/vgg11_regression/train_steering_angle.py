import torch
import torch.nn as nn
import torch.utils.data

from vision_models_pytorch.datasets.car_steering_angle.dataset import CarSteeringAngle
from vision_models_pytorch.datasets.utk_face.transforms import (
    data_transform,
    train_transform,
)
from vision_models_pytorch.models.vgg11_regression.model import vgg_11_regression
from torchinfo import summary

from vision_models_pytorch.models.vgg11_regression.train import train_vgg11_regression


def main():
    model = vgg_11_regression()

    train_dataset = CarSteeringAngle(
        root_path="downloaded_datasets/car_steering_angle",
        split="train",
        transform=train_transform(),
        seed=42,
    )

    valid_dataset = CarSteeringAngle(
        root_path="downloaded_datasets/car_steering_angle",
        split="valid",
        transform=data_transform(),
        seed=42,
    )

    summary(model, input_size=(1, 3, 224, 224), row_settings=("var_names",))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=8
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=8
    )

    train_vgg11_regression(
        model,
        train_dataloader,
        valid_dataloader,
        resume_checkpoint_path="weights/vgg11_regression/driving/vgg_11_regression_steering_checkpoint_50_mae=1.4743.pt",
        checkpoint_prefix="vgg_11_regression_steering",
    )


if __name__ == "__main__":
    main()
