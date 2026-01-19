import torch
import torch.nn as nn
import torch.utils.data

from vision_models_pytorch.datasets.car_steering_angle.dataset import CarSteeringAngle
from vision_models_pytorch.datasets.utk_face.transforms import (
    data_transform,
    train_transform,
)
from vision_models_pytorch.models.resnet18_regression.model import resnet18_regression
from vision_models_pytorch.models.resnet18_regression.train import train_resnet18_regression

from torchinfo import summary



def main():
    model = resnet18_regression()

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

    train_resnet18_regression(
        model,
        train_dataloader,
        valid_dataloader,
        #resume_checkpoint_path="weights/vgg11_regression/driving/vgg_11_regression_steering_checkpoint_50_mae=1.4743.pt",
        resume_checkpoint_path='checkpoints/resnet18_regression/resnet18_steering_checkpoint_50_mae=1.2878.pt',
        checkpoint_dir="checkpoints/resnet18_regression",
        checkpoint_prefix="resnet18_steering",
    )


if __name__ == "__main__":
    main()
