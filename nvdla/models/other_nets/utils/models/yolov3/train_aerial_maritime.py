import torch
from torch.utils.data import DataLoader

from vision_models_pytorch.models.yolov3.datasets.aerial_maritime import AerialMaritime

from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS

from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS

from vision_models_pytorch.models.yolov3.train import train_yolo

from pytorchyolo.models import load_model


batch_size = 32
img_size = 416
n_cpu = 8


def main():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    train_dataset = AerialMaritime(
        "downloaded_datasets/aerial_maritime",
        split="train",
        img_size=img_size,
        transform=AUGMENTATION_TRANSFORMS,
    )
    valid_dataset = AerialMaritime(
        "downloaded_datasets/aerial_maritime",
        split="valid",
        img_size=img_size,
        transform=DEFAULT_TRANSFORMS,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=valid_dataset.collate_fn,
    )

    model_cfg_path = "config/yolov3/yolov3-tiny-aerial.cfg"

    model = load_model(model_cfg_path, None)
    print(model.hyperparams)

    train_yolo(
        model_cfg_path,
        train_dataloader,
        valid_dataloader,
        "logs/yolov3_tiny_aerial",
        train_dataset.get_class_names(),
        epochs=1000,
        pretrained_weights_path="weights/yolov3/darknet53.conv.74",
        checkpoint_path="checkpoints_boats_tiny/yolov3_ckpt_{epoch}.pth",
        verbose=True,
        evaluation_interval=2,
        checkpoint_interval=2,
    )


if __name__ == "__main__":
    main()
