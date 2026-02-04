import torch
from torch.utils.data import DataLoader

from vision_models_pytorch.models.yolov3.datasets.aerial_maritime import AerialMaritime

from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS

from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS

from vision_models_pytorch.models.yolov3.datasets.kitti import Kitti
from vision_models_pytorch.models.yolov3.train import train_yolo

from pytorchyolo.models import load_model

batch_size = 16
n_cpu = 16
image_size = 416


def main():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    train_dataset = Kitti(
        split="train",
        img_size=image_size,
        transform=AUGMENTATION_TRANSFORMS,
        load_in_ram=False,
    )
    valid_dataset = Kitti(
        split="valid",
        img_size=image_size,
        transform=DEFAULT_TRANSFORMS,
        load_in_ram=False,
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

    model_cfg_path = "config/yolov3/yolov3-kitti.cfg"

    model = load_model(model_cfg_path, None)
    print(model.hyperparams)

    train_yolo(
        model_cfg_path,
        train_dataloader,
        valid_dataloader,
        "logs/yolov3_kitti",
        train_dataset.get_class_names(),
        checkpoint_path="checkpoints_kitti/yolov3_ckpt_{epoch}.pth",
        epochs=2000,
        pretrained_weights_path="weights/yolov3/gmdarknet53.conv.74",
        verbose=True,
        evaluation_interval=2,
        checkpoint_interval=2,
    )


if __name__ == "__main__":
    main()
