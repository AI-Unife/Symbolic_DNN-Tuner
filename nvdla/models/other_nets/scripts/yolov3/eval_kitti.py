import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.utils.data import DataLoader

from pytorchyolo.models import load_model
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS

from vision_models_pytorch.models.yolov3.datasets.kitti import Kitti
from vision_models_pytorch.models.yolov3.eval import eval_yolov3

image_size = 416
batch_size = 16
n_cpu = 16
model_cfg_path = "config/yolov3/yolov3-kitti.cfg"
weights_path = (
    "/home/miele/vision-models-pytorch/weights/yolov3/kitti/yolov3_ckpt_mAP_0.707.pth"
)


def main():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    test_dataset = Kitti(
        split="valid",
        img_size=image_size,
        transform=DEFAULT_TRANSFORMS,
        load_in_ram=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=n_cpu,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
    )

    model = load_model(model_cfg_path, weights_path)

    eval_yolov3(model, test_dataloader, test_dataset.get_class_names(), image_size)


if __name__ == "__main__":
    main()
