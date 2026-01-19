import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorchyolo.models import load_model
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS

from vision_models_pytorch.models.yolov3.datasets.aerial_maritime import AerialMaritime
from vision_models_pytorch.models.yolov3.eval import eval_yolov3

image_size = 416
batch_size = 16
n_cpu = 16
model_cfg_path = "config/yolov3/yolov3-aerial.cfg"
weights_path = "weights/yolov3/aerial-maritime/yolov3_aerial_maritime_best.pth"


def main():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    test_dataset = AerialMaritime(
        root_path="downloaded_datasets/aerial_maritime",
        split="valid",
        img_size=image_size,
        transform=DEFAULT_TRANSFORMS,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=n_cpu,
        collate_fn=test_dataset.collate_fn,
    )

    model = load_model(model_cfg_path, weights_path)

    eval_yolov3(model, test_dataloader, test_dataset.get_class_names(), image_size)


if __name__ == "__main__":
    main()
