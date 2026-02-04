import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import prune


from pytorchyolo.models import load_model
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS

from vision_models_pytorch.models.yolov3.datasets.kitti import Kitti
from vision_models_pytorch.models.yolov3.eval import eval_yolov3

from torchinfo import summary

image_size = 416
batch_size = 16
n_cpu = 16
model_cfg_path = "config/yolov3/yolov3-kitti.cfg"
weights_path = (
    "/home/miele/vision-models-pytorch/checkpoints/yolov3/kitti/yolov3_ckpt_272.pth"
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
    summary(
        model, input_size=(1, 3, image_size, image_size), row_settings=("var_names",)
    )
    for i in range(12, 100):
        try:
            prune.ln_structured(
                model.module_list[i].get_submodule(f"conv_{i}"),
                "weight",
                amount=0.5,
                dim=0,
                n=float("-inf"),
            )
            print(f"prugning {i}")
        except AttributeError:
            pass

    module = model.module_list[12].get_submodule(f"conv_{12}")
    for hook in module._forward_pre_hooks.values():
        if hook._tensor_name == "weight":  # select out the correct hook
            break

    print(hook)  # pruning history in the container

    eval_yolov3(model, test_dataloader, test_dataset.get_class_names(), image_size)


if __name__ == "__main__":
    main()
