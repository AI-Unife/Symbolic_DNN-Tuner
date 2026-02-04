from typing import Sequence
import torch
import torch.utils.data
import torch.nn as nn


from pytorchyolo.test import _evaluate


def eval_yolov3(
    model: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    class_names: Sequence[str],
    img_size=416,
    iou_thres=0.5,
    conf_thres=0.1,
    nms_thres=0.5,
    verbose=True,
):
    model.eval()

    _evaluate(
        model,
        test_dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose,
    )
