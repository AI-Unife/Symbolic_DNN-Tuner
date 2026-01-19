import torch
import torch.nn as nn


def make_valid_step(
    model: nn.Module,
    device,
):
    def _valid_step(engine, batch):
        model.eval()
        with torch.no_grad():
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            predictions = model(image)
            return predictions["out"], label

    return _valid_step
