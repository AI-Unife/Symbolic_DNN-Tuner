import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision

from vision_models_pytorch.datasets.cifar100.transforms import create_image_transform
from vision_models_pytorch.models.googlenet.model import GoogleNet

from ignite.metrics import Accuracy, TopKCategoricalAccuracy

from torchinfo import summary

batch_size = 64
num_cpu = 16

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    weights_path="weights/googlenet/cifar100/GoogLeNet_CIFAR100.pth",
    device=DEFAULT_DEVICE,
):
    model = GoogleNet(num_class=100)

    weights = torch.load(weights_path)

    model.load_state_dict(weights)
    model.to(device)

    summary(model, input_size=(1, 3, 32, 32), row_settings=("var_names",))

    transform = create_image_transform()

    valid_dataset = torchvision.datasets.CIFAR100(
        "downloaded_datasets/cifar100", train=False, transform=transform, download=True
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_cpu,
        pin_memory=True,
    )

    metrics = {
        "top_1_accuracy": Accuracy(),
        "top_5_accuracy": TopKCategoricalAccuracy(),
    }

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Waits for everything to finish running
    torch.cuda.synchronize()

    sample_count = 0
    warmup = 20

    time = []

    for data, label in valid_dataloader:
        data = data.to(device)
        label = label.to(device)
        start.record()
        result = model(data)
        end.record()
        torch.cuda.synchronize()
        time.append(start.elapsed_time(end))

    print(time)


if __name__ == "__main__":
    main()
