import torch
import torchvision
import numpy as np

class TransformableSubset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, data_transform=None, target_transform=None, fused_transform=None
    ):
        self.dataset = dataset
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.fused_transform = fused_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.data_transform:
            image = self.data_transform(image)
        if self.target_transform:
            label = self.target_transform(image)
        if self.fused_transform:
            image, label = self.fused_transform(image, label)
        return image, label


def create_target_transform(output_size=(128, 128)):
    def target_transform(target):
        tensor = np.asarray(target)
        tensor = np.copy(tensor)
        tensor = torch.from_numpy(tensor)
        tensor = tensor.unsqueeze(0)
        tensor = torchvision.transforms.functional.resize(
            tensor,
            output_size,
            torchvision.transforms.InterpolationMode.NEAREST,
            antialias=None,
        )
        tensor = tensor[0].long() - 1
        return tensor

    return target_transform


def create_image_transform(output_size=(128, 128)):
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(
                output_size,
                torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=None,
            ),
            # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return image_transform


def get_oxfordpet(datadir: str, batch_size: int, num_workers: int):
    data = torchvision.datasets.OxfordIIITPet(
        datadir,
        download=True,
        split="test",
        target_types="segmentation",
        target_transform=create_target_transform(),
    )
    test_dataset = TransformableSubset(data, data_transform=create_image_transform())
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    return testloader