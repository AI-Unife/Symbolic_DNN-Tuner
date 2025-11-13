import torchvision

from torch.utils.data import DataLoader, Dataset


class TransformableSubset(Dataset):
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


def data_transform(size=(128,128)):
    data_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size)
        ]
    )
    return data_transforms


def getGTSRB(datadir, batch_size = 64, num_workers = 8):
    dataset = torchvision.datasets.GTSRB(
        datadir, split="test", download=True
    )

    dataset = TransformableSubset(
        dataset, data_transform=data_transform(size=(32,32))
    )
    
    testloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
    )

    return testloader