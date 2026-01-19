import os
from typing import Literal
import torch
import torch.utils.data
from PIL import Image

from vision_models_pytorch.models.vgg16_ssd.datasets.collate import collate_fn
from vision_models_pytorch.models.vgg16_ssd.datasets.transforms import transform

from vision_models_pytorch.datasets.aerial_maritime.dataset_utils import (
    get_classes,
    parse_annotation_file,
)


class AerialMaritime(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path,
        split: Literal["train", "valid", "test"] = "valid",
        transform=transform,
    ):
        self.split = split.lower()
        self.transform = transform
        assert self.split in {"test", "train", "valid"}
        self.root_path = root_path
        self.split_path = os.path.join(root_path, self.split)

        self.split_data = list(
            parse_annotation_file(self.split_path, class_offset=1).items()
        )

    def __getitem__(self, i):
        image_name, (bbs, classes) = self.split_data[i]

        image_path = os.path.join(self.root_path, self.split, image_name)

        # Read image
        image = Image.open(image_path, mode="r")
        image = image.convert("RGB")

        # print(f'{bbs} {classes}')
        boxes = torch.FloatTensor(bbs)  # (n_objects, 4)
        labels = torch.LongTensor(classes)  # (n_objects)
        difficulties = torch.zeros_like(labels).byte()  # (n_objects)

        # Apply transformations
        if self.transform is not None:
            image, boxes, labels, difficulties = transform(
                image, boxes, labels, difficulties, split=self.split.upper()
            )

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.split_data)

    def get_labels(self):
        return get_classes()

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        return collate_fn(batch)
