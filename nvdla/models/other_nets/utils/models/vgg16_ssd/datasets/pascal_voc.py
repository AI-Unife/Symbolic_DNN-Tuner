import torch
import torch.utils.data

import os
import json

from vision_models_pytorch.models.vgg16_ssd.datasets.collate import collate_fn
from vision_models_pytorch.models.vgg16_ssd.datasets.transforms import transform
from PIL import Image


class PascalVOCDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False, transform=transform):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        self.transform = transform

        assert self.split in {"TRAIN", "TEST"}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + "_images.json"), "r") as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + "_objects.json"), "r") as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode="r")
        image = image.convert("RGB")

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects["boxes"])  # (n_objects, 4)
        labels = torch.LongTensor(objects["labels"])  # (n_objects)
        difficulties = torch.ByteTensor(objects["difficulties"])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        if self.transform:
            image, boxes, labels, difficulties = self.transform(
                image, boxes, labels, difficulties, split=self.split
            )

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def get_labels(self):
        return (
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        return collate_fn(batch)
