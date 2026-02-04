from typing import Literal
import torch
import torch.utils.data
import torchvision

import os
import csv
import numpy as np

from natsort import natsorted
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt


class CamVid(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        transform=None,
        target_transform=None,
        transforms=None,
        grouped_classes=True,
        mask_mode: Literal["rgb", "ids"] = "ids",
    ):
        self.root = root
        self.split = split
        self.mask_mode = mask_mode
        self.grouped_classes = grouped_classes  # http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        self.images_path = os.path.join(self.root, split)
        self.labels_path = os.path.join(self.root, f"{split}_labels")

        self.all_images = [
            os.path.join(self.images_path, filename)
            for filename in natsorted(os.listdir(self.images_path))
        ]
        self.all_labels = [
            os.path.join(self.labels_path, filename)
            for filename in natsorted(os.listdir(self.labels_path))
        ]

        self.class_dict_path = os.path.join(self.root, "class_dict.csv")
        self.color_to_id = {}
        self.id_to_color = {}

        self.classes = []
        with open(self.class_dict_path) as f:
            reader = csv.reader(f)
            # Skip headers line
            next(reader)
            for class_id, row in enumerate(reader):
                name, r, g, b, group_class_id = row
                name = name.strip(" ")
                r = int(r.strip(" "))
                g = int(g.strip(" "))
                b = int(b.strip(" "))
                group_class_id = int(group_class_id.strip(" "))
                if self.grouped_classes:
                    self.id_to_color[group_class_id] = (r, g, b)
                    self.color_to_id[(r, g, b)] = group_class_id
                else:
                    self.id_to_color[class_id] = (r, g, b)
                    self.color_to_id[(r, g, b)] = class_id
                    self.classes.append(name)
        if self.grouped_classes:
            self.classes = ["Void", "MovingObjects", "Road", "Ceiling", "FixedObjects"]

    def __len__(self):
        return len(self.all_images)

    def get_filenames(self, idx):
        image_path = self.all_images[idx]
        label_path = self.all_labels[idx]
        return image_path, label_path

    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        label_path = self.all_labels[idx]

        out_image = Image.open(image_path).convert("RGB")
        mask = Image.open(label_path).convert("RGB")
        mask = torchvision.transforms.functional.pil_to_tensor(mask)
        if self.transform:
            out_image = self.transform(out_image)
        if self.target_transform:
            mask = self.target_transform(mask)
        if self.mask_mode != "rgb":
            mask = rgb_to_mask(mask, self.color_to_id).long()
            sns.heatmap(mask).get_figure().savefig(os.path.basename(image_path))
            plt.clf()

        if self.transforms:
            out_image, mask = self.transforms(out_image, mask)
        return out_image, mask


def rgb_to_mask(img, color_to_id_map):
    """
    Converts a RGB image mask of shape to Binary Mask of shape [batch_size, classes, h, w]

    Args
    ----
    - img : tensor.Tensor \n
        A RGB img mask. Shape [...,C,H,W]
    - color_map : dict[tuple, int] \n
        Dictionary representing color mappings

    Return
    ----
    - out \n
        A Binary Mask of shape [...,H,W]
    """
    output = torch.zeros(img.size()[:-3] + img.size()[-2:], dtype=torch.long)
    for (r, g, b), class_id in color_to_id_map.items():
        mask = (img == torch.tensor([r, g, b]).unsqueeze(-1).unsqueeze(-1)).all(dim=-3)
        output += class_id * mask

    return output


def mask_to_rgb(mask, id_to_color_map):
    """
    Converts a RGB image mask of shape to Binary Mask of shape [batch_size, classes, h, w]

    Args
    ----
    - mask \n
        A RGB img mask
    - color_map \n
        Dictionary representing color mappings

    Return
    ----
    - out \n
        A Binary Mask of shape [batch_size, classes, h, w]
    """
    output = torch.zeros((3,) + mask.size(), dtype=torch.uint8)
    for class_id, (r, g, b) in id_to_color_map.items():
        print(class_id)
        print((r, g, b))
        apply_mask = mask == class_id
        print(apply_mask)
        output[0] += r * apply_mask
        output[1] += g * apply_mask
        output[2] += b * apply_mask

        print(output[0])

    return output
