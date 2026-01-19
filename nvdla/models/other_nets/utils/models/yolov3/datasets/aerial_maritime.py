from typing import Literal, Tuple
import warnings
import torch
import torch.utils.data
from PIL import Image


import numpy as np

import os


from vision_models_pytorch.datasets.aerial_maritime.dataset_utils import (
    get_classes,
    parse_annotation_file,
)
from vision_models_pytorch.models.yolov3.datasets.transforms import resize


class AerialMaritime(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path: str,
        split: Literal["test", "valid", "train"] = "valid",
        img_size=416,
        transform=None,
    ):
        self.root_path = root_path
        self.split_path = os.path.join(root_path, split)
        self.annotation_path = os.path.join(self.split_path, "_annotations.txt")
        self.classes_path = os.path.join(self.split_path, "_classes.txt")

        image_names = [
            file_name
            for file_name in os.listdir(self.split_path)
            if not file_name.startswith("_")
        ]

        self.image_paths = [
            os.path.join(self.split_path, file_name) for file_name in image_names
        ]

        labels = parse_annotation_file(self.split_path)

        self.labels = [labels[name] for name in image_names]

        self.img_size = img_size
        self.max_objects = 100
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        bbs, classes = self.labels[index]

        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        h, w, c = img.shape

        bbs = np.array(bbs, dtype=np.float32)  # (n_boxes, 4)
        classes = np.expand_dims(
            np.array(classes, dtype=np.float32), axis=-1
        )  # (n_boxes, 1)

        # add classes at the first position of the array
        # for each bbox axis 1 is
        # 0 -> class
        # 1,2 -> absolute top left x, y
        # 3,4 -> absolute bottom right x, y
        # Convert it to
        # 0 -> class
        # 1,2 -> relative center x, y
        # 3,4 -> relative box h, w

        boxes = np.concatenate([classes, bbs], axis=1)
        new_boxes = np.zeros_like(boxes)

        new_boxes[:, [1, 2]] = (boxes[:, [3, 4]] + boxes[:, [1, 2]]) / 2
        new_boxes[:, [3, 4]] = boxes[:, [3, 4]] - boxes[:, [1, 2]]
        new_boxes[:, [1, 3]] /= np.float32(w)
        new_boxes[:, [2, 4]] /= np.float32(h)
        new_boxes[:, 0] = boxes[:, 0]

        # Apply Transform
        if self.transform:
            img, bb_targets = self.transform((img, new_boxes))
        img = resize(img, self.img_size)
        return img_path, img, bb_targets

    def get_class_names(self) -> Tuple[str]:
        return get_classes()

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        imgs = torch.stack(imgs)

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.image_paths)
