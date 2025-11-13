from typing import Literal, Tuple
import warnings
import torch
import torch.utils.data
from PIL import Image


import numpy as np

import os

from tqdm import tqdm

import vision_models_pytorch.datasets.kitti.kitti_utils as kitti_utils
from vision_models_pytorch.models.yolov3.datasets.transforms import resize


class Kitti(torch.utils.data.Dataset[Tuple[str, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root_path="downloaded_datasets/kitti/training",
        split="train",
        split_file=None,
        img_size=416,
        transform=None,
        load_in_ram=False,
    ) -> None:
        self.split = split.lower()
        self.transform = transform
        self.root_path = root_path
        self.img_size = img_size
        self.load_in_ram = load_in_ram

        if split is not None and split_file is None:
            split_file = os.path.join(root_path, f"{split}.txt")

        img_paths, labels_path = kitti_utils.get_split_files(root_path, split_file)
        self.image_paths = img_paths
        self.ann_paths = labels_path

        self.bboxes = []
        self.classes = []
        for ann_path in self.ann_paths:
            bboxes, classes = kitti_utils.parse_label_file(ann_path)
            self.bboxes.append(bboxes)
            self.classes.append(classes)

        if self.load_in_ram:
            print("Loading data in RAM")
            self.preloaded_data = []
            for i in tqdm(range(len(self.image_paths))):
                self.preloaded_data.append(self.load_image(i))
            print("Loaded in RAM")
        self.batch_count = 0

    def load_image(self, i):
        image_path = self.image_paths[i]
        bbs = self.bboxes[i]
        classes = self.classes[i]
        # Read image
        image = Image.open(image_path, mode="r")
        img = np.array(image.convert("RGB"), np.uint8)
        h, w, c = img.shape

        bbs = np.array(bbs, dtype=np.float32)  # (n_boxes, 4)
        classes = np.expand_dims(
            np.array(classes, dtype=np.float32), axis=-1
        )  # (n_boxes, 1)

        boxes = np.concatenate([classes, bbs], axis=1)
        new_boxes = np.zeros_like(boxes)

        new_boxes[:, [1, 2]] = (boxes[:, [3, 4]] + boxes[:, [1, 2]]) / 2
        new_boxes[:, [3, 4]] = boxes[:, [3, 4]] - boxes[:, [1, 2]]
        new_boxes[:, [1, 3]] /= np.float32(w)
        new_boxes[:, [2, 4]] /= np.float32(h)
        new_boxes[:, 0] = boxes[:, 0] - 1

        # Apply Transform
        if self.transform:
            img, bb_targets = self.transform((img, new_boxes))
        img = resize(img, self.img_size)
        return image_path, img, bb_targets

    def __getitem__(self, i):
        if self.load_in_ram:
            image_path, img, bb_targets = self.preloaded_data[i]
        else:
            image_path, img, bb_targets = self.load_image(i)
        return image_path, img, bb_targets

    def __len__(self):
        return len(self.image_paths)

    def get_class_names(self):
        return tuple(kitti_utils.KITTI_CLASSES)

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
