import os
import torch
import torch.utils.data
from PIL import Image

import vision_models_pytorch.datasets.kitti.kitti_utils as kitti_utils
from vision_models_pytorch.models.vgg16_ssd.datasets.collate import collate_fn
from vision_models_pytorch.models.vgg16_ssd.datasets.transforms import transform


class Kitti(torch.utils.data.Dataset):
    """
    Kitti dataset loader for VGG16 SSD Model.
    Only "train" and "valid" split come with annotations.
    """

    def __init__(
        self,
        data_folder="downloaded_datasets/kitti/training",
        split="train",
        split_file=None,
        transform=transform,
    ) -> None:
        """
        Kitti dataset for VGG16 SSD object detection models

        Args
        ---
        * ``data_folder : str``. Path to the folder that contains the images and the split data.
            The data folder contains:
                * `image_2` a folder containing all the png images of the left camera.
                * `label_2` a folder containing all the labels. The format is the same described by the authors of Kitti benchmark
                * `valid.txt` and `train.txt`. Two text files containing the numbers of the images that are in each split (one per line).
        * ``split : "train" | "valid"``. The split to load, only "train" and "valid" are supported.
        * ``split_file : Optional[str]``. Path to an optional custom split file with the same format of ``valid.txt``. If not specified the default ones will be used.
        * ``transform : Optional[Callable]``. A custom transform to use. If None and split is "train", the augmentation will be applied.
        """

        self.split = split.lower()
        self.transform = transform
        self.data_folder = data_folder

        if split is not None and split_file is None:
            split_file = os.path.join(data_folder, f"{split}.txt")

        img_paths, labels_path = kitti_utils.get_split_files(data_folder, split_file)
        self.image_paths = img_paths
        self.ann_paths = labels_path

        self.bboxes = []
        self.classes = []
        for ann_path in self.ann_paths:
            bboxes, classes = kitti_utils.parse_label_file(ann_path)
            self.bboxes.append(bboxes)
            self.classes.append(classes)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        bbs = self.bboxes[i]
        classes = self.classes[i]

        # Read image
        image = Image.open(image_path, mode="r")
        image = image.convert("RGB")

        # print(f'{bbs} {classes}')
        boxes = torch.FloatTensor(bbs).reshape((-1, 4))  # (n_objects, 4)
        labels = torch.LongTensor(classes)  # (n_objects)
        difficulties = torch.zeros_like(labels).byte()  # (n_objects)

        # Apply transformations
        if self.transform is not None:
            image, boxes, labels, difficulties = transform(
                image, boxes, labels, difficulties, split=self.split.upper()
            )

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.image_paths)

    def get_labels(self):
        return tuple(kitti_utils.KITTI_CLASSES)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        return collate_fn(batch)
