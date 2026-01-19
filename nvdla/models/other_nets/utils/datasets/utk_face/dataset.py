import torch
import torch.utils.data
import os
from PIL import Image

from vision_models_pytorch.utils.split_creator import create_train_val_test_split_txt


class UtkFace(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path="downloaded_dataset/utk_face",
        split="test",
        transform=None,
        seed=None,
    ):
        # Check if params are correct
        self.splits = ["train", "valid", "test"]
        assert split in self.splits, "Split must be one of: train,valid,test"

        self.root_path = root_path
        self.split = split
        self.transform = transform

        self.image_folder_path = os.path.join(self.root_path, "images")
        self.splits_folder_path = os.path.join(self.root_path, "splits")
        self.seed = seed

        # If split files are not defined, define them
        if not all(
            os.path.exists(os.path.join(self.splits_folder_path, f"{splt}.txt"))
            for splt in self.splits
        ):
            create_train_val_test_split_txt(
                self.image_folder_path, self.splits_folder_path, seed=self.seed
            )

        # Load split file
        self.split_file_path = os.path.join(
            self.splits_folder_path, f"{self.split}.txt"
        )
        self.images_filenames = []
        with open(self.split_file_path, "r") as f:
            for line in f.readlines():
                self.images_filenames.append(line.strip("\n"))

    def __getitem__(self, index):
        image_filename = self.images_filenames[index]
        image_full_path = os.path.join(self.image_folder_path, image_filename)
        # The age label is in the image name (before first underscore)
        age_label = torch.tensor(
            [int(image_filename.split("_")[0])], dtype=torch.float32
        )
        image = Image.open(image_full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, age_label

    def __len__(self):
        return len(self.images_filenames)
