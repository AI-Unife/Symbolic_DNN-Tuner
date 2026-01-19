import torch
import torch.utils.data
import os
from PIL import Image
from natsort import natsorted

from vision_models_pytorch.utils.split_creator import create_train_val_test_split_txt


class CarSteeringAngle(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path="downloaded_dataset/car_steering_angle",
        split="test",
        transform=None,
        seed=None,
    ):
        # Check if split is cprrect
        self.splits = ["train", "valid", "test"]
        assert split in self.splits, "Split must be one of: train,valid,test"

        self.root_path = root_path
        self.split = split
        self.transform = transform
        # Images are in the root
        self.image_folder_path = os.path.join(self.root_path, "images")
        self.splits_folder_path = os.path.join(self.root_path, "splits")
        self.labels_file_path = os.path.join(self.root_path, "labels", "angles.txt")
        self.seed = seed

        # If split files are not defined, define them
        if not all(
            os.path.exists(os.path.join(self.splits_folder_path, f"{splt}.txt"))
            for splt in self.splits
        ):
            create_train_val_test_split_txt(
                self.image_folder_path,
                self.splits_folder_path,
                seed=self.seed,
                # Consider only labeled samples (up to 45405.jpg)
                file_filter=lambda x: int(x[:-4]) <= 45405,
            )

        # Load split file
        self.split_file_path = os.path.join(
            self.splits_folder_path, f"{self.split}.txt"
        )
        self.images_filenames = []
        with open(self.split_file_path, "r") as f:
            for line in f.readlines():
                self.images_filenames.append(line.strip("\n"))

        all_labels = {}
        with open(self.labels_file_path, "r") as f:
            for line in f.readlines():
                image, angle = line.strip("\n").split(" ")
                all_labels[image] = float(angle)
        # Filter
        self.labels = [all_labels[file] for file in self.images_filenames]

    def __getitem__(self, index):
        image_filename = self.images_filenames[index]
        label = self.labels[index]
        image_full_path = os.path.join(self.image_folder_path, image_filename)
        # The age label is in the image name (before first underscore)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        image = Image.open(image_full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_tensor

    def __len__(self):
        return len(self.images_filenames)
