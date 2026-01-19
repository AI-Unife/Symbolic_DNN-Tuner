from typing import Callable, Optional, Sequence
import numpy as np
from sklearn.model_selection import train_test_split
import os


def create_train_val_test_split_txt(
    image_folder_path: str,
    splits_folder_path: str,
    seed: Optional[int] = None,
    split_proportion: Sequence[float] = [0.7, 0.2, 0.1],
    split_names: Sequence[str] = ["train.txt", "valid.txt", "test.txt"],
    file_filter: Optional[Callable[[str], bool]] = None,
) -> None:
    """
    Split the dataset in three parts (train, validation, test), creating
    for each split a txt file with the name of the files included in the split.

    Args
    ----
    * ``image_folder_path``: str

        Path to the folder that contains all images
    * ``split_folder_path``: str

        Path where to save the split .txt files
    * ``seed``: int

        Seed for initializing random state
    * ``split_proportion``: Sequence[float]

        An array of three numbers. All the components of the array will be divided
        by their sum, so that they array sums to 1. Each of the three number represents
        the proportion of images that will end up in respectively the train, validation
        and test splits.
    * ``split_names`` : Sequence[str]

        An array of three strings, containg the name of the file to be created respectively
        for the train, validation and test split.
    * ``file_filter`` : Optional[Callable[[str], bool]]

        A function that takes in input a file name and returns a bool that indicates wether the file should
        be included in the splits. This function can be used to exclude certain files from the split by their name.
    """
    random_state = np.random.RandomState(seed=seed)

    assert (
        len(split_proportion) == len(split_names) == 3
    ), "There must exactly be three splits"
    # Normalize split proportion so it sums to 1.0
    split_proportion = np.array(split_proportion)
    split_proportion = split_proportion / split_proportion.sum()
    file_filter = file_filter or (lambda _: True)

    image_file_names = list(filter(file_filter, os.listdir(image_folder_path)))
    os.makedirs(splits_folder_path, exist_ok=True)
    # train 70%, val 20%, test 10%
    # split train from testval

    train_prop, valid_prop, test_prop = split_proportion

    images_train, images_testval = train_test_split(
        image_file_names, train_size=train_prop, random_state=random_state
    )
    # then split testval in test and val, calculating the division between the two datasets
    valid_sub_prop = valid_prop / (valid_prop + test_prop)

    images_val, image_test = train_test_split(
        images_testval, train_size=valid_sub_prop, random_state=random_state
    )
    # Write a txt file for each split. Each line contains the base name of the image
    for splt, images in zip(split_names, [images_train, images_val, image_test]):
        split_file_path = os.path.join(splits_folder_path, splt)
        with open(split_file_path, "w") as f:
            for image in images:
                f.write(image + "\n")
