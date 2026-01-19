import os
from typing import List
import numpy as np

KITTI_CLASSES = (
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
)

kitti_map = {cl: i + 1 for i, cl in enumerate(KITTI_CLASSES)}
kitti_rev_map = {i + 1: cl for i, cl in enumerate(KITTI_CLASSES)}
kitti_rev_map[0] = "Misc"


def create_test_val_split(
    dataset_folder: str = "downloaded_datasets/kitti/training",
    train_proportion: float = 0.8,
    seed=42,
    image_subdir="image_2",
    label_subdir="label_2",
    train_output_path="train.txt",
    val_output_path="val.txt",
):
    np.random.seed(seed)
    images_dir_path = os.path.join(dataset_folder, image_subdir)
    labels_dir_path = os.path.join(dataset_folder, label_subdir)
    images_file_names = os.listdir(images_dir_path)
    labels_file_names = os.listdir(labels_dir_path)

    assert len(images_file_names) == len(
        labels_file_names
    ), "Labels and Images count differ"
    # All filenames are number padded to 6 digits
    # Formats are .png and .txt
    image_file_nums = [int(name[:-4]) for name in images_file_names]
    label_file_nums = [int(name[:-4]) for name in labels_file_names]

    selected_train: List[int] = sorted(
        np.random.choice(
            image_file_nums, int(train_proportion * len(image_file_nums)), replace=False
        )
    )
    selected_val = sorted(set(image_file_nums) - set(selected_train))

    with open(train_output_path, "w") as f:
        for item in selected_train:
            f.write(f"{str(item).zfill(6)}\n")
    with open(val_output_path, "w") as f:
        for item in selected_val:
            f.write(f"{str(item).zfill(6)}\n")


def get_split_files(
    dataset_folder: str,
    split_file_path: str,
    image_subdir="image_2",
    label_subdir="label_2",
):
    image_paths = list()
    labels_paths = list()
    with open(split_file_path, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            path = os.path.join(dataset_folder, image_subdir, f"{line}.png")
            image_paths.append(path)
            path = os.path.join(dataset_folder, label_subdir, f"{line}.txt")
            labels_paths.append(path)

    assert len(image_paths) == len(labels_paths)

    return image_paths, labels_paths


def parse_label_file(label_file_path: str):
    bboxes = []
    classes = []
    with open(label_file_path, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            data = line.split(" ")
            obj_class = data[0]
            if obj_class == "Misc" or obj_class == "DontCare":
                continue

            class_id = kitti_map[obj_class]
            truncated = float(data[1])
            occluded = int(data[2])
            obs_angle = float(data[3])
            bbox = list(map(float, data[4:8]))
            dim_3d = list(map(float, data[8:11]))
            loc_3d = list(map(float, data[11:14]))
            rot_y_3d = float(data[14])
            # score = data[15]
            bboxes.append(bbox)
            classes.append(class_id)
    return bboxes, classes


def create_detection_file(output_path: str, class_ids, bboxes, scores):
    assert len(class_ids) == len(bboxes) == len(scores)

    output = [["invalid" for i in range(16)] for _ in range(len(class_ids))]

    for i, (clz_id, bbox, score) in enumerate(zip(class_ids, bboxes, scores)):
        output[i][0] = kitti_rev_map[clz_id]
        output[i][4:8] = bbox
        output[i][15] = score

    with open(output_path, "w") as f:
        for line in output:
            f.write(" ".join(line) + "\n")


if __name__ == "__main__":
    create_test_val_split()
