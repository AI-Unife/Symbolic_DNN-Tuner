import os
from typing import Dict, List, Tuple


def get_classes() -> Tuple[str]:
    return ("boat", "car", "dock", "jetski", "lift")


def parse_annotation_file(
    split_path: str, class_offset=0
) -> Dict[str, Tuple[List[int], int]]:
    """
    Parses an annotation file for aerial maritime images, extracting bounding boxes and class labels.

    This function reads a given annotation file named "_annotations.txt" located in the split directory toghether with images.
    Each line in the annotation file should contain an image name followed by space-separated bounding box
    annotations and their corresponding class labels. Each bounding box and class label are specified as
    `x0,y0,x1,y1,c`, where `(x0,y0)` and `(x1,y1)` are the coordinates of the top-left and bottom-right corners
    of the bounding box, respectively, and `c` is the class label index.

    Args
    ----
        split_path (str): The path to the directory containing the "_annotations.txt" file.

        class_offset (int): Increments the id of all class field by a constant number.

    Returns
    ----
        Dict[str, Tuple[List[int], int]]: A dictionary where each key is an image name and its value is a tuple
        containing a list of bounding boxes and a list of incremented class labels for that image. Each bounding
        box is represented as a list of four integers `[x0, y0, x1, y1]`, and class labels are stored as integers.
    """
    annotation_path = os.path.join(split_path, "_annotations.txt")
    data = {}
    with open(annotation_path, "r") as f:
        for _, line in enumerate(f.readlines()):
            line = line.strip("\n ")
            tokens = line.split(" ")
            image_name = tokens[0]
            bbs = []
            classes = []
            for bb_string in tokens[1:]:
                x0, y0, x1, y1, c = tuple(map(int, bb_string.split(",")))
                bbs.append([x0, y0, x1, y1])
                classes.append(c + class_offset)

            data[image_name] = (bbs, classes)

    return data
