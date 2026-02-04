from dataclasses import dataclass
from typing import Tuple


@dataclass
class CityscapesClass:
    name: str
    id: int
    train_id: int
    category: str
    category_id: int
    has_instances: bool
    ignore_in_eval: bool
    color: Tuple[int, int, int]


CITYSCAPE_CLASSES = [
    CityscapesClass("unlabeled", 0, 0, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("ego vehicle", 1, 0, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("rectification border", 2, 0, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("out of roi", 3, 0, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("static", 4, 0, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("dynamic", 5, 0, "void", 0, False, True, (111, 74, 0)),
    CityscapesClass("ground", 6, 0, "void", 0, False, True, (81, 0, 81)),
    CityscapesClass("road", 7, 1, "flat", 1, False, False, (128, 64, 128)),
    CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    CityscapesClass("parking", 9, 0, "flat", 1, False, True, (250, 170, 160)),
    CityscapesClass("rail track", 10, 0, "flat", 1, False, True, (230, 150, 140)),
    CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    CityscapesClass("wall", 12, 2, "construction", 2, False, False, (102, 102, 156)),
    CityscapesClass("fence", 13, 2, "construction", 2, False, False, (190, 153, 153)),
    CityscapesClass(
        "guard rail", 14, 0, "construction", 2, False, True, (180, 165, 180)
    ),
    CityscapesClass("bridge", 15, 0, "construction", 2, False, True, (150, 100, 100)),
    CityscapesClass("tunnel", 16, 0, "construction", 2, False, True, (150, 120, 90)),
    CityscapesClass("pole", 17, 3, "object", 3, False, False, (153, 153, 153)),
    CityscapesClass("polegroup", 18, 0, "object", 3, False, True, (153, 153, 153)),
    CityscapesClass("traffic light", 19, 3, "object", 3, False, False, (250, 170, 30)),
    CityscapesClass("traffic sign", 20, 3, "object", 3, False, False, (220, 220, 0)),
    CityscapesClass("vegetation", 21, 4, "nature", 4, False, False, (107, 142, 35)),
    CityscapesClass("terrain", 22, 4, "nature", 4, False, False, (152, 251, 152)),
    CityscapesClass("sky", 23, 5, "sky", 5, False, False, (70, 130, 180)),
    CityscapesClass("person", 24, 6, "human", 6, True, False, (220, 20, 60)),
    CityscapesClass("rider", 25, 6, "human", 6, True, False, (255, 0, 0)),
    CityscapesClass("car", 26, 7, "vehicle", 7, True, False, (0, 0, 142)),
    CityscapesClass("truck", 27, 7, "vehicle", 7, True, False, (0, 0, 70)),
    CityscapesClass("bus", 28, 7, "vehicle", 7, True, False, (0, 60, 100)),
    CityscapesClass("caravan", 29, 0, "vehicle", 7, True, True, (0, 0, 90)),
    CityscapesClass("trailer", 30, 0, "vehicle", 7, True, True, (0, 0, 110)),
    CityscapesClass("train", 31, 7, "vehicle", 7, True, False, (0, 80, 100)),
    CityscapesClass("motorcycle", 32, 7, "vehicle", 7, True, False, (0, 0, 230)),
    CityscapesClass("bicycle", 33, 7, "vehicle", 7, True, False, (119, 11, 32)),
    CityscapesClass("license plate", -1, 0, "vehicle", 7, False, True, (0, 0, 142)),
]

CITYSCAPE_ID_TO_TRAIN = {cl.id: cl.category_id for cl in CITYSCAPE_CLASSES}

CITYSCAPE_N_CLASSES = len(set(CITYSCAPE_ID_TO_TRAIN.values()))

CITYSCAPE_CLASS_NAMES = {cl.category_id: cl.category for cl in CITYSCAPE_CLASSES}
