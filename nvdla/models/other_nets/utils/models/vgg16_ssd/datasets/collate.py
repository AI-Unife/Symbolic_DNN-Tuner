import torch


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    This describes how to combine these tensors of different sizes. We use lists.

    Note: this need not be defined in this Class, can be standalone.

    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    boxes = list()
    labels = list()
    difficulties = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
        difficulties.append(b[3])

    images = torch.stack(images, dim=0)

    return (
        images,
        boxes,
        labels,
        difficulties,
    )  # tensor (N, 3, 300, 300), 3 lists of N tensors each
