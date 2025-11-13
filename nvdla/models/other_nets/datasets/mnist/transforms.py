import torchvision


def data_transform(resize=None):
    """
    Transformations for MNIST Dataset in inference mode.

    Args
    ---
    * ``resize : Optional[Tuple[int,int]]``: If None image is kept to 28x28, otherwise it is resized to the size provided

    Returns
    ---
    The inference mode torchvision transform for MNIST
    """

    base_transforms = [
            lambda x: x.convert('RGB'),
            torchvision.transforms.ToTensor(),
    ]
    
    if resize is not None:
        base_transforms += [torchvision.transforms.Resize(resize)]

    data_transforms = torchvision.transforms.Compose(
        base_transforms
    )

    return data_transforms

def train_transform(resize=None):
    """
    Transformations for MNIST Dataset in training mode.

    Args
    ---
    * ``resize : Optional[Tuple[int,int]]``: If None image is kept to 28x28, otherwise it is resized to the size provided

    Returns
    ---
    The training mode torchvision transform for MNIST
    """

    base_transforms = [
            lambda x: x.convert('RGB'),
            torchvision.transforms.ToTensor(),
    ]
    
    if resize is not None:
        base_transforms += [torchvision.transforms.Resize(resize)]

    base_transforms += [torchvision.transforms.RandomHorizontalFlip()]

    train_transforms = torchvision.transforms.Compose(
        base_transforms
    )

    return train_transforms