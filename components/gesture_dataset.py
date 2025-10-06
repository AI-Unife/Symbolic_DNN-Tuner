import tensorflow as tf
import numpy as np
import tonic
import tonic.transforms as transforms
from sklearn.model_selection import train_test_split
from pathlib import Path

import config as cfg

class ToOneHotTimeCoding:
    """
    Converts a target label into a one-hot encoded representation, repeated over multiple time frames.

    Parameters:
    - n_classes (int): Total number of classes.
    - n_frames (int): Number of time frames for which the one-hot encoding should be repeated.

    Returns:
    - tf.Tensor: A stacked one-hot encoded tensor for all time frames.
    """
    def __init__(self, n_classes, n_frames):
        self.n_classes = n_classes
        self.n_frames = n_frames

    def __call__(self, target):
        one_hot = tf.one_hot(target, self.n_classes)  # Convert target to one-hot encoding
        to_stack = [one_hot] * self.n_frames
        return tf.stack(to_stack)  # Repeat across time frames


class SumPolarity:
    """
    Sums the two polarity channels of an event-based frame.

    Returns:
    - tf.Tensor: Image with summed polarity channels.
    """
    def __call__(self, image):
        return image[:, 0, :] + image[:, 1, :]


class DropPolarity:
    """
    Drops one of the polarity channels (either positive or negative events).

    Parameters:
    - p (int): Index of the polarity channel to keep (0 or 1).

    Returns:
    - tf.Tensor: Image with a single polarity channel.
    """
    def __call__(self, image, p=0):
        return image[:, p, :]
    
class BothPolarity:
    """
    Keeps both polarity channels of an event-based frame.

    Returns:
    - tf.Tensor: Image with both polarity channels.
    """
    def __call__(self, image):
        # print("image shape: ", image.shape)
        
        new_image = image.reshape(cfg.FRAMES*2, 64, 64)  # (FRAMES, Polarity,  64, 64) --> (FRAMES * Polarity , 64, 64)
        
        return new_image  


class SubPolarity:
    """
    Computes the difference between the two polarity channels.

    Returns:
    - tf.Tensor: Image where positive and negative events are subtracted.
    """
    def __call__(self, image):
        return image[:, 0, :] - image[:, 1, :]


def select_polarity_transform(polarity):
    """Returns the appropriate polarity transformation based on user input."""
    polarity_mapping = {
        "sum": SumPolarity(),
        "sub": SubPolarity(),
        "drop": DropPolarity(),
        "both": BothPolarity()
    }
    if polarity not in polarity_mapping:
        raise ValueError("Invalid polarity option! Choose from ['sum', 'sub', 'drop']")
    return polarity_mapping[polarity]

class ROIDataset(tonic.dataset.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(save_to='.data/')
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        self.samples = []
        self.targets = []
        self.classes = []

        # Raccoglie i file e classi (cartelle)
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.classes:
                self.classes.append(class_name)

            label = self.classes.index(class_name)

            for file in class_dir.glob("*.npy"):
                self.samples.append(file)
                self.targets.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        events = np.load(self.samples[index])  # shape (N, 4)
        label = self.targets[index]

        if self.transform:
            events = self.transform(events)
        if self.target_transform:
            label = self.target_transform(label)

        return events, label
    

def dataset_to_numpy(dataset):
    x_list, y_list = [], []
    for x, y in dataset:
        if cfg.MODE == "fwdPass":
            x_list.append(np.transpose(np.array(x), (0, 2, 3, 1)))
        elif cfg.MODE == "hybrid":
            # Step 1: Reshape per raggruppare la prima dimensione in gruppi di 4
            X_grouped = x.reshape(cfg.FRAMES//cfg.NUM_CHANNELS, cfg.NUM_CHANNELS, 2, 64, 64)  # (4 gruppi, 4 elementi, 2, 64, 64)

            # Step 2: Portare la dimensione dei 4 elementi insieme ai 2 canali
            X_transposed = X_grouped.transpose(0, 3, 4, 1, 2)  # (4, 64, 64, 4, 2)

            # Step 3: Unire le ultime due dimensioni (4*2 = 8)
            x_final = X_transposed.reshape(cfg.FRAMES//cfg.NUM_CHANNELS, 64, 64, cfg.NUM_CHANNELS * 2)    
            x_list.append(x_final)
        else:
            x_list.append(np.transpose(np.array(x), (1, 2, 0)))  # Convert to NumPy
        y_list.append(np.array(y))  # Convert to NumPy

    return np.array(x_list), np.array(y_list)


def get_datasets_numpy():
    """
    Loads and processes the specified dataset using Tonic and returns NumPy arrays.

    Returns:
    - tuple: ((x_train, y_train), (x_test, y_test)) as NumPy arrays.
    """
    
    dataset_path = "/hpc/home/bzzlca/AIDA4Edge/data/"
    polarity = cfg.POLARITY
    n_pol = 2 if polarity == "both" else 1
    cache_dir= f"/hpc/home/bzzlca/AIDA4Edge/tf/cache/DVSGesture_{cfg.MODE}_{polarity}_{cfg.FRAMES}_{cfg.NUM_CHANNELS}_{n_pol}/"
    print("cache_dir: ", cache_dir)
    # exit()
    transform = [
        transforms.Denoise(filter_time=10000),
        transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(64, 64)),
        transforms.ToFrame(sensor_size=(64, 64, 2), n_time_bins=cfg.FRAMES)
    ]
    
    if cfg.MODE == "fwdPass":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.FRAMES)
    elif cfg.MODE == "hybrid":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.FRAMES//cfg.NUM_CHANNELS)
    else:
        transform.append(select_polarity_transform(polarity))
        target_transform = None
    
    transform = transforms.Compose(transform)
    # Load dataset
    train = tonic.datasets.DVSGesture(
        save_to=dataset_path, transform=transform, target_transform=target_transform,
        train=True
    )

    test = tonic.datasets.DVSGesture(
        save_to=dataset_path, transform=transform, train=False, target_transform=target_transform
    )
    
    cached_train = tonic.DiskCachedDataset(train, cache_path=cache_dir + 'train')
    cached_test = tonic.DiskCachedDataset(test, cache_path=cache_dir + 'test')

    # Convert dataset to NumPy arrays
    x_train, y_train = dataset_to_numpy(cached_train)
    x_test, y_test = dataset_to_numpy(cached_test)

    return (x_train, y_train), (x_test, y_test)

def get_ROI_numpy():
    """
    Loads and processes the specified dataset using Tonic and returns NumPy arrays.

    Returns:
    - tuple: ((x_train, y_train), (x_test, y_test)) as NumPy arrays.
    """
    
    dataset_path = "datasets/DVS_ROI/"
    polarity = cfg.POLARITY
    n_pol = 2 if polarity == "both" else 1
    cache_dir= f"/hpc/home/bzzlca/AIDA4Edge/tf/cache/DVS_ROI_{cfg.MODE}_{polarity}_{cfg.FRAMES}_{cfg.NUM_CHANNELS}_{n_pol}/"
    print("cache_dir: ", cache_dir)
    transform = [
        transforms.Denoise(filter_time=10000),
        transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(64, 64)),
        transforms.ToFrame(sensor_size=(64, 64, 2), n_time_bins=cfg.FRAMES)
    ]
    
    if cfg.MODE == "fwdPass":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.FRAMES)
    elif cfg.MODE == "hybrid":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.FRAMES//cfg.NUM_CHANNELS)
    else:
        transform.append(select_polarity_transform(polarity))
        target_transform = None
    
    transform = transforms.Compose(transform)
    # Load dataset
    train = ROIDataset(
        root=dataset_path+'/train', transform=transform, target_transform=target_transform
    )

    test = ROIDataset(
        root=dataset_path+'/test', transform=transform, target_transform=target_transform
    )
    
    cached_train = tonic.DiskCachedDataset(train, cache_path=cache_dir + 'train')
    cached_test = tonic.DiskCachedDataset(test, cache_path=cache_dir + 'test')

    # Convert dataset to NumPy arrays
    x_train, y_train = dataset_to_numpy(cached_train)
    x_test, y_test = dataset_to_numpy(cached_test)

    return (x_train, y_train), (x_test, y_test)

def ROI_data():
    return gesture_data(ROI=True)

def gesture_data(num_classes = 11, ROI=False):
    # The data, split between train and test sets:
    if ROI:
        (x_train, y_train), (x_test, y_test) = get_ROI_numpy()
    else:
        (x_train, y_train), (x_test, y_test) = get_datasets_numpy()
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    if cfg.MODE == "depth":
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    print("shape of x_train: ", x_train.shape)
    print("shape of y_train: ", y_train.shape)
    print("shape of x_test: ", x_test.shape)
    print("shape of y_test: ", y_test.shape)
    return x_train, x_test, y_train, y_test

