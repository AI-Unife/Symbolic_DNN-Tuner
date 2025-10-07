from __future__ import annotations

from typing import Tuple, List
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tonic
import tonic.transforms as transforms
from sklearn.model_selection import train_test_split  # kept for compatibility if needed elsewhere

import config as cfg


# ------------------------------- Targets -------------------------------------

class ToOneHotTimeCoding:
    """
    Convert an integer class label to one-hot and repeat it across time frames.

    Args:
        n_classes: total number of classes.
        n_frames: number of time frames to replicate the one-hot vector over.

    Returns:
        A tensor of shape [T, C] where T == n_frames and C == n_classes.
    """
    def __init__(self, n_classes: int, n_frames: int):
        self.n_classes = n_classes
        self.n_frames = n_frames

    def __call__(self, target: int) -> tf.Tensor:
        one_hot = tf.one_hot(target, self.n_classes)  # [C]
        # Repeat across time: list-of-T copies stacked -> [T, C]
        to_stack = [one_hot] * self.n_frames
        return tf.stack(to_stack, axis=0)


# ----------------------------- Polarity ops ----------------------------------

class SumPolarity:
    """
    Sum the two polarity channels (+ and -) of an event-based framed tensor.

    Expects input shaped [T, 2, H, W]; returns [T, H, W].
    """
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Robustness: accept any shape where channel index is 1 and equals 2.
        # We assume 'image' was produced by transforms.ToFrame(sensor_size=(H, W, 2), ...)
        # which yields shape [T, 2, H, W].
        assert image.ndim == 4 and image.shape[1] == 2, f"Expected [T,2,H,W], got {image.shape}"
        return image[:, 0, ...] + image[:, 1, ...]


class DropPolarity:
    """
    Keep a single polarity channel (0 for positive, 1 for negative).

    Expects [T, 2, H, W]; returns [T, H, W].
    """
    def __call__(self, image: np.ndarray, p: int = 0) -> np.ndarray:
        assert image.ndim == 4 and image.shape[1] == 2, f"Expected [T,2,H,W], got {image.shape}"
        assert p in (0, 1), "p must be 0 or 1"
        return image[:, p, ...]


class BothPolarity:
    """
    Keep both polarity channels by concatenating them along the time axis.

    Expects [T, 2, H, W]; returns [(T*2), H, W].
    """
    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim == 4 and image.shape[1] == 2, f"Expected [T,2,H,W], got {image.shape}"
        T, C, H, W = image.shape  # C == 2
        # Reshape (T, 2, H, W) -> (T*2, H, W) without relying on cfg.FRAMES/64
        return image.transpose(0, 2, 3, 1).reshape(T * C, H, W).transpose(0, 1, 2)


class SubPolarity:
    """
    Subtract negative from positive polarity.

    Expects [T, 2, H, W]; returns [T, H, W].
    """
    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim == 4 and image.shape[1] == 2, f"Expected [T,2,H,W], got {image.shape}"
        return image[:, 0, ...] - image[:, 1, ...]


def select_polarity_transform(polarity: str):
    """
    Pick the appropriate polarity transform by name.

    Valid options: 'sum', 'sub', 'drop', 'both'
    """
    mapping = {
        "sum": SumPolarity(),
        "sub": SubPolarity(),
        "drop": DropPolarity(),
        "both": BothPolarity(),
    }
    if polarity not in mapping:
        raise ValueError("Invalid polarity option! Choose from ['sum', 'sub', 'drop', 'both'].")
    return mapping[polarity]


# ------------------------------- Datasets ------------------------------------

class ROIDataset(tonic.dataset.Dataset):
    """
    Minimal dataset reading .npy frames arranged as:
        root/
          class_a/*.npy
          class_b/*.npy
          ...
    Each .npy is expected to contain a framed tensor compatible with the transform chain.

    Args:
        root: root directory containing per-class folders of .npy files.
        transform: optional transform to apply to the loaded sample.
        target_transform: optional transform to apply to the label (e.g., ToOneHotTimeCoding).
    """
    def __init__(self, root: str | Path, transform=None, target_transform=None):
        super().__init__(save_to=".data/")
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        self.samples: List[Path] = []
        self.targets: List[int] = []
        self.classes: List[str] = []

        if not self.root.exists():
            raise FileNotFoundError(f"ROI root not found: {self.root}")

        # Collect files and build class index
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.classes:
                self.classes.append(class_name)
            label = self.classes.index(class_name)

            for file in sorted(class_dir.glob("*.npy")):
                self.samples.append(file)
                self.targets.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        events = np.load(self.samples[index])  # shape depends on how frames are saved
        label = self.targets[index]

        if self.transform:
            events = self.transform(events)
        if self.target_transform:
            label = self.target_transform(label)

        return events, label


# ----------------------------- Numpy adapters --------------------------------

def dataset_to_numpy(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a (cached) tonic dataset into NumPy arrays with the layout expected by the model.

    Output X layout depends on cfg.MODE:
      - "fwdPass":    input per time step with 2 channels -> [B, T, H, W, 2]
      - "hybrid":     merge groups of NUM_CHANNELS frames and both polarities -> [B, T', H, W, 2*NUM_CHANNELS]
      - "depth":      single frame with possibly combined polarity -> [B, H, W, C]
    Targets (y):
      - "fwdPass"/"hybrid": [B, T, C] (time-repeated one-hot via ToOneHotTimeCoding)
      - "depth":            integer labels; one-hot added later in gesture_data()
    """
    x_list, y_list = [], []
    for x, y in dataset:
        # x is typically [T, 2, H, W] after ToFrame(sensor_size=(H, W, 2), ...)
        arr = np.array(x)

        if cfg.MODE == "fwdPass":
            # [T, 2, H, W] -> [T, H, W, 2]
            x_list.append(np.transpose(arr, (0, 2, 3, 1)))

        elif cfg.MODE == "hybrid":
            # Expect BothPolarity OR a prior framing such that arr has shape [T, 2, H, W]
            # We reshape into groups of NUM_CHANNELS along time and stack polarities as channels.
            T, C, H, W = arr.shape  # C should be 2
            assert T % cfg.NUM_CHANNELS == 0, f"T={T} not divisible by NUM_CHANNELS={cfg.NUM_CHANNELS}"
            # Group frames: (T//NUM_CHANNELS, NUM_CHANNELS, C, H, W)
            grouped = arr.reshape(T // cfg.NUM_CHANNELS, cfg.NUM_CHANNELS, C, H, W)
            # Move to (T', H, W, NUM_CHANNELS, C)
            transposed = grouped.transpose(0, 3, 4, 1, 2)
            # Merge channel dims -> (T', H, W, NUM_CHANNELS*C) == (T', H, W, 2*NUM_CHANNELS)
            x_final = transposed.reshape(T // cfg.NUM_CHANNELS, H, W, cfg.NUM_CHANNELS * C)
            x_list.append(x_final)

        else:  # "depth" or any other single-frame mode
            # [T, 2, H, W] -> assume polarity transform reduced to [T, H, W] or similar
            # If still [T, 2, H, W], collapse time by sum; otherwise, keep as is.
            if arr.ndim == 4 and arr.shape[1] == 2:
                arr = arr.sum(axis=0)  # simple collapse to [H, W] (choose your policy)
            # Ensure shape is [H, W] or [H, W, C]
            if arr.ndim == 2:
                arr = arr[..., None]  # [H, W, 1]
            elif arr.ndim == 3 and arr.shape[0] not in (1, 2, 3, 4):  # likely [T, H, W]
                arr = np.transpose(arr, (1, 2, 0))
            x_list.append(arr)

        y_list.append(np.array(y))

    return np.array(x_list), np.array(y_list)


# ---------------------------- Dataset loaders --------------------------------

# ---------- resolve dataset/cache paths with fallback ----------
def _resolve_paths(dataset_path: str | Path, cache_dir: str | Path) -> tuple[str, str]:
    """
    Ensure dataset_path and cache_dir exist. If not:
      - dataset_path -> './data'
      - cache_dir    -> './cache'
    Returns normalized absolute paths (str).
    """
    dp = Path(dataset_path)
    cd = Path(cache_dir)

    # dataset path fallback
    if not dp.exists():
        dp = Path("./data")
        dp.mkdir(parents=True, exist_ok=True)
        print(f"[info] dataset_path not found. Falling back to: {dp.resolve()}")

    # cache path fallback
    if not cd.exists():
        cd = Path("./cache")
        cd.mkdir(parents=True, exist_ok=True)
        print(f"[info] cache_dir not found. Falling back to: {cd.resolve()}")

    return str(dp.resolve()), str(cd.resolve())

def _ensure_cache_dir(cache_dir: str) -> None:
    Path(cache_dir, "train").mkdir(parents=True, exist_ok=True)
    Path(cache_dir, "test").mkdir(parents=True, exist_ok=True)

def get_datasets_numpy():
    """
    Load DVSGesture through Tonic, apply transforms, and return NumPy arrays.

    Returns:
        ((x_train, y_train), (x_test, y_test))
    """
    dataset_path = "/hpc/home/bzzlca/AIDA4Edge/data/"
    polarity = cfg.POLARITY
    n_pol = 2 if polarity == "both" else 1
    cache_dir = f"/hpc/home/bzzlca/AIDA4Edge/tf/cache/DVSGesture_{cfg.MODE}_{polarity}_{cfg.FRAMES}_{cfg.NUM_CHANNELS}_{n_pol}/"
    # resolve with fallback
    dataset_path, cache_dir = _resolve_paths(dataset_path, cache_dir)
    _ensure_cache_dir(cache_dir)
    print("dataset_path:", dataset_path)
    print("cache_dir:", cache_dir)

    # Base framing to [T, 2, H, W] with H=W=64
    tfms: List = [
        transforms.Denoise(filter_time=10000),
        transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(64, 64)),
        transforms.ToFrame(sensor_size=(64, 64, 2), n_time_bins=cfg.FRAMES),
    ]

    # Labels repeated across time if the model expects temporal supervision
    if cfg.MODE == "fwdPass":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.FRAMES)
    elif cfg.MODE == "hybrid":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.FRAMES // cfg.NUM_CHANNELS)
    else:
        # For single-frame modes, optionally apply a polarity transform to images
        tfms.append(select_polarity_transform(polarity))
        target_transform = None

    transform = transforms.Compose(tfms)

    train = tonic.datasets.DVSGesture(
        save_to=dataset_path, transform=transform, target_transform=target_transform, train=True
    )
    test = tonic.datasets.DVSGesture(
        save_to=dataset_path, transform=transform, target_transform=target_transform, train=False
    )

    cached_train = tonic.DiskCachedDataset(train, cache_path=os.path.join(cache_dir, "train"))
    cached_test = tonic.DiskCachedDataset(test, cache_path=os.path.join(cache_dir, "test"))

    x_train, y_train = dataset_to_numpy(cached_train)
    x_test, y_test = dataset_to_numpy(cached_test)

    return (x_train, y_train), (x_test, y_test)


def get_ROI_numpy():
    """
    Load a folder-structured ROI dataset via ROIDataset and return NumPy arrays.

    Returns:
        ((x_train, y_train), (x_test, y_test))
    """
    dataset_path = "datasets/DVS_ROI/"
    polarity = cfg.POLARITY
    n_pol = 2 if polarity == "both" else 1
    cache_dir = f"/hpc/home/bzzlca/AIDA4Edge/tf/cache/DVS_ROI_{cfg.MODE}_{polarity}_{cfg.FRAMES}_{cfg.NUM_CHANNELS}_{n_pol}/"
    _ensure_cache_dir(cache_dir)
    print("cache_dir:", cache_dir)

    tfms: List = [
        transforms.Denoise(filter_time=10000),
        # Reuse DVSGesture sensor size only as a (H,W) hint for downsampling,
        # since we target a fixed (64, 64) output anyway.
        transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(64, 64)),
        transforms.ToFrame(sensor_size=(64, 64, 2), n_time_bins=cfg.FRAMES),
    ]

    if cfg.MODE == "fwdPass":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.FRAMES)
    elif cfg.MODE == "hybrid":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.FRAMES // cfg.NUM_CHANNELS)
    else:
        tfms.append(select_polarity_transform(polarity))
        target_transform = None

    transform = transforms.Compose(tfms)

    train = ROIDataset(root=os.path.join(dataset_path, "train"), transform=transform, target_transform=target_transform)
    test = ROIDataset(root=os.path.join(dataset_path, "test"), transform=transform, target_transform=target_transform)

    cached_train = tonic.DiskCachedDataset(train, cache_path=os.path.join(cache_dir, "train"))
    cached_test = tonic.DiskCachedDataset(test, cache_path=os.path.join(cache_dir, "test"))

    x_train, y_train = dataset_to_numpy(cached_train)
    x_test, y_test = dataset_to_numpy(cached_test)

    return (x_train, y_train), (x_test, y_test)


# ------------------------------ Public API -----------------------------------

def ROI_data():
    """Compatibility shim if external code expects this name."""
    return gesture_data(ROI=True)


def gesture_data(num_classes: int = 11, ROI: bool = False):
    """
    End-to-end loader producing NumPy arrays ready for model consumption.

    Shapes:
      - fwdPass:
          X: [B, T, H, W, 2]
          y: [B, T, C]
      - hybrid:
          X: [B, T', H, W, 2*NUM_CHANNELS] with T' = T/NUM_CHANNELS
          y: [B, T', C]
      - depth:
          X: [B, H, W, C]
          y: integer labels (will be one-hot here)

    Returns:
        x_train, x_test, y_train, y_test
    """
    if ROI:
        (x_train, y_train), (x_test, y_test) = get_ROI_numpy()
    else:
        (x_train, y_train), (x_test, y_test) = get_datasets_numpy()

    # Convert labels to one-hot for single-frame modes
    if cfg.MODE == "depth":
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    print("shape of x_train:", x_train.shape)
    print("shape of y_train:", y_train.shape)
    print("shape of x_test:", x_test.shape)
    print("shape of y_test:", y_test.shape)
    return x_train, x_test, y_train, y_test