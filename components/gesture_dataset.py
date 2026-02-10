from __future__ import annotations

from typing import Tuple, List, Callable, Optional
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tonic
import tonic.transforms as transforms
from tonic.dataset import Dataset 
from tonic.datasets import DVSGesture 
from sklearn.model_selection import train_test_split  # kept for compatibility if needed elsewhere

from exp_config import load_cfg


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
        raise ValueError(f"Invalid polarity {polarity} option! Choose from ['sum', 'sub', 'drop', 'both'].")
    return mapping[polarity]


# ------------------------------- Datasets ------------------------------------

class DVSGestureROI(DVSGesture):
    """A modified version of the DVSGesture eventset that returns regions-of-interest (ROIs) and their coordinates."""

    sensor_size = (32, 32, 2)
    dtype = np.dtype(
        [("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)]
    )
    dtype_position = np.dtype(
        [
            ("x", np.int16),
            ("y", np.int16),
            ("s", np.int16),
            ("p", bool),
            ("t", np.float32),
        ]
    )
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        position_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        Dataset.__init__(
            self,
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.position_transform = position_transform
        self.train = train

        if train:
            self.folder_name = "train"
        else:
            self.folder_name = "test"

        self.users = []
        self.lighting = []
        self.position = []
        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            rel_path = os.path.relpath(path, file_path)
            if rel_path != ".":
                user, lighting = rel_path.split("_", 1)
                user = int(user[4:])
                dirs.sort()
                for file in files:
                    if file.endswith(".npy"):
                        if file.endswith("_positions.npy"):
                            continue
                        self.data.append(os.path.join(path, file))
                        self.targets.append(int(file[:-4]))
                        pos_file = file.removesuffix(".npy") + (
                            "_positions.npy"
                        )
                        self.position.append(os.path.join(path, pos_file))
                        self.users.append(user)
                        self.lighting.append(lighting)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target, position) where target is the index of the target class and position is the position of the ROI in the input.
        """
        events = np.load(self.data[index])
        target = self.targets[index]
        position = np.load(self.position[index])
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.position_transform is not None:
            position = self.position_transform(position)
        if self.transforms is not None:
            events, target, position = self.transforms(
                events, target, position
            )
        x = {"data": events, "pos": position}
        return x, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (
            self._is_file_present()
            and self._folder_contains_at_least_n_files_of_type(100, ".npy")
        )


class ROIMapTransform(object):
    """Transform ROI position events into frames for downstream processing"""

    def __init__(
        self,
        full_input_size: tuple = (128, 128, 2),  # DVSGesture size
        output_size: tuple = (32, 32, 1),  # Target size for downstream
        time_window: int = None,
        n_time_bins: int = None,
    ):
        self.downsample_transform = transforms.Downsample(
            sensor_size=full_input_size[:2] + (1,), target_size=output_size[:2]
        )
        if time_window is not None:  
            self.frame_transform = transforms.ToFrame(
                sensor_size=output_size,
                time_window=time_window,
            )
        elif n_time_bins is not None:
            self.frame_transform = transforms.ToFrame(
                sensor_size=output_size,
                n_time_bins=n_time_bins,
            )
        else:
            self.frame_transform = transforms.ToFrame(
                sensor_size=output_size,
                time_window=1000,
                )
        self.transform = transforms.Compose(
            [self.downsample_transform, self.frame_transform]
        )

    def __call__(self, events):
        return self.transform(events)


# ----------------------------- Numpy adapters --------------------------------

def reshape_x_pos(arr: np.ndarray, pos: Optional[np.ndarray], cfg) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Reshape x and optionally pos arrays to match model input expectations.

    Args:
        arr: Input data array.
        pos: Optional position array (may be None for non-ROI datasets).
        cfg: Configuration object with mode and channels attributes.

    Returns:
        A tuple (x_reshaped, pos_reshaped_or_None).
    """
    if cfg.mode == "fwdPass":
        # [T, 2, H, W] -> [T, H, W, 2]
        x_out = np.transpose(arr, (0, 2, 3, 1))
        pos_out = np.transpose(pos, (0, 2, 3, 1)) if pos is not None else None
        return x_out, pos_out

    elif cfg.mode == "hybrid":
        # Expect BothPolarity OR a prior framing such that arr has shape [T, 2, H, W]
        # We reshape into groups of NUM_CHANNELS along time and stack polarities as channels.
        T, C, H, W = arr.shape  # C should be 2
        assert T % cfg.channels == 0, f"T={T} not divisible by NUM_CHANNELS={cfg.channels}"
        # Group frames: (T//NUM_CHANNELS, NUM_CHANNELS, C, H, W)
        grouped = arr.reshape(T // cfg.channels, cfg.channels, C, H, W)
        # Move to (T', H, W, NUM_CHANNELS, C)
        transposed = grouped.transpose(0, 3, 4, 1, 2)
        # Merge channel dims -> (T', H, W, NUM_CHANNELS*C) == (T', H, W, 2*NUM_CHANNELS)
        x_final = transposed.reshape(T // cfg.channels, H, W, cfg.channels * C)
        # If a pos map is provided and looks like frames, try a simple transpose to keep it aligned.
        pos_out = None
        if pos is not None and hasattr(pos, "ndim") and pos.ndim == 4:
            try:
                pos_out = np.transpose(pos, (0, 2, 3, 1))
            except Exception:
                pos_out = None
        return x_final, pos_out

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
        if pos is not None and hasattr(pos, "ndim") and pos.ndim == 4:
            if pos.ndim == 2:
                pos = pos[..., None]  # [H, W, 1]
            elif pos.ndim == 3 and pos.shape[0] not in (1, 2, 3, 4):  # likely [T, H, W]
                pos = np.transpose(pos, (1, 2, 0))
        return arr, pos

def dataset_to_numpy(dataset, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a (cached) tonic dataset into NumPy arrays (or object array for ROI) with the layout expected by the model.

    Output X layout depends on cfg.MODE:
      - "fwdPass":    input per time step with 2 channels -> [B, T, H, W, 2]
      - "hybrid":     merge groups of NUM_CHANNELS frames and both polarities -> [B, T', H, W, 2*NUM_CHANNELS]
      - "depth":      single frame with possibly combined polarity -> [B, H, W, C]
    For ROI datasets, each element of X is a dict {"data": <array>, "pos": <array>} and the returned X
    is an object array with shape (N,). For non-ROI, X is a numeric array.
    Targets (y):
      - "fwdPass"/"hybrid": [B, T, C] (time-repeated one-hot via ToOneHotTimeCoding)
      - "depth":            integer labels; one-hot added later in gesture_data()
    """
    x_list, y_list = [], []

    for x, y in dataset:
        # Handle ROI items (dicts with 'data' and 'pos')
        if isinstance(x, dict) or (hasattr(cfg, "dataset") and "roi" in cfg.dataset.lower()):
            events = x["data"]
            pos = x.get("pos", None)
            arr = np.array(events)
            x_reshaped, _ = reshape_x_pos(arr, np.array(pos) if pos is not None else None, cfg)
            # pos_mean = None
            # if pos is not None:
            #     # Example analysis: compute center of mass of ROI position map in first frame
            #     # A.shape = (16, 32, 32, 1)
            #     A = np.squeeze(pos, axis=1)  # -> (16, 32, 32)

            #     H, W = A.shape[1], A.shape[2]

            #     yy, xx = np.indices((H, W))  # yy, xx -> (32, 32)

            #     tot = A.sum(axis=(1, 2))  # (16,)

            #     mean_y = (A * yy).sum(axis=(1, 2)) / tot
            #     mean_x = (A * xx).sum(axis=(1, 2)) / tot

            #     pos_mean = np.stack([mean_y, mean_x], axis=1)  # (16, 2)

            x_list.append({"data": x_reshaped, "pos": pos})
        else:
            # Non-ROI: plain arrays
            arr = np.array(x)
            x_reshaped, _ = reshape_x_pos(arr, None, cfg)
            x_list.append(x_reshaped)

        y_list.append(np.array(y))

    # Return X as object array so that elements can be dicts (ROI) or arrays; y as numeric array
    return np.array(x_list, dtype=object), np.array(y_list)


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
        # cd = Path("./cache")
        cd.mkdir(parents=True, exist_ok=True)
        print(f"[info] cache_dir not found. Falling back to: {cd.resolve()}")

    return str(dp.resolve()), str(cd.resolve())

def _ensure_cache_dir(cache_dir: str) -> None:
    Path(cache_dir, "train").mkdir(parents=True, exist_ok=True)
    Path(cache_dir, "test").mkdir(parents=True, exist_ok=True)

def get_datasets_numpy(cfg):
    """
    Load DVSGesture through Tonic, apply transforms, and return NumPy arrays.

    Returns:
        ((x_train, y_train), (x_test, y_test))
    """
    dataset_path = './data/'
    polarity = cfg.polarity
    n_pol = 2 if polarity == "both" else 1
    cache_dir = f"./cache/DVSGesture_{cfg.mode}_{polarity}_{cfg.frames}_{cfg.channels}_{n_pol}/"
    # resolve with fallback
    dataset_path, cache_dir = _resolve_paths(dataset_path, cache_dir)
    _ensure_cache_dir(cache_dir)
    print("dataset_path:", dataset_path)
    print("cache_dir:", cache_dir)

    # Base framing to [T, 2, H, W] with H=W=64
    tfms: List = [
        transforms.Denoise(filter_time=10000),
        transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(64, 64)),
        transforms.ToFrame(sensor_size=(64, 64, 2), n_time_bins=cfg.frames),
    ]

    # Labels repeated across time if the model expects temporal supervision
    if cfg.mode == "fwdPass":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.frames)
    elif cfg.mode == "hybrid":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.frames // cfg.channels)
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

    x_train, y_train = dataset_to_numpy(cached_train, cfg)
    x_test, y_test = dataset_to_numpy(cached_test, cfg)

    return (x_train, y_train), (x_test, y_test)


def get_ROI_numpy(cfg):
    """
    Load a folder-structured ROI dataset via ROIDataset and return NumPy arrays.

    Returns:
        ((x_train, y_train), (x_test, y_test))
    """
    dataset_path = "rois_and_coordinates/datasets"
    polarity = cfg.polarity
    n_pol = 2 if polarity == "both" else 1
    cache_dir = f"./cache/DVS_ROI_{cfg.mode}_{cfg.frames}_{cfg.channels}/"
    # cache_dir = f"cache/DVS_ROI_{cfg.mode}_{polarity}_{cfg.frames}_{cfg.channels}_{n_pol}/"
    _ensure_cache_dir(cache_dir)
    print("cache_dir:", cache_dir)

    tfms: List = [
        transforms.Denoise(filter_time=10000),
        # Reuse DVSGesture sensor size only as a (H,W) hint for downsampling,
        # since we target a fixed (64, 64) output anyway.
        # transforms.Downsample(sensor_size=tonic.datasets.DVSGesture.sensor_size, target_size=(64, 64)),
        transforms.ToFrame(sensor_size=(32, 32, 2), n_time_bins=cfg.frames),
    ]

    if cfg.mode == "fwdPass":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.frames)
    elif cfg.mode == "hybrid":
        target_transform = ToOneHotTimeCoding(n_classes=11, n_frames=cfg.frames // cfg.channels)
    else:
        tfms.append(select_polarity_transform(polarity))
        target_transform = None

    transform = transforms.Compose(tfms)

    train =  DVSGestureROI(
        dataset_path,
        train=True,
        transform=transform,
        target_transform=target_transform,
        position_transform=ROIMapTransform(n_time_bins=cfg.frames),
    )
    print("Loaded ROI training dataset with", len(train), "samples.")
    # test = DVSGestureROI(
    #     dataset_path,
    #     train=False,
    #     transform=transform,
    #     target_transform=target_transform,
    #     position_transform=ROIMapTransform(n_time_bins=cfg.frames),
    # )

    cached_train = tonic.DiskCachedDataset(train, cache_path=os.path.join(cache_dir, "train"))
    # cached_test = tonic.DiskCachedDataset(test, cache_path=os.path.join(cache_dir, "test"))

    x, y = dataset_to_numpy(cached_train, cfg) 
    # Handle multi-dimensional y (e.g., [B, T, C] one-hot): use first time step to determine class
    if y.ndim > 1:
        y_for_split = np.argmax(y[:, 0, :], axis=1) if y.ndim == 3 else np.argmax(y, axis=1)
    else:
        y_for_split = y
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y_for_split
    )
    # x_train, y_train = dataset_to_numpy(cached_train, cfg)
    # x_test, y_test = dataset_to_numpy(cached_test, cfg)

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
    cfg = load_cfg()
    if ROI:
        (x_train, y_train), (x_test, y_test) = get_ROI_numpy(cfg=cfg)
    else:
        (x_train, y_train), (x_test, y_test) = get_datasets_numpy(cfg=cfg)

    # Convert labels to one-hot for single-frame modes
    if cfg.mode == "depth":
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)


    return x_train, y_train, x_test, y_test