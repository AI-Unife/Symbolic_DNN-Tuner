from __future__ import annotations

import os
import sys
import hashlib
import pickle
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf  # optional: convenient dtypes
except Exception:  # pragma: no cover
    tf = None


# ---------------------------- MD5 utilities ----------------------------------

def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute MD5 over a file in streaming chunks."""
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str) -> bool:
    """Return True if `fpath` exists and matches the expected MD5."""
    return os.path.isfile(fpath) and (md5 == calculate_md5(fpath))


def check_integrity(root: str, file_list_with_md5: Sequence[Sequence[str]]) -> bool:
    """Verify that all files exist and pass MD5 checks."""
    for filename, md5 in file_list_with_md5:
        fpath = os.path.join(root, filename)
        if not (os.path.isfile(fpath) and check_md5(fpath, md5)):
            return False
    return True


# ---------------------------- Dataset spec -----------------------------------

TRAIN_LIST: List[List[str]] = [
    ["train_data_batch_1",  "27846dcaa50de8e21a7d1a35f30f0e91"],
    ["train_data_batch_2",  "c7254a054e0e795c69120a5727050e3f"],
    ["train_data_batch_3",  "4333d3df2e5ffb114b05d2ffc19b1e87"],
    ["train_data_batch_4",  "1620cdf193304f4a92677b695d70d10f"],
    ["train_data_batch_5",  "348b3c2fdbb3940c4e9e834affd3b18d"],
    ["train_data_batch_6",  "6e765307c242a1b3d7d5ef9139b48945"],
    ["train_data_batch_7",  "564926d8cbf8fc4818ba23d2faac7564"],
    ["train_data_batch_8",  "f4755871f718ccb653440b9dd0ebac66"],
    ["train_data_batch_9",  "bb6dd660c38c58552125b1a92f86b5d4"],
    ["train_data_batch_10", "8f03f34ac4b42271a294f91bf480f29b"],
]
VALID_LIST: List[List[str]] = [
    ["val_data", "3410e3017fdaefba8d5073aaa65e4bd6"],
]


# ----------------------------- I/O helpers -----------------------------------

def _load_batches(root: str, file_list: Sequence[Sequence[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read and concatenate pickled batches into NumPy arrays (HWC layout).

    Assumes each pickle contains:
      - entry["data"]: shape (N, 3*16*16) in CHW order
      - entry["labels"]: list/array with class ids 1..K (will be shifted to 0..K-1)
    """
    data_chunks: List[np.ndarray] = []
    labels: List[int] = []

    for file_name, _ in file_list:
        path = os.path.join(root, file_name)
        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")

        # Defensive checks
        if "data" not in entry or "labels" not in entry:
            raise ValueError(f"Corrupted batch file (missing keys) at: {path}")

        arr = np.asarray(entry["data"])
        if arr.ndim != 2 or arr.shape[1] != 3 * 16 * 16:
            raise ValueError(f"Unexpected data shape {arr.shape} in {path}; expected (N, 768).")

        data_chunks.append(arr)
        labels.extend(entry["labels"])

    # (N, 3*16*16) -> (N, C, H, W) -> (N, H, W, C)
    X = np.vstack(data_chunks).reshape(-1, 3, 16, 16)
    X = np.transpose(X, (0, 2, 3, 1))
    y = np.asarray(labels, dtype=np.int64) - 1  # shift to zero-based
    return X, y


def _filter_classes(
    X: np.ndarray, y: np.ndarray, use_num_of_class_only: Optional[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only classes in [0, use_num_of_class_only-1] if requested."""
    if use_num_of_class_only is None:
        return X, y
    if not (isinstance(use_num_of_class_only, int) and 0 < use_num_of_class_only <= 1000):
        raise ValueError("`use_num_of_class_only` must be an int in (0, 1000].")
    mask = (y >= 0) & (y < use_num_of_class_only)
    return X[mask], y[mask]


# ------------------------------ Public API -----------------------------------

def load_imagenet16(
    root: str,
    use_num_of_class_only: int | None = None,
    normalize: bool = False,
    dtype: str = "float32",
    strict_md5: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ImageNet-16 (NAS-Bench-201 format) from pickled files and return:
        x_train, y_train, x_test, y_test

    Args:
        root: Directory containing 'train_data_batch_*' and 'val_data' files.
        use_num_of_class_only: If set, keep only the first K classes (0..K-1).
        normalize: If True, scale to [0,1] and apply standard mean/std normalization.
        dtype: Image dtype after optional normalization ('float32' recommended with normalize=True).
        strict_md5: If True (default), verify MD5 for all expected files and raise on mismatch.
                    If False, only check that files exist (skip MD5), useful for faster startup.

    Returns:
        x_train: (Ntr, 16, 16, 3)
        y_train: (Ntr,)
        x_test : (Nte, 16, 16, 3)
        y_test : (Nte,)
    """
    # Fallback to ./data/imagenet16 if `root` does not exist
    if not os.path.isdir(root):
        fallback = os.path.join(".", "data", "imagenet16")
        os.makedirs(fallback, exist_ok=True)
        print(f"[info] dataset root '{root}' not found. Falling back to: {os.path.abspath(fallback)}")
        root = fallback

    # Integrity (MD5 or existence)
    if strict_md5:
        ok = check_integrity(root, TRAIN_LIST + VALID_LIST)
    else:
        ok = all(os.path.isfile(os.path.join(root, name)) for name, _ in (TRAIN_LIST + VALID_LIST))

    if not ok:
        raise RuntimeError(
            "Dataset not found or corrupted.\n"
            f"Expected files under: {os.path.abspath(root)}\n"
            "If you intentionally skip checks, call load_imagenet16(..., strict_md5=False)."
        )

    # Load splits
    x_train, y_train = _load_batches(root, TRAIN_LIST)
    x_test,  y_test  = _load_batches(root, VALID_LIST)

    # Optional class filtering (labels are zero-based now)
    x_train, y_train = _filter_classes(x_train, y_train, use_num_of_class_only)
    x_test,  y_test  = _filter_classes(x_test,  y_test,  use_num_of_class_only)

    # Dtype & normalization
    if normalize:
        x_train = x_train.astype(dtype) / 255.0
        x_test  = x_test.astype(dtype) / 255.0

        # Common normalization for ImageNet16 / NAS-Bench-201
        mean = np.array([122.68, 116.66, 104.01]) / 255.0
        std  = np.array([63.22,  61.26,  65.09 ]) / 255.0
        x_train = (x_train - mean) / std
        x_test  = (x_test  - mean) / std
    else:
        # Keep compact uint8 if not normalizing
        x_train = x_train.astype(np.uint8)
        x_test  = x_test.astype(np.uint8)

    y_train = y_train.astype(np.int64)
    y_test  = y_test.astype(np.int64)
    return x_train, y_train, x_test, y_test


def show_samples(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    n: int = 9,
    class_names: Optional[List[str]] = None,
    cmap = None,
) -> None:
    """
    Show `n` random images from the dataset grid.

    Args:
        x: Images (N, H, W, C).
        y: Optional labels (N,).
        n: Number of images to display.
        class_names: Optional list of class names.
        cmap: Optional Matplotlib colormap (for grayscale visualization).
    """
    if len(x) == 0:
        print("[warn] empty image array; nothing to display.")
        return

    n = max(1, min(n, len(x)))
    idxs = np.random.choice(len(x), n, replace=False)

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2, rows * 2))

    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i + 1)
        img = x[idx]
        # Bring normalized images back to [0,1] for display (best-effort)
        if img.dtype != np.uint8:
            vmin, vmax = np.min(img), np.max(img)
            rng = vmax - vmin + 1e-8
            img = (img - vmin) / rng
        plt.imshow(img, cmap=cmap)
        if y is not None:
            label = int(y[idx])
            if class_names and 0 <= label < len(class_names):
                plt.title(class_names[label], fontsize=8)
            else:
                plt.title(f"class {label}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ------------------------------ Example usage --------------------------------

if __name__ == "__main__":
    root = "./ImageNet16"
    # strict_md5=True keeps the original safety; set False to skip MD5 for speed.
    x_train, y_train, x_test, y_test = load_imagenet16(
        root, use_num_of_class_only=120, normalize=False, strict_md5=True
    )
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    show_samples(x_train, y_train, n=12)