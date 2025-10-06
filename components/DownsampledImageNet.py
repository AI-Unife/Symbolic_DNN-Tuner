# imagenet16_tf.py
import os, sys, hashlib, pickle
import numpy as np

try:
    import tensorflow as tf  # opzionale: solo per type/dtypes convenienti
except Exception:
    tf = None


# -------- util md5 --------
def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def check_md5(fpath, md5):
    return md5 == calculate_md5(fpath)

def check_integrity(root, file_list_with_md5):
    for filename, md5 in file_list_with_md5:
        fpath = os.path.join(root, filename)
        if not (os.path.isfile(fpath) and check_md5(fpath, md5)):
            return False
    return True


# -------- dataset spec (come nell'originale) --------
TRAIN_LIST = [
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
VALID_LIST = [
    ["val_data", "3410e3017fdaefba8d5073aaa65e4bd6"],
]


def _load_batches(root, file_list):
    """Legge e concatena i batch picklati in NumPy arrays HWC."""
    data_chunks, labels = [], []
    for file_name, _ in file_list:
        path = os.path.join(root, file_name)
        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        # entry["data"]: (N, 3*16*16) in ordine CHW; entry["labels"]: 1..K
        data_chunks.append(entry["data"])
        labels.extend(entry["labels"])

    X = np.vstack(data_chunks).reshape(-1, 3, 16, 16)           # N, C, H, W
    X = np.transpose(X, (0, 2, 3, 1))                           # -> N, H, W, C
    y = np.asarray(labels, dtype=np.int64) - 1                  # zero-based
    return X, y


def _filter_classes(X, y, use_num_of_class_only):
    """Mantiene solo le classi [0, use_num_of_class_only-1]."""
    if use_num_of_class_only is None:
        return X, y
    assert isinstance(use_num_of_class_only, int) and 0 < use_num_of_class_only < 1000
    mask = (y >= 0) & (y < use_num_of_class_only)
    return X[mask], y[mask]


def load_imagenet16(
    root: str,
    use_num_of_class_only: int = None,
    normalize: bool = False,
    dtype="float32",
):
    """
    Carica ImageNet-16-120 dai file picklati e restituisce:
        x_train, y_train, x_test, y_test

    Parametri:
      - root: cartella che contiene i file 'train_data_batch_*' e 'val_data'
      - use_num_of_class_only: se impostato, filtra alle prime K classi (0..K-1)
      - normalize: se True, scala a [0,1] e applica la normalizzazione 'ufficiale'
      - dtype: dtype finale delle immagini ('float32' consigliato se normalize=True)

    Ritorna:
      x_train: (Ntr, 16, 16, 3)
      y_train: (Ntr,)
      x_test : (Nte, 16, 16, 3)
      y_test : (Nte,)
    """
    # integrità
    if not check_integrity(root, TRAIN_LIST + VALID_LIST):
        raise RuntimeError("Dataset not found or corrupted (MD5 mismatch).")

    # carica split
    x_train, y_train = _load_batches(root, TRAIN_LIST)
    x_test,  y_test  = _load_batches(root, VALID_LIST)

    # filtra classi opzionale (usa etichette zero-based)
    x_train, y_train = _filter_classes(x_train, y_train, use_num_of_class_only)
    x_test,  y_test  = _filter_classes(x_test,  y_test,  use_num_of_class_only)

    # tipo & normalizzazione
    if normalize:
        # converti a float e scala a [0,1]
        x_train = x_train.astype(dtype) / 255.0
        x_test  = x_test.astype(dtype)  / 255.0

        # normalizzazione usata frequentemente per ImageNet16 (NAS-Bench-201)
        mean = np.array([122.68, 116.66, 104.01]) / 255.0
        std  = np.array([63.22,  61.26,  65.09 ]) / 255.0
        x_train = (x_train - mean) / std
        x_test  = (x_test  - mean) / std
    else:
        # lascia uint8 se non normalizzi
        x_train = x_train.astype(np.uint8)
        x_test  = x_test.astype(np.uint8)

    y_train = y_train.astype(np.int64)
    y_test  = y_test.astype(np.int64)

    return x_train, y_train, x_test, y_test