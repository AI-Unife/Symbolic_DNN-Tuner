from __future__ import annotations

import numpy as np
from datasets import load_dataset

def get_balanced_subset(x, y, n_per_class=500):
    """
    Obtain a balanced subset of the dataset.

    """
    x_list = []
    y_list = []

    classes = np.unique(y)

    for c in classes:
        mask = (y == c)
        mask = mask.reshape(-1)

        x_c = x[mask]

        y_c = y[mask]

        x_c = x_c[:n_per_class]
        y_c = y_c[:n_per_class]

        x_list.append(x_c)
        y_list.append(y_c)

    new_x = np.concatenate(x_list, axis=0)
    new_y = np.concatenate(y_list, axis=0)

    perm = np.random.permutation(len(new_x))
    return new_x[perm], new_y[perm]

class TunerDataset:

    X_train: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray
    Y_test: np.ndarray
    n_classes: int
    
    def data_as_float32(self):
        self.X_train = self.X_train.astype(np.float32)
        self.X_test  = self.X_test.astype(np.float32)

    def normalize_data(self):
        self.X_train = self.X_train.astype(np.float32) / 255.0
        self.X_test  = self.X_test.astype(np.float32) / 255.0

    def load_light_cifar(self):
        self.n_classes = 10
        self.load_hf_dataset("cifar10", image_key="img", label_key="label")

        self.X_train, self.Y_train = get_balanced_subset(self.X_train, self.Y_train, n_per_class=100)
        self.X_test, self.Y_test = get_balanced_subset(self.X_test, self.Y_test, n_per_class=100)


    def load_custom_dataset(self, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

    def load_hf_dataset(self, name: str, image_key: str, label_key: str):
        dataset = load_dataset(name)
        train = dataset["train"].with_format("numpy")
        test = dataset["test"].with_format("numpy")

        self.X_train = np.asarray(train[image_key])
        self.Y_train = np.asarray(train[label_key])
        self.X_test = np.asarray(test[image_key])
        self.Y_test = np.asarray(test[label_key])

        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

    def load_cifar_10(self):
        self.n_classes = 10
        self.load_hf_dataset("cifar10", image_key="img", label_key="label")
        self.normalize_data()

    def load_cifar_100(self):
        self.n_classes = 100
        self.load_hf_dataset("cifar100", image_key="img", label_key="label")
        self.normalize_data()

    def load_mnist(self):
        self.n_classes = 10
        self.load_hf_dataset(
            "mnist",
            image_key="image",
            label_key="label",
        )
        self.normalize_data()
        
    def load_tiny_imagenet(self):
        import pandas as pd
        from PIL import Image
        from io import BytesIO
        
        splits = {
            'train': 'data/train-00000-of-00001-1359597a978bc4fa.parquet',
            'valid': 'data/valid-00000-of-00001-70d52db3c749a935.parquet'
        }

        # Carica i parquet dal dataset Hugging Face
        df_train = pd.read_parquet("hf://datasets/zh-plus/tiny-imagenet/" + splits["train"])
        df_test = pd.read_parquet("hf://datasets/zh-plus/tiny-imagenet/" + splits["valid"])

        # Helper per estrarre l'immagine indipendentemente dal formato della colonna
        def open_image_from_row(img_field):
            # img_field può essere bytes/bytearray o un dict con chiave "bytes"
            if isinstance(img_field, (bytes, bytearray)):
                data = img_field
            elif isinstance(img_field, dict) and "bytes" in img_field:
                data = img_field["bytes"]
            else:
                # In alcuni dataset l'immagine è già un oggetto PIL (raro con parquet)
                # oppure un path (non nel tuo caso). Gestiamo anche questi.
                if isinstance(img_field, Image.Image):
                    return img_field.convert("RGB")
                raise TypeError(f"Formato immagine non riconosciuto: {type(img_field)}")
            img = Image.open(BytesIO(data)).convert("RGB")
            # Tiny-ImageNet è 64x64; assicuriamolo nel caso sia necessario
            if img.size != (64, 64):
                img = img.resize((64, 64))
            return img

        # Carica in liste (più veloce di np.concatenate in loop)
        x_train_list, y_train_list = [], []
        for _, row in df_train.iterrows():
            img = open_image_from_row(row["image"])
            x_train_list.append(np.array(img))  # (64, 64, 3), dtype uint8
            y_train_list.append(int(row["label"]))

        x_test_list, y_test_list = [], []
        for _, row in df_test.iterrows():
            img = open_image_from_row(row["image"])
            x_test_list.append(np.array(img))
            y_test_list.append(int(row["label"]))

        # Converte in array; opzionale: normalizzazione in [0,1]
        x_train = np.stack(x_train_list)
        x_test = np.stack(x_test_list)
        y_train = np.array(y_train_list, dtype=np.int64)
        y_test = np.array(y_test_list, dtype=np.int64)

        self.X_train = x_train
        self.X_test = x_test
        self.Y_train = y_train
        self.Y_test = y_test
        self.normalize_data()

    def load_gesture(self):
        """Load DVSGesture dataset using the specialized gesture_dataset module."""
        from components.gesture_dataset import gesture_data
        
        self.n_classes = 11
        self.X_train, self.Y_train, self.X_test, self.Y_test = gesture_data(num_classes=11, ROI=False)
        
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')
    
    def load_roi_gesture(self):
        """Load DVSGesture dataset using the specialized gesture_dataset module, with ROI."""
        from components.gesture_dataset import gesture_data
        
        self.n_classes = 11
        X_train_raw, self.Y_train, X_test_raw, self.Y_test = gesture_data(num_classes=11, ROI=True)
        if isinstance(X_test_raw[0], dict):
            self.X_train = np.array([item["data"] for item in X_train_raw]).astype("float32")
            self.pos_train = np.array([item["pos"] for item in X_train_raw])
            self.X_test = np.array([item["data"] for item in X_test_raw]).astype("float32")
            self.pos_test = np.array([item["pos"] for item in X_test_raw])
            print(self.pos_train.shape, 'pos train samples')
            print(self.pos_test.shape, 'pos test pos samples')
        else: 
            self.X_train = np.array(X_train_raw).astype("float32")
            self.X_test = np.array(X_test_raw).astype("float32")
        
        print(self.X_train.shape, 'X train samples')
        print(self.Y_train.shape, 'Y train label samples')
        print(self.X_test.shape, 'X test samples')
        print(self.Y_test.shape, 'Y test label samples')