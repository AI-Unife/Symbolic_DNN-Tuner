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

    def load_cifar_100(self):
        self.n_classes = 100
        self.load_hf_dataset("cifar100", image_key="img", label_key="label")

    def load_mnist(self):
        self.n_classes = 10
        self.load_hf_dataset(
            "mnist",
            image_key="image",
            label_key="label",
        )
    def load_gesture(self):
        """Load DVSGesture dataset using the specialized gesture_dataset module."""
        from components.gesture_dataset import gesture_data
        
        self.n_classes = 11
        self.X_train, self.Y_train, self.X_test, self.Y_test = gesture_data(num_classes=11, ROI=False)
        
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')