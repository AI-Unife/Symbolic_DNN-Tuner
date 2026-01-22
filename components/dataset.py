from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100

from typing import Any

class TunerDataset:

    X_train: Any
    X_test: Any
    Y_train: Any
    Y_test: Any
    n_classes: int

    def cifar_data(self):
        self.n_classes = 10
        # The data, split between train and test sets:
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = cifar10.load_data()
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

        # Convert class vectors to binary class matrices.
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train, self.n_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test, self.n_classes)

    def cifar_data_100(self):
        self.n_classes = 100
        # The data, split between train and test sets:
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = cifar100.load_data()
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

        # Convert class vectors to binary class matrices.
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train, self.n_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test, self.n_classes)

        return self.X_train, self.X_test, self.Y_train, self.Y_test, self.n_classes

    def mnist(self):
        mnist = tf.keras.datasets.mnist
        self.n_classes = 10
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data()

        print('X_train shape:', self.X_train.shape)
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train, self.n_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test, self.n_classes)

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1)

        return self.X_train, self.X_test, self.Y_train, self.Y_test, self.n_classes
