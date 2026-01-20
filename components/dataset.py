##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from tensorflow.keras.datasets import cifar10, cifar100, mnist
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO


Dataset2Class = {
    "cifar10": 10,
    "cifar100": 100,
    "light": 2,
    "tinyimagenet": 1000,
}

def filter_zeros_ones(x, y):
    keep = (y == 0) | (y == 1)
    x = x[keep]
    y = y[keep]
    return x, y

def get_datasets(name):
    num_classes = Dataset2Class[name]

    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Convert class vectors to binary class matrices.
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    elif name == "light":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize the pixel values to be between 0 and 1
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        x_train, y_train = filter_zeros_ones(x_train, y_train)
        x_test, y_test = filter_zeros_ones(x_test, y_test)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    elif name == "cifar100":
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        # Convert class vectors to binary class matrices.
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    print("shape of x_train:", x_train.shape)
    print("shape of y_train:", y_train.shape)
    print("shape of x_test:", x_test.shape)
    print("shape of y_test:", y_test.shape)
    print("num_classes:", num_classes)
    return x_train, y_train, x_test, y_test, num_classes
