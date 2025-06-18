import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100

import numpy as np
import os


def cifar_data():
    num_classes = 10
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, num_classes


def cifar_data_100():
    num_classes = 100
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, num_classes


def mnist():
    mnist = tf.keras.datasets.mnist
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print('X_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return x_train, x_test, y_train, y_test, num_classes


def imagenet_data(data_dir, img_size=224):
    """
    Loads ImageNet from a directory with 'train/' and 'val/' folders.
    Outputs x_train, x_test, y_train, y_test, num_classes.
    All arrays are fully loaded into memory (may require >100GB RAM).
    """
    num_classes = 1000

    def load_images_from_folder(folder):
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            folder,
            labels="inferred",
            label_mode="int",  # will convert to one-hot later
            image_size=(img_size, img_size),
            batch_size=256,
            shuffle=False
        )
        x_list = []
        y_list = []
        for batch_x, batch_y in ds:
            x_list.append(batch_x.numpy())
            y_list.append(batch_y.numpy())
        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return x, y

    print("Loading training set...")
    x_train, y_train = load_images_from_folder(os.path.join(data_dir, "train"))
    print("Loading validation set...")
    x_test, y_test = load_images_from_folder(os.path.join(data_dir, "val"))

    print(f"{x_train.shape[0]} train samples")
    print(f"{x_test.shape[0]} test samples")

    # Normalize images
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Convert labels to one-hot
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, num_classes