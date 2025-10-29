##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, torch
import os.path as osp
import numpy as np
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100

from .DownsampledImageNet import load_imagenet16
# from .gesture_dataset import gesture_data
# import myconfig as cfg

Dataset2Class = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet1ks": 1000,
    "imagenet1k": 1000,
    "imagenet16": 1000,
    "imagenet16150": 150,
    "imagenet16120": 120,
    "imagenet16200": 200,
    "gesture": 11,
    "roigesture": 11
}

# imagenet root
root = "./ImageNet16"


def get_datasets(name):
    num_classes = Dataset2Class[name]

    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # if cfg.NAME_EXP == "test":
        #     (x_train, y_train) = (x_test, y_test)
        # Convert class vectors to binary class matrices.
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    elif name == "cifar100":
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        # Convert class vectors to binary class matrices.
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # elif name == "gesture":
    #     x_train, y_train, x_test, y_test = gesture_data(num_classes, ROI=False)

    # elif name == "roigesture":
    #     x_train, y_train, x_test, y_test = gesture_data(num_classes, ROI=True)
    elif name == "imagenet16":
        x_train, y_train, x_test, y_test = load_imagenet16(root)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        assert len(x_train) == 1281167 and len(x_test) == 50000
    elif name == "imagenet16120":
        x_train, y_train, x_test, y_test = load_imagenet16(root, 120)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        assert len(x_train) == 151700 and len(x_test) == 6000
    elif name == "imagenet16150":
        x_train, y_train, x_test, y_test = load_imagenet16(root, 150)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        assert len(x_train) == 190272 and len(x_test) == 7500
    elif name == "imagenet16200":
        x_train, y_train, x_test, y_test = load_imagenet16(root, 200)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        assert len(x_train) == 254775 and len(x_test) == 10000
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    print("shape of x_train:", x_train.shape)
    print("shape of y_train:", y_train.shape)
    print("shape of x_test:", x_test.shape)
    print("shape of y_test:", y_test.shape)
    print("num_classes:", num_classes)
    return x_train, y_train, x_test, y_test, num_classes
