##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from tensorflow.keras.datasets import cifar10, cifar100
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

from .DownsampledImageNet import load_imagenet16
from .gesture_dataset import gesture_data

Dataset2Class = {
    "cifar10": 10,
    "cifar100": 100,
    "tinyimagenet": 1000,
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
        # Convert class vectors to binary class matrices.
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    elif name == "cifar100":
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        # Convert class vectors to binary class matrices.
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    elif name == "gesture":
        x_train, y_train, x_test, y_test = gesture_data(num_classes, ROI=False)

    elif name == "roigesture":
        x_train, y_train, x_test, y_test = gesture_data(num_classes, ROI=True)
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
    elif name == "tinyimagenet":
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
        x_train = np.stack(x_train_list).astype("float32") / 255.0
        x_test = np.stack(x_test_list).astype("float32") / 255.0
        y_train = np.array(y_train_list, dtype=np.int64)
        y_test = np.array(y_test_list, dtype=np.int64)

        # One-hot
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    print("shape of x_train:", x_train.shape)
    print("shape of y_train:", y_train.shape)
    print("shape of x_test:", x_test.shape)
    print("shape of y_test:", y_test.shape)
    print("num_classes:", num_classes)
    return x_train, y_train, x_test, y_test, num_classes
