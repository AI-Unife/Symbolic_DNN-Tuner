#!/bin/bash

DATASET_DIR="./coco_data/val2017"
ANNOTATIONS_DIR="./coco_data/annotations"

if [ -d $DATASET_DIR ]; then
    echo "$DATASET_DIR already exists, skipping download."
else
    wget -c "http://images.cocodataset.org/zips/val2017.zip" -P ./coco_data
fi

if [ -d $ANNOTATIONS_DIR ]; then
    echo "$ANNOTATIONS_DIR already exists, skipping download."
else
    wget -c "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -P ./coco_data
fi

if [ "$(which unzip)" == "" ]; then
    echo "unzip not found, unable to extract the downloaded files. Please proceed manually."
    echo "If you wish to install the unzip utility, run 'sudo apt install unzip'."
    exit
else
    unzip ./coco_data/\*.zip -d ./coco_data
    rm ./coco_data/*.zip
fi