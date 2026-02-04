#!/bin/bash
# ------------------------------------------------------------------
# [Author] Robert Alexander Limas Sierra
#          Basic scripting code for training several IA models
# ------------------

models=(\
    # "lenet"\
    # "alexnet"\
    # "vgg11"\
    "vit"\
    )

for model in "${models[@]}";
do
    python train.py --model $model --epochs 50 --batchsize 64
done
