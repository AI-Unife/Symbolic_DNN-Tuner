#!/bin/bash
# ------------------------------------------------------------------
# [Author] Robert Alexander Limas Sierra
#          Basic scripting code for training several IA models
# ------------------

models=(\
    "lenet"\
    "alexnet"\
    "vgg11"\
    "vgg19"\
    "res18"\
    "res34"\
    "res50"\
    "res101"\
    "convmixer"\
    "mlpmixer"\
    "vit_small"\
    "vit_tiny"\
    "simplevit"\
    "vit"\
    "cait"\
    "cait_small"\
    "swin"\
    )

for model in "${models[@]}";
do
    python train.py --net $model --resume --n_epochs 350
done
