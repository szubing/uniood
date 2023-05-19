#!/bin/bash

BACKBONES=(
        #  "RN50"
        #  "ViT-B/16"
        #  "ViT-L/14"
        #  "dinov2_vitb14"
         "dinov2_vitl14"
         "ViT-L/14@336px"
         )

DATASETS=("office31"
        "officehome"
        "visda"
        "domainnet")

for backbone in ${BACKBONES[@]}
do
    for dataset in ${DATASETS[@]}
    do
        python feature.py \
        --dataset ${dataset} \
        --backbone ${backbone}
    done
done