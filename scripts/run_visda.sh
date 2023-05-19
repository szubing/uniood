#!/bin/bash

METHODS=(
         "SO"
         "DANCE"
         "OVANet"
         "UniOT")
DATASET="visda"
sdomain="syn"
tdomain="real"
SEED=(1 2 3)
BACKBONE="resnet50"

for method in ${METHODS[@]}
do
    for seed in ${SEED[@]}
    do
        python main.py \
        --dataset ${DATASET} \
        --source_domain ${sdomain} \
        --target_domain ${tdomain} \
        --n_share 6 \
        --n_source_private 3 \
        --max_iter 10000 \
        --seed ${seed} \
        --method ${method} \
        --backbone ${BACKBONE}
    done
done