#!/bin/bash

METHODS=(
         "SO"
         "DANCE"
         "OVANet"
         "UniOT")
DATASET="domainnet"
DOMAINS=("painting"
         "real" 
         "sketch")
SEED=(1 2 3)
BACKBONE="resnet50"

for method in ${METHODS[@]}
do
    for sdomain in ${DOMAINS[@]}
    do
        for tdomain in ${DOMAINS[@]}
        do
            if [ $sdomain != $tdomain ]; then
                for seed in ${SEED[@]}
                do
                    python main.py \
                    --dataset ${DATASET} \
                    --source_domain ${sdomain} \
                    --target_domain ${tdomain} \
                    --n_share 150 \
                    --n_source_private 50 \
                    --max_iter 10000 \
                    --seed ${seed} \
                    --method ${method} \
                    --backbone ${BACKBONE}
                done
            fi
        done
    done
done