#!/bin/bash

METHODS=(
         "SO"
         "DANCE"
         "OVANet"
         "UniOT"
         )
DATASET="office31"
DOMAINS=("amazon"
         "dslr" 
         "webcam")
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
                    --n_share 10 \
                    --n_source_private 10 \
                    --max_iter 10000 \
                    --seed ${seed} \
                    --method ${method} \
                    --backbone ${BACKBONE}
                done
            fi
        done
    done
done