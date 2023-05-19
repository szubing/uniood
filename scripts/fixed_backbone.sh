#!/bin/bash

METHODS=(
         "SO"
         "DANCE"
         "OVANet"
         "UniOT"
)
SEED=(1 2 3)
BACKBONE=(
    # "RN50"
    # "ViT-B/16"
    # "ViT-L/14"
    # "dinov2_vitb14"
    "dinov2_vitl14"
    "ViT-L/14@336px"
)

# visda
DATASET="visda"
sdomain="syn"
tdomain="real"

for method in ${METHODS[@]}
do
    for backbone in ${BACKBONE[@]}
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
            --backbone ${backbone} \
            --fixed_backbone \
            --save_checkpoint
        done
    done
done

# office31
DATASET="office31"
DOMAINS=("amazon"
         "dslr" 
         "webcam")

for method in ${METHODS[@]}
do
    for backbone in ${BACKBONE[@]}
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
                        --max_iter 5000 \
                        --seed ${seed} \
                        --method ${method} \
                        --backbone ${backbone} \
                        --fixed_backbone \
                        --save_checkpoint
                    done
                fi
            done
        done
    done
done

# officehome
DATASET="officehome"
DOMAINS=("Art"
         "Clipart" 
         "Product"
         "RealWorld")

for method in ${METHODS[@]}
do
    for backbone in ${BACKBONE[@]}
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
                        --n_source_private 5 \
                        --max_iter 5000 \
                        --seed ${seed} \
                        --method ${method} \
                        --backbone ${backbone} \
                        --fixed_backbone \
                        --save_checkpoint
                    done
                fi
            done
        done
    done
done

# domainnet
DATASET="domainnet"
DOMAINS=("painting"
         "real" 
         "sketch")

for method in ${METHODS[@]}
do
    for backbone in ${BACKBONE[@]}
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
                        --backbone ${backbone} \
                        --fixed_backbone \
                        --save_checkpoint
                    done
                fi
            done
        done
    done
done