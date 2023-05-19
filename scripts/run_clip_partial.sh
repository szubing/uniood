#!/bin/bash

METHODS=(
        "ClipZeroShot"
        "ClipCrossModel"
        "SO"
        "DANCE"
        "OVANet"
        "UniOT"
        "WiSE-FT"
        "ClipDistill"
)
SEED=(1 2 3)
BACKBONE=(
    # "RN50"
    # "ViT-B/16"
    # "ViT-L/14"
    "ViT-L/14@336px"
)

# visda
DATASET="visda"
sdomain="syn"
tdomain="real"

for method in ${METHODS[@]}
do
    if [ $method == "ClipZeroShot" ]; then
        SEED=(1); else
        SEED=(1 2 3)
    fi
    for backbone in ${BACKBONE[@]}
    do
        for seed in ${SEED[@]}
        do
            python main.py \
            --dataset ${DATASET} \
            --source_domain ${sdomain} \
            --target_domain ${tdomain} \
            --n_share 6 \
            --n_source_private 6 \
            --max_iter 20000 \
            --seed ${seed} \
            --method ${method} \
            --backbone ${backbone} \
            --fixed_backbone \
            --save_checkpoint \
            --num_workers 0
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
    if [ $method == "ClipZeroShot" ]; then
        SEED=(1); else
        SEED=(1 2 3)
    fi
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
                        --n_source_private 21 \
                        --max_iter 10000 \
                        --seed ${seed} \
                        --method ${method} \
                        --backbone ${backbone} \
                        --fixed_backbone \
                        --save_checkpoint \
                        --num_workers 0
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
    if [ $method == "ClipZeroShot" ]; then
        SEED=(1); else
        SEED=(1 2 3)
    fi
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
                        --n_share 25 \
                        --n_source_private 40 \
                        --max_iter 10000 \
                        --seed ${seed} \
                        --method ${method} \
                        --backbone ${backbone} \
                        --fixed_backbone \
                        --save_checkpoint \
                        --num_workers 0
                    done
                fi
            done
        done
    done
done

# # domainnet is not need for cloased and partial DA because of the lack of samples in some classes
# DATASET="domainnet"
# DOMAINS=("painting"
#          "real" 
#          "sketch")

# for method in ${METHODS[@]}
# do
#     if [ $method == "ClipZeroShot" ]; then
#         SEED=(1); else
#         SEED=(1 2 3)
#     fi
#     for backbone in ${BACKBONE[@]}
#     do
#         for sdomain in ${DOMAINS[@]}
#         do
#             for tdomain in ${DOMAINS[@]}
#             do
#                 if [ $sdomain != $tdomain ]; then
#                     for seed in ${SEED[@]}
#                     do
#                         python main.py \
#                         --dataset ${DATASET} \
#                         --source_domain ${sdomain} \
#                         --target_domain ${tdomain} \
#                         --n_share 150 \
#                         --n_source_private 0 \
#                         --max_iter 10000 \
#                         --seed ${seed} \
#                         --method ${method} \
#                         --backbone ${backbone} \
#                         --fixed_backbone \
#                         --save_checkpoint \
#                         --num_workers 0
#                     done
#                 fi
#             done
#         done
#     done
# done