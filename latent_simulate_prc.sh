#!/bin/bash

# Define an array of FPR values
FPR_VALUES=(0.1 0.01 0.001 0.0001 0.00001)

# Define an array of attack types
ATTACK_TYPES=(stealthy min_distortion white_noise)

# Iterate over each FPR value and attack type
for FPR in "${FPR_VALUES[@]}"; do
    for ATTACK in "${ATTACK_TYPES[@]}"; do
        echo "Running with FPR=$FPR and ATTACK=$ATTACK"
        python3 latent_space_encode_decode.py \
            --method prc \
            --test_num 20 \
            --eps_low 1 \
            --eps_high 200 \
            --n_eps 100 \
            --attack "$ATTACK" \
            --fpr "$FPR"
    done
done
