#!/bin/bash

for X in {4..10}; do
    mattergen-generate "results/Li-Co-O_guided_env_3-2_${X}" \
        --pretrained-name=chemical_system \
        --batch_size=50 \
        --properties_to_condition_on="{'chemical_system':'Li-Co-O'}" \
        --record_trajectories=False \
        --diffusion_guidance_factor=2.0 \
        --guidance="{'environment': {'Co-O':6}}" \
        --diffusion_loss_weight=1.0 \
        --print_loss=False \
        --self_rec_steps=3 \
        --back_step=2 >> log1.txt 2>&1
    echo "Generated samples for Li-Co-O with environment Co-O:6 at step $X"
done

main_file="/Data/auguste.de-lambilly/mattergenbis/results/Li-Co-O_guided_env_3-2/generated_crystals.extxyz"

for X in {2..10}; do
    src="/Data/auguste.de-lambilly/mattergenbis/results/Li-Co-O_guided_env_3-2_${X}/generated_crystals.extxyz"
    if [ -f "$src" ]; then
        cat "$src" >> "$main_file"
    else
        echo "Warning: $src does not exist, skipping."
    fi
done