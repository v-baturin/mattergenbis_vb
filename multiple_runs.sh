#!/bin/bash

NB=50
DIR="results/Li-Co-O_guided_3_env_3-2_"

"" >> log1.txt

for X in {1..20}; do
    echo "Generating $NB samples for Li-Co-O"
    start_time=$(date +%s)
    mattergen-generate "$DIR${X}" \
        --pretrained-name=chemical_system \
        --batch_size=$NB \
        --properties_to_condition_on="{'chemical_system':'Li-Co-O'}" \
        --record_trajectories=False \
        --diffusion_guidance_factor=2.0 \
        --guidance="{'environment': {'Co-O':3}}" \
        --diffusion_loss_weight=1.0 \
        --print_loss=False \
        --self_rec_steps=3 \
        --back_step=2 >> log1.txt 2>&1
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Generated samples for Li-Co-O with environment Co-O:3 at step $X"
    echo "Duration: ${duration} seconds at $(date +%H:%M:%S)"
done

main_file="/Data/auguste.de-lambilly/mattergenbis/${DIR}1/generated_crystals.extxyz"

for X in {2..20}; do
    src="/Data/auguste.de-lambilly/mattergenbis/results/${DIR}${X}/generated_crystals.extxyz"
    if [ -f "$src" ]; then
        cat "$src" >> "$main_file"
    else
        echo "Warning: $src does not exist, skipping."
    fi
done