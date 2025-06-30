#!/bin/bash


if [[ "$1" == "--help" ]]; then
    echo "Usage:"
    echo "  ./multiple_runs.sh NB LOG MUL BASE DIR G ALG"
    echo ""
    echo "Arguments:"
    echo "  NB   : Number of samples per run (default: 20)"
    echo "  LOG  : Log file name (default: log2.txt)"
    echo "  MUL  : Number of runs (default: 50)"
    echo "  BASE : Base directory for results (default: /Data/auguste.de-lambilly/mattergenbis/)"
    echo "  SYS  : System to generate "
    echo "  ENV  : Environment for the system (default: 3)"
    echo "  R    : Self-recursion steps (default: 3)"
    echo "  B    : Back step (default: 2)"
    echo "  G    : Diffusion loss weight (default: 1.0)"
    echo "  ALG  : Algorithm flag, True or False (default: True)"
    echo "  GPU  : GPU index to use (optional, default: None)"
    echo ""
    echo "Example:"
    echo "  ./multiple_runs.sh 20 log2.txt 50 /Data/auguste.de-lambilly/mattergenbis/ Li-Co-O 3 3 2 1.0 True"
    exit 0
fi

# Default values for parameters
NB=${1:-20}
LOG=${2:-log2.txt}
MUL=${3:-50}
BASE=${4:-/Data/auguste.de-lambilly/mattergenbis/}
SYS=${5:-Li-Co-O}
ENV=${6:-3}
G=${6:-1.0}
R=${6:-3}
B=${7:-2}
ALG=${8:-True}
GPU=${9:-None}

if [ "$ALG" == "True" ]; then
    al=2
    DIR="results/${SYS}_guided2__env${ENV}_" 
else
    al=1
    DIR="results/${SYS}_guided__env${ENV}_"
fi

if [ $G != 1.0 ]; then
    DIR=${DIR}"g${G}_"
fi

DIR=${DIR}"${R}-${B}_"

echo "" > $LOG

for X in $(seq 1 "$MUL"); do
    echo "Generating $NB samples for $SYS into ${DIR}${X} at $(date +%H:%M:%S)"
    start_time=$(date +%s)
    mattergen-generate "$DIR${X}" \
        --pretrained-name=chemical_system \
        --batch_size=$NB \
        --properties_to_condition_on="{'chemical_system':'${SYS}'}" \
        --record_trajectories=False \
        --diffusion_guidance_factor=2.0 \
        --guidance="{'environment': {'Co-O':$ENV}}" \
        --diffusion_loss_weight=$G \
        --print_loss=False \
        --self_rec_steps=$R \
        --back_step=$B \
        --algo=$ALG \
        --force_gpu=$GPU >> $LOG 2>&1
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Generated samples for Li-Co-O with environment Co-O:3 at step $X"
    echo "Duration: ${duration} seconds at $(date +%H:%M:%S)"
done

main_file="${BASE}/${DIR}1/generated_crystals.extxyz"

for X in $(seq 2 "$MUL"); do
    src="${BASE}/${DIR}${X}/generated_crystals.extxyz"
    if [ -f "$src" ]; then
        cat "$src" >> "$main_file"
    else
        echo "Warning: $src does not exist, skipping."
    fi
done

echo "Everything copied to $main_file"