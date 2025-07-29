#!/bin/bash


if [[ "$1" == "--help" ]]; then
    echo "Usage:"
    echo "  ./multiple_runs.sh NB LOG MUL BASE SYS ENV G R B ALG [GPU] [MOD]"
    echo ""
    echo "Arguments:"
    echo "  NB   : Number of samples per run (default: 20)"
    echo "  LOG  : Log file name (default: log2.txt)"
    echo "  MUL  : Number of runs (default: 50)"
    echo "  BASE : Base directory for results (default: /Data/auguste.de-lambilly/mattergenbis/)"
    echo "  SYS  : System to generate "
    echo "  ENV  : Environment conditions for the system (default: 'Co-O':3)"
    echo "  R    : Self-recursion steps (default: 3)"
    echo "  B    : Back step (default: 2)"
    echo "  G    : Forward Diffusion loss weight (default: 1.0)"
    echo "  K    : Backward Diffusion loss weight (default: 1.0)"
    echo "  Norm : Normalize the diffusion loss (default: True)"
    echo "  ALG  : Algorithm flag, True or False (default: True)"
    echo "  GPU  : GPU index to use (optional, default: None)"
    echo "  MOD  : Mode for the environment loss (default: None which means l1)"
    echo ""
    echo "Example:"
    echo "  ./multiple_runs.sh 20 log.txt 50 /Data/auguste.de-lambilly/mattergenbis/ Li-Co-O "'Co-O':3" 1.0 0.01 True 3 2 True 0 huber"
    exit 0
fi

# Default values for parameters
NB=${1:-20}
LOG=${2:-log2.txt}
MUL=${3:-50}
BASE=${4:-/Data/auguste.de-lambilly/mattergenbis/}
SYS=${5:-Li-Co-O}
ENV=${6:-"'Co-O':3"}
G=${7:-1.0}
K=${8:-1.0}
Norm=${9:-True}
R=${10:-3}
B=${11:-2}
ALG=${12:-True}
MOD=${13:-None}
GPU=${14:-None}

if [ "$ALG" == "True" ]; then
    al=2
    SUF="_guided2_" 
else
    al=1
    SUF="_guided_"
fi

SUF=${SUF}"env${ENV}_"

if [ $G != 1.0 ]; then
    SUF=${SUF}"g${G}_"
fi

if [ $K != 1.0 ]; then
    SUF=${SUF}"k${K}_"
fi

SUF=${SUF}"${Norm}_${R}-${B}"

if [ "$MOD" != "None" ]; then
    SUF=${SUF}"_${MOD}"
fi

if [ "$GPU" != "None" ]; then
    SUF=${SUF}"_gpu${GPU}"
fi

DIR="results/${SYS}${SUF}_"

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
        --guidance="{'environment': {'mode':$MOD, $ENV}}" \
        --diffusion_loss_weight=[$G,$K,$Norm] \
        --print_loss=False \
        --self_rec_steps=$R \
        --back_step=$B \
        --algo=$ALG \
        --force_gpu=$GPU >> $LOG 2>&1
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Generated samples for $SYS with environment $ENV at step $X"
    echo "Duration: ${duration} seconds at $(date +%H:%M:%S)"
done

main_file="${BASE}results/${SYS}_f/generated_crystals${SUF}.extxyz"
# Create the main file if it doesn't exist
if [ ! -f "$main_file" ]; then
    echo "Creating main file $main_file."
    if [ ! -d "$(dirname "$main_file")" ]; then
        mkdir -p "$(dirname "$main_file")"
    fi
    touch "$main_file"
fi

for X in $(seq 1 "$MUL"); do
    src="${BASE}${DIR}${X}/generated_crystals.extxyz"
    if [ -f "$src" ]; then
        cat "$src" >> "$main_file"
    else
        echo "Warning: $src does not exist, skipping."
    fi
done

echo "Everything copied to $main_file"