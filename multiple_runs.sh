#!/bin/bash


if [[ "$1" == "--help" ]]; then
    echo "Usage:"
    echo "  ./multiple_runs.sh NB LOG MUL BASE SYS ENV G K Norm R B ALG MOD GPU"
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
    echo "  ALG  : Algorithm flag, number of the algorithm (default: 0)"
    echo "  GPU  : GPU index to use (optional, default: None)"
    echo "  MOD  : Mode for the environment loss (default: None which means l1)"
    echo ""
    echo "Example:"
    echo " bash multiple_runs.sh 20 log.txt 50 /Data/auguste.de-lambilly/mattergenbis/ Li-Co-O \"'Co-O':6\" 0.0001 0.0001 True 3 2 False huber 0"
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
ALG=${12:-0}
MOD=${13:-None}
GPU=${14:-None}

SUF="_guided-${ALG}_"

clean_env="${ENV//\'/}"    # Remove all single quotes
clean_env="${clean_env//:/}" 

SUF=${SUF}"env${clean_env}_"

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

durations="${SUF}"  # Initialize durations variable

for X in $(seq 1 "$MUL"); do
    echo "Generating $NB samples for $SYS into ${DIR}${X} at $(date +%H:%M:%S)"
    start_time=$(date +%s)
    while true; do
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
        if tail -n 3 $LOG | grep "torch\.cuda\.OutOfMemoryError"; then
            echo "CUDA Out of memory error, waiting 60 seconds before retrying..."
            sleep 60
        else
            break
        fi
    done
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    durations+=", $duration"  # Append duration to durations variable
    echo "Generated samples for $SYS with environment $ENV at step $X"
    echo "Duration: ${duration} seconds at $(date +%H:%M:%S)"
done

main_file="${BASE}results/${SYS}_f/generated_crystals${SUF}.extxyz"
hard_save="/users/eleves-b/2021/auguste.de-lambilly/results/${SYS}_f/generated_crystals${SUF}.extxyz"


# Create the main file if it doesn't exist
if [ ! -f "$main_file" ]; then
    echo "Creating main file $main_file."
    if [ ! -d "$(dirname "$main_file")" ]; then
        mkdir -p "$(dirname "$main_file")"
    fi
    touch "$main_file"
fi

if [ ! -f "$hard_save" ]; then
    echo "Creating main file $hard_save."
    if [ ! -d "$(dirname "$hard_save")" ]; then
        mkdir -p "$(dirname "$hard_save")"
    fi
    touch "$hard_save"
fi

# Save durations to durations.txt
durations_file="$(dirname "$hard_save")/durations.txt"
echo "$durations" >> "$durations_file"

for X in $(seq 1 "$MUL"); do
    src="${BASE}${DIR}${X}/generated_crystals.extxyz"
    if [ -f "$src" ]; then
        cat "$src" >> "$main_file"
        cat "$src" >> "$hard_save"
    else
        echo "Warning: $src does not exist, skipping."
    fi
done

echo "Everything copied to $main_file"