#!/usr/bin/env bash

set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  ./multiple_runs.sh [OPTIONS]

Core options:
  --batch-size N                 Starting batch size (default: 20)
  --num-batches N                Batches per run (default: 1)
  --runs N                       Number of independent runs (default: 50)
  --system ELEMENTS              Chemical system (default: Li-Co-O)
  --environment TARGETS          Inner environment targets (default: 'Co-O':3)
  --guidance-type TYPE           Guidance function (default: environment)
  --guidance-params DICT         Complete inner guidance dictionary; overrides
                                 --environment, --loss-mode, and --alpha
  --loss-mode MODE               l1, l2, or huber (default: l1)
  --alpha FLOAT                  Sigmoid steepness (default: 2.0)

Generation options:
  --forward-weight FLOAT         Forward diffusion loss weight (default: 1.0)
  --backward-weight FLOAT        Backward diffusion loss weight (default: 1.0)
  --normalize BOOL               true or false (default: true)
  --self-rec-steps N             Self-recurrence steps (default: 3)
  --back-step N                  Back steps (default: 2)
  --algorithm N                  Guidance algorithm (default: 0)
  --diffusion-guidance-factor F  Diffusion guidance factor (default: 2.0)
  --gpu INDEX                    GPU index, or None (default: None)
  --gpu-memory-gb FLOAT          Optional GPU memory limit
  --extra-arg ARG                Append one mattergen-generate argument; repeatable

OOM recovery:
  --oom-retries N                Retries after the initial attempt (default: 30)
  --oom-backoff-percent N        Batch retained after OOM, 1-99 (default: 80)
  --min-batch-size N             Smallest retry batch (default: 1)
  --oom-wait-seconds N           Cooldown between retries (default: 10)

Output and utility options:
  --base-dir PATH                Base directory for results (default: ./)
  --log-file PATH                Combined log file (default: log2.txt)
  --dry-run                      Print commands without running MatterGen
  -h, --help                     Show this help

Environment targets may use A-[B1,B2,...] or [A1,A2,...]-B. Only one side may
contain multiple species.

Legacy interfaces are retained in this same script:
  positional: NB LOG RUNS BASE SYS ENV G K NORM R B ALG MODE GPU [ALPHA]
  short flags: -x -s -t -p -b -m -d -u -v -c -r -B -a -M -o -l
               -R -O -N -W -f

Example:
  ./multiple_runs.sh --batch-size 22 --runs 50 --system Ni-Pd-H \
    --environment "'[Pd,Ni]-H':6" --forward-weight 0.01 \
    --backward-weight 0.01 --normalize true --self-rec-steps 3 \
    --back-step 2 --algorithm 1 --loss-mode huber --gpu 2 --alpha 3
EOF
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

missing_value() {
    fail "$1 requires a value. Run './multiple_runs.sh --help' for usage."
}

require_positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 expects a positive integer; got '$2'."
}

require_nonnegative_integer() {
    [[ "$2" =~ ^(0|[1-9][0-9]*)$ ]] || fail "$1 expects a non-negative integer; got '$2'."
}

require_number() {
    local number_pattern='^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
    [[ "$2" =~ $number_pattern ]] || fail "$1 expects a number; got '$2'."
}

detect_oom() {
    local attempt_log=$1
    grep -E -i \
        -e 'CUDA out of memory' \
        -e 'RuntimeError:.*out of memory' \
        -e 'CUBLAS_STATUS_ALLOC_FAILED' \
        -e 'cuMemAlloc' \
        -e 'ResourceExhausted' \
        -e 'std::bad_alloc' \
        -e 'MemoryError' \
        -e 'Killed process .* out of memory' \
        -e '\bOOM\b' \
        "$attempt_log" >/dev/null 2>&1
}

BATCH_SIZE=20
NUM_BATCHES=1
RUNS=50
LOG_FILE=log2.txt
BASE_DIR=./
SYSTEM=Li-Co-O
ENVIRONMENT="'Co-O':3"
GUIDANCE_TYPE=environment
GUIDANCE_PARAMS=
FORWARD_WEIGHT=1.0
BACKWARD_WEIGHT=1.0
NORMALIZE=True
SELF_REC_STEPS=3
BACK_STEP=2
ALGORITHM=0
LOSS_MODE=None
GPU=None
ALPHA=2.0
DIFFUSION_GUIDANCE_FACTOR=2.0
GPU_MEMORY_GB=
OOM_RETRIES=30
OOM_BACKOFF_PERCENT=80
MIN_BATCH_SIZE=1
OOM_WAIT_SECONDS=10
DRY_RUN=false
LEGACY_FRACTION=
LEGACY_EXTRA=
EXTRA_ARGS=()

GUIDANCE_PARAMS_SET=false
ENVIRONMENT_OPTIONS_SET=false

# Preserve the original root-script positional interface.
if [[ $# -gt 0 && "$1" != -* ]]; then
    [[ $# -le 15 ]] || fail "too many positional arguments; expected at most 15."
    [[ $# -ge 1 ]] && BATCH_SIZE=$1
    [[ $# -ge 2 ]] && LOG_FILE=$2
    [[ $# -ge 3 ]] && RUNS=$3
    [[ $# -ge 4 ]] && BASE_DIR=$4
    [[ $# -ge 5 ]] && SYSTEM=$5
    [[ $# -ge 6 ]] && ENVIRONMENT=$6
    [[ $# -ge 7 ]] && FORWARD_WEIGHT=$7
    [[ $# -ge 8 ]] && BACKWARD_WEIGHT=$8
    [[ $# -ge 9 ]] && NORMALIZE=$9
    [[ $# -ge 10 ]] && SELF_REC_STEPS=${10}
    [[ $# -ge 11 ]] && BACK_STEP=${11}
    [[ $# -ge 12 ]] && ALGORITHM=${12}
    [[ $# -ge 13 ]] && LOSS_MODE=${13}
    [[ $# -ge 14 ]] && GPU=${14}
    [[ $# -ge 15 ]] && ALPHA=${15}
    ENVIRONMENT_OPTIONS_SET=true
    shift "$#"
else
    while [[ $# -gt 0 ]]; do
        option=$1
        if [[ "$option" == --*=* ]]; then
            value=${option#*=}
            option=${option%%=*}
            set -- "$option" "$value" "${@:2}"
        fi

        case "$option" in
            -h|--help)
                usage
                exit 0
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --extra-arg)
                [[ $# -ge 2 ]] || missing_value "$option"
                EXTRA_ARGS+=("$2")
                shift 2
                ;;
            -x|--runs|-s|--system|-t|--guidance-type|-p|--guidance-params|-b|--batch-size|-m|--num-batches|-d|--diffusion-guidance-factor|-u|--forward-weight|-v|--backward-weight|-c|--normalize|-r|--self-rec-steps|-B|--back-step|-a|--algorithm|-M|--gpu-memory-gb|-o|--base-dir|-l|--log-file|-R|--oom-retries|-O|--oom-backoff-percent|-N|--min-batch-size|-W|--oom-wait-seconds|-f|--gpu|--environment|--loss-mode|--alpha|-F|--fraction|-e|--extra-args)
                [[ $# -ge 2 && -n "${2:-}" ]] || missing_value "$option"
                value=$2
                shift 2
                case "$option" in
                    -x|--runs) RUNS=$value ;;
                    -s|--system) SYSTEM=$value ;;
                    -t|--guidance-type) GUIDANCE_TYPE=$value ;;
                    -p|--guidance-params)
                        GUIDANCE_PARAMS=$value
                        GUIDANCE_PARAMS_SET=true
                        ;;
                    -b|--batch-size) BATCH_SIZE=$value ;;
                    -m|--num-batches) NUM_BATCHES=$value ;;
                    -d|--diffusion-guidance-factor) DIFFUSION_GUIDANCE_FACTOR=$value ;;
                    -u|--forward-weight) FORWARD_WEIGHT=$value ;;
                    -v|--backward-weight) BACKWARD_WEIGHT=$value ;;
                    -c|--normalize) NORMALIZE=$value ;;
                    -r|--self-rec-steps) SELF_REC_STEPS=$value ;;
                    -B|--back-step) BACK_STEP=$value ;;
                    -a|--algorithm) ALGORITHM=$value ;;
                    -M|--gpu-memory-gb) GPU_MEMORY_GB=$value ;;
                    -o|--base-dir) BASE_DIR=$value ;;
                    -l|--log-file) LOG_FILE=$value ;;
                    -R|--oom-retries) OOM_RETRIES=$value ;;
                    -O|--oom-backoff-percent) OOM_BACKOFF_PERCENT=$value ;;
                    -N|--min-batch-size) MIN_BATCH_SIZE=$value ;;
                    -W|--oom-wait-seconds) OOM_WAIT_SECONDS=$value ;;
                    -f|--gpu) GPU=$value ;;
                    --environment)
                        ENVIRONMENT=$value
                        ENVIRONMENT_OPTIONS_SET=true
                        ;;
                    --loss-mode)
                        LOSS_MODE=$value
                        ENVIRONMENT_OPTIONS_SET=true
                        ;;
                    --alpha)
                        ALPHA=$value
                        ENVIRONMENT_OPTIONS_SET=true
                        ;;
                    -F|--fraction) LEGACY_FRACTION=$value ;;
                    -e|--extra-args) LEGACY_EXTRA=$value ;;
                esac
                ;;
            --*|-*)
                fail "unknown option '$option'. Run './multiple_runs.sh --help' for usage."
                ;;
            *)
                fail "unexpected positional argument '$option'. Do not mix positional and option forms."
                ;;
        esac
    done
fi

require_positive_integer --batch-size "$BATCH_SIZE"
require_positive_integer --num-batches "$NUM_BATCHES"
require_positive_integer --runs "$RUNS"
require_nonnegative_integer --self-rec-steps "$SELF_REC_STEPS"
require_nonnegative_integer --back-step "$BACK_STEP"
require_nonnegative_integer --oom-retries "$OOM_RETRIES"
require_nonnegative_integer --oom-wait-seconds "$OOM_WAIT_SECONDS"
require_positive_integer --min-batch-size "$MIN_BATCH_SIZE"
require_positive_integer --oom-backoff-percent "$OOM_BACKOFF_PERCENT"
((OOM_BACKOFF_PERCENT < 100)) || fail "--oom-backoff-percent must be between 1 and 99."
((MIN_BATCH_SIZE <= BATCH_SIZE)) || fail "--min-batch-size cannot exceed --batch-size."

require_number --forward-weight "$FORWARD_WEIGHT"
require_number --backward-weight "$BACKWARD_WEIGHT"
require_number --diffusion-guidance-factor "$DIFFUSION_GUIDANCE_FACTOR"
require_number --alpha "$ALPHA"
if ! awk -v alpha="$ALPHA" 'BEGIN { exit !(alpha > 0) }'; then
    fail "--alpha expects a positive number; got '$ALPHA'."
fi
if [[ -n "$GPU_MEMORY_GB" ]]; then
    require_number --gpu-memory-gb "$GPU_MEMORY_GB"
fi

case "${NORMALIZE,,}" in
    true) NORMALIZE=True ;;
    false) NORMALIZE=False ;;
    *) fail "--normalize expects true or false; got '$NORMALIZE'." ;;
esac

case "${LOSS_MODE,,}" in
    none) LOSS_MODE=None ;;
    l1|l2|huber) LOSS_MODE=${LOSS_MODE,,} ;;
    *) fail "--loss-mode expects l1, l2, or huber; got '$LOSS_MODE'." ;;
esac

case "${GPU,,}" in
    none) GPU=None ;;
    *) require_nonnegative_integer --gpu "$GPU" ;;
esac

case "${ALGORITHM,,}" in
    true) ALGORITHM=True ;;
    false) ALGORITHM=False ;;
    *) require_nonnegative_integer --algorithm "$ALGORITHM" ;;
esac

[[ "$GUIDANCE_TYPE" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] \
    || fail "invalid --guidance-type '$GUIDANCE_TYPE'."

if $GUIDANCE_PARAMS_SET && $ENVIRONMENT_OPTIONS_SET; then
    fail "--guidance-params cannot be combined with --environment, --loss-mode, or --alpha."
fi

if $GUIDANCE_PARAMS_SET; then
    [[ "$GUIDANCE_PARAMS" == \{*\} ]] \
        || fail "--guidance-params must be a dictionary literal enclosed in braces."
    INNER_GUIDANCE=$GUIDANCE_PARAMS
else
    if [[ "$LOSS_MODE" == None ]]; then
        mode_literal=None
    else
        mode_literal="'$LOSS_MODE'"
    fi
    INNER_GUIDANCE="{'mode':$mode_literal, 'alpha':$ALPHA, $ENVIRONMENT}"
fi
GUIDANCE="{'$GUIDANCE_TYPE': $INNER_GUIDANCE}"

if [[ -n "$LEGACY_EXTRA" ]]; then
    read -r -a legacy_extra_args <<< "$LEGACY_EXTRA"
    EXTRA_ARGS+=("${legacy_extra_args[@]}")
fi

if [[ "$BASE_DIR" != */ ]]; then
    BASE_DIR="${BASE_DIR}/"
fi

type_tag=$(printf '%s' "$GUIDANCE_TYPE" | tr -cd 'A-Za-z0-9._-')
param_tag=$(printf '%s' "$INNER_GUIDANCE" | tr -d "{}[]'\":, " | tr -cd 'A-Za-z0-9._+-')
[[ -n "$type_tag" ]] || type_tag=guidance
[[ -n "$param_tag" ]] || param_tag=params

settings_tag="g${FORWARD_WEIGHT}_k${BACKWARD_WEIGHT}_${NORMALIZE}_${SELF_REC_STEPS}-${BACK_STEP}_alg${ALGORITHM}"
if [[ "$GPU" != None ]]; then
    settings_tag="${settings_tag}_gpu${GPU}"
fi

OUTPUT_ROOT="${BASE_DIR}results/${SYSTEM}/${type_tag}/${param_tag}/${settings_tag}"
AGGREGATE_FILE="${OUTPUT_ROOT}/generated_crystals.extxyz"
DURATIONS_FILE="${OUTPUT_ROOT}/durations.csv"
mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"
printf 'run,duration_seconds,final_batch_size,attempts\n' > "$DURATIONS_FILE"

log() {
    local line
    line="$(date -Is) >>> $*"
    printf '%s\n' "$line"
    printf '%s\n' "$line" >> "$LOG_FILE"
}

log_error() {
    local line
    line="$(date -Is) >>> $*"
    printf '%s\n' "$line" >&2
    printf '%s\n' "$line" >> "$LOG_FILE"
}

if ! $DRY_RUN; then
    script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log "Using active virtual environment: $VIRTUAL_ENV"
    elif [[ -f "$script_dir/../.venv/bin/activate" ]]; then
        log "Activating ../.venv"
        # shellcheck disable=SC1091
        source "$script_dir/../.venv/bin/activate"
    elif command -v mattergen-generate >/dev/null 2>&1; then
        log "Using mattergen-generate from PATH"
    else
        log_error "No active virtual environment and mattergen-generate is not in PATH"
        exit 127
    fi
fi

log "System: $SYSTEM"
log "Guidance: $GUIDANCE"
log "Output: $OUTPUT_ROOT"
log "Runs: $RUNS; starting batch size: $BATCH_SIZE; num_batches: $NUM_BATCHES"
log "OOM retries: $OOM_RETRIES; backoff: ${OOM_BACKOFF_PERCENT}%; minimum batch: $MIN_BATCH_SIZE"

for ((run = 1; run <= RUNS; run++)); do
    run_path="${OUTPUT_ROOT}/run_${run}"
    mkdir -p "$run_path"
    current_batch_size=$BATCH_SIZE
    attempt=0
    run_start=$(date +%s)

    while true; do
        attempt=$((attempt + 1))
        attempt_log="${run_path}/attempt_${attempt}.log"
        command=(
            mattergen-generate "$run_path"
            --pretrained-name=chemical_system
            --batch_size="$current_batch_size"
            --num_batches="$NUM_BATCHES"
            --properties_to_condition_on="{'chemical_system':'${SYSTEM}'}"
            --record_trajectories=False
            --diffusion_guidance_factor="$DIFFUSION_GUIDANCE_FACTOR"
            --guidance="$GUIDANCE"
            --diffusion_loss_weight="[$FORWARD_WEIGHT,$BACKWARD_WEIGHT,$NORMALIZE]"
            --print_loss=False
            --self_rec_steps="$SELF_REC_STEPS"
            --back_step="$BACK_STEP"
            --algo="$ALGORITHM"
            --force_gpu="$GPU"
        )

        if [[ -n "$GPU_MEMORY_GB" ]]; then
            command+=(--gpu_memory_gb="$GPU_MEMORY_GB")
        fi
        if [[ -n "$LEGACY_FRACTION" ]]; then
            command+=(-f "$LEGACY_FRACTION")
        fi
        if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
            command+=("${EXTRA_ARGS[@]}")
        fi

        {
            printf '\n# Run %d/%d, attempt %d, batch_size=%s\n' \
                "$run" "$RUNS" "$attempt" "$current_batch_size"
            printf 'CMD: '
            printf '%q ' "${command[@]}"
            printf '\n'
        } >> "$LOG_FILE"

        if $DRY_RUN; then
            printf 'DRY RUN: '
            printf '%q ' "${command[@]}"
            printf '\n'
            break
        fi

        log "Run $run/$RUNS, attempt $attempt: batch_size=$current_batch_size"
        "${command[@]}" > "$attempt_log" 2>&1
        status=$?
        cat "$attempt_log" >> "$LOG_FILE"

        if [[ $status -eq 0 ]]; then
            log "Run $run succeeded on attempt $attempt with batch_size=$current_batch_size"
            break
        fi

        oom_detected=false
        if [[ $status -eq 9 || $status -eq 137 || $status -eq 143 ]] \
            || detect_oom "$attempt_log"; then
            oom_detected=true
        fi

        if ! $oom_detected; then
            log_error "Run $run failed with non-OOM exit status $status; see $attempt_log"
            exit "$status"
        fi

        if ((attempt > OOM_RETRIES)); then
            log_error "Run $run exceeded $OOM_RETRIES OOM retries"
            exit 1
        fi

        next_batch_size=$(((current_batch_size * OOM_BACKOFF_PERCENT + 99) / 100))
        if ((next_batch_size >= current_batch_size)); then
            next_batch_size=$((current_batch_size - 1))
        fi
        if ((next_batch_size < MIN_BATCH_SIZE)); then
            log_error "Cannot reduce batch below minimum $MIN_BATCH_SIZE (current: $current_batch_size)"
            exit 1
        fi

        log "OOM detected; reducing batch_size $current_batch_size -> $next_batch_size"
        if ((OOM_WAIT_SECONDS > 0)); then
            sleep "$OOM_WAIT_SECONDS"
        fi
        current_batch_size=$next_batch_size
    done

    run_end=$(date +%s)
    duration=$((run_end - run_start))
    printf '%d,%d,%d,%d\n' "$run" "$duration" "$current_batch_size" "$attempt" \
        >> "$DURATIONS_FILE"
done

if $DRY_RUN; then
    log "Dry run complete; no generation was executed"
    exit 0
fi

: > "$AGGREGATE_FILE"
for ((run = 1; run <= RUNS; run++)); do
    source_file="${OUTPUT_ROOT}/run_${run}/generated_crystals.extxyz"
    if [[ ! -f "$source_file" ]]; then
        log_error "Expected output is missing: $source_file"
        exit 1
    fi
    cat "$source_file" >> "$AGGREGATE_FILE"
done

log "All runs completed"
log "Combined structures: $AGGREGATE_FILE"
log "Durations: $DURATIONS_FILE"
