#!/usr/bin/env bash

set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  ./multiple_runs.sh --guidance DICT [OPTIONS]
  ./multiple_runs.sh --config FILE

Core options:
  --batch-size N                 Starting batch size (default: 20)
  --num-batches N                Batches per run (default: 1)
  --runs N                       Number of independent runs (default: 50)
  --system ELEMENTS              Chemical system (default: Li-Co-O)
  --guidance DICT                One-entry guidance dictionary (required)
  --config FILE                  Read all settings from YAML; cannot be combined
                                 with any other option

Generation options:
  --forward-weight FLOAT         Forward guidance weight g (default: 1.0)
  --backward-weight FLOAT        Backward guidance weight k (default: 1.0)
  --normalize BOOL               true or false (default: true)
  --self-rec-steps N             Self-recurrence steps (default: 3)
  --back-step N                  Back steps (default: 2)
  --algorithm N                  Guidance algorithm (default: 0)
  --diffusion-guidance-factor F  Diffusion guidance factor (default: 2.0)
  --gpu INDEX                    GPU index, or None (default: None)
  --gpu-memory-gb FLOAT          Optional GPU memory limit

OOM recovery:
  --oom-retries N                Retries after the initial attempt (default: 30)
  --oom-backoff-percent N        Batch retained after OOM, 1-99 (default: 80)
  --min-batch-size N             Smallest retry batch (default: 1)
  --oom-wait-seconds N           Cooldown between retries (default: 10)

Output and utility options:
  --base-dir PATH                Base directory for results (default: ./)
  --log-file PATH                Combined log file (default: log2.txt)
  --dry-run                      Print commands without running MatterGen
  --help                          Show this help

Example:
  ./multiple_runs.sh --batch-size 22 --runs 50 --system Ni-Pd-H \
    --guidance "{'mean_coordination': {'mode':'huber', 'alpha':3, '[Pd,Ni]-H':6}}" \
    --forward-weight 0.01 \
    --backward-weight 0.01 --normalize true --self-rec-steps 3 \
    --back-step 2 --algorithm 1 --gpu 2

YAML example:
  ./multiple_runs.sh --config examples/multiple_runs/mean_coordination.yaml
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
GUIDANCE=
FORWARD_WEIGHT=1.0
BACKWARD_WEIGHT=1.0
NORMALIZE=True
SELF_REC_STEPS=3
BACK_STEP=2
ALGORITHM=0
GPU=None
DIFFUSION_GUIDANCE_FACTOR=2.0
GPU_MEMORY_GB=
OOM_RETRIES=30
OOM_BACKOFF_PERCENT=80
MIN_BATCH_SIZE=1
OOM_WAIT_SECONDS=10
DRY_RUN=false

if [[ ${1:-} == --config ]]; then
    [[ $# -eq 2 ]] || fail "--config must be used alone: ./multiple_runs.sh --config FILE"
    CONFIG_FILE=$2
    [[ -n "$CONFIG_FILE" ]] || missing_value --config
    [[ -f "$CONFIG_FILE" ]] || fail "YAML config file not found: $CONFIG_FILE"

    if ! config_args=$(python3 - "$CONFIG_FILE" <<'PY'
import shlex
import sys

import yaml

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
except (OSError, yaml.YAMLError) as exc:
    raise SystemExit(f"Error: cannot read YAML config {path!r}: {exc}")

if not isinstance(config, dict):
    raise SystemExit("Error: YAML config must contain a top-level mapping.")

option_names = {
    "batch_size": "--batch-size",
    "num_batches": "--num-batches",
    "runs": "--runs",
    "system": "--system",
    "guidance": "--guidance",
    "diffusion_guidance_factor": "--diffusion-guidance-factor",
    "gpu": "--gpu",
    "gpu_memory_gb": "--gpu-memory-gb",
    "oom_retries": "--oom-retries",
    "oom_backoff_percent": "--oom-backoff-percent",
    "min_batch_size": "--min-batch-size",
    "oom_wait_seconds": "--oom-wait-seconds",
    "base_dir": "--base-dir",
    "log_file": "--log-file",
    "dry_run": "--dry-run",
}

unknown = sorted(set(config) - set(option_names))
if unknown:
    raise SystemExit(f"Error: unknown YAML setting(s): {', '.join(unknown)}")
if "guidance" not in config:
    raise SystemExit("Error: YAML config requires a 'guidance' mapping.")
guidance = config["guidance"]
if not isinstance(guidance, dict):
    raise SystemExit("Error: YAML 'guidance' must be a mapping.")
unknown_guidance = sorted(set(guidance) - {"type", "parameters", "settings"})
if unknown_guidance:
    raise SystemExit(
        f"Error: unknown YAML guidance setting(s): {', '.join(unknown_guidance)}"
    )
missing_guidance = sorted({"type", "parameters"} - set(guidance))
if missing_guidance:
    raise SystemExit(
        f"Error: missing YAML guidance setting(s): {', '.join(missing_guidance)}"
    )
if not isinstance(guidance["type"], str) or not guidance["type"]:
    raise SystemExit("Error: YAML 'guidance.type' must be a non-empty string.")
guidance_settings = guidance.get("settings", {})
if not isinstance(guidance_settings, dict):
    raise SystemExit("Error: YAML 'guidance.settings' must be a mapping.")
guidance_option_names = {
    "forward_weight": "--forward-weight",
    "backward_weight": "--backward-weight",
    "normalize": "--normalize",
    "self_rec_steps": "--self-rec-steps",
    "back_step": "--back-step",
    "algorithm": "--algorithm",
}
unknown_settings = sorted(set(guidance_settings) - set(guidance_option_names))
if unknown_settings:
    raise SystemExit(
        f"Error: unknown YAML guidance.settings value(s): {', '.join(unknown_settings)}"
    )

def scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "None"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise SystemExit(f"Error: expected a scalar YAML value, got {type(value).__name__}.")

args = []
for key, option in option_names.items():
    if key not in config:
        continue
    value = config[key]
    if key == "guidance":
        args.extend((option, repr({value["type"]: value["parameters"]})))
        for setting, setting_option in guidance_option_names.items():
            if setting in guidance_settings:
                args.extend((setting_option, scalar(guidance_settings[setting])))
    elif key == "dry_run":
        if not isinstance(value, bool):
            raise SystemExit("Error: YAML 'dry_run' must be true or false.")
        if value:
            args.append(option)
    else:
        args.extend((option, scalar(value)))

print(shlex.join(args))
PY
    ); then
        exit 2
    fi
    eval "set -- $config_args"
fi

declare -A OPTION_VARIABLES=(
    [--runs]=RUNS
    [--system]=SYSTEM
    [--guidance]=GUIDANCE
    [--batch-size]=BATCH_SIZE
    [--num-batches]=NUM_BATCHES
    [--diffusion-guidance-factor]=DIFFUSION_GUIDANCE_FACTOR
    [--forward-weight]=FORWARD_WEIGHT
    [--backward-weight]=BACKWARD_WEIGHT
    [--normalize]=NORMALIZE
    [--self-rec-steps]=SELF_REC_STEPS
    [--back-step]=BACK_STEP
    [--algorithm]=ALGORITHM
    [--gpu-memory-gb]=GPU_MEMORY_GB
    [--base-dir]=BASE_DIR
    [--log-file]=LOG_FILE
    [--oom-retries]=OOM_RETRIES
    [--oom-backoff-percent]=OOM_BACKOFF_PERCENT
    [--min-batch-size]=MIN_BATCH_SIZE
    [--oom-wait-seconds]=OOM_WAIT_SECONDS
    [--gpu]=GPU
)

while [[ $# -gt 0 ]]; do
    option=$1
    case "$option" in
        --help)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --config)
            fail "--config must be used alone: ./multiple_runs.sh --config FILE"
            ;;
        --*)
            variable=${OPTION_VARIABLES[$option]:-}
            [[ -n "$variable" ]] \
                || fail "unknown option '$option'. Run './multiple_runs.sh --help' for usage."
            [[ $# -ge 2 && -n "${2:-}" ]] || missing_value "$option"
            printf -v "$variable" '%s' "$2"
            shift 2
            ;;
        -*)
            fail "unknown option '$option'. Run './multiple_runs.sh --help' for usage."
            ;;
        *)
            fail "unexpected positional argument '$option'. Use named options or --config FILE."
            ;;
    esac
done

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
if [[ -n "$GPU_MEMORY_GB" ]]; then
    require_number --gpu-memory-gb "$GPU_MEMORY_GB"
fi

case "${NORMALIZE,,}" in
    true) NORMALIZE=True ;;
    false) NORMALIZE=False ;;
    *) fail "--normalize expects true or false; got '$NORMALIZE'." ;;
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

[[ -n "$GUIDANCE" ]] || fail "--guidance is required. Alternatively, use --config FILE."
if ! GUIDANCE=$(python3 - "$GUIDANCE" <<'PY'
import ast
import sys

try:
    guidance = ast.literal_eval(sys.argv[1])
except (SyntaxError, ValueError) as exc:
    raise SystemExit(f"Error: --guidance must be a valid Python dictionary literal: {exc}")
if not isinstance(guidance, dict) or not guidance:
    raise SystemExit("Error: --guidance must be a non-empty dictionary.")
if len(guidance) != 1:
    raise SystemExit("Error: --guidance must define exactly one guidance type.")
if not all(isinstance(name, str) and name for name in guidance):
    raise SystemExit("Error: every top-level --guidance key must be a non-empty string.")
print(repr(guidance))
PY
); then
    exit 2
fi

if [[ "$BASE_DIR" != */ ]]; then
    BASE_DIR="${BASE_DIR}/"
fi

type_tag=$(python3 -c \
    'import ast, sys; print("+".join(ast.literal_eval(sys.argv[1])))' "$GUIDANCE" \
    | tr -cd 'A-Za-z0-9._+-')
param_tag=$(printf '%s' "$GUIDANCE" | tr -d "{}[]'\":, " | tr -cd 'A-Za-z0-9._+-')
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
