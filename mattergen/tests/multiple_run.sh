#!/usr/bin/env bash
# multiple_run.sh — robust, OOM-aware runner for mattergen-generate
# -----------------------------------------------------------------------------
# Purpose:
#   Automatically restart runs that occasionally fail with GPU/CPU OOM,
#   reducing --batch_size stepwise until the run succeeds (or a floor is hit).
#
# Key features:
#   - Safe quoting of Python-literal CLI values (dicts/lists/True/False).
#   - Output directories encode system, guidance TYPE+PARAMS, and key settings.
#   - OOM detection via exit codes and common error messages in the log.
#   - Exponential-like backoff on --batch_size after each OOM.
#   - Separate RUN_N folders, but retries re-use the same folder for resuming.
#
# Examples (mirror your two scenarios):
# 1) Dominant environment guidance (Co–O coord = 3)
#    ./multiple_run.sh -x 1 \
#      -s "Li-Co-O" -t dominant_environment -p "{'Co-O':[3]}" \
#      -b 12 -m 86 -d 2.0 -u 1.0 -v 1.0 -c True -r 3 -B 2 -a False -M 22 -F 0
#
# 2) Environment guidance with Huber mode (Si–O coord = 6)
#    ./multiple_run.sh -x 1 \
#      -s "Si-O" -t environment -p "{'mode':'huber','Si-O':[6, 2.4]}" \
#      -b 30 -m 5 -d 2.0 -u 0.01 -v 0.01 -c True -r 3 -B 2 -a False -M 20
# -----------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'USAGE'
Usage: multiple_run.sh [options]
  -x RUNS                 Number of runs (loop iterations). Default: 1
  -s SYS                  Chemical system string (e.g. "Li-Co-O"). Default: "Li-Co-O"
  -t GUIDTYPE             Guidance type (e.g. environment | dominant_environment). Default: "environment"
  -p GUIDPARAM            Guidance param as Python dict literal, e.g. "{'Co-O':[3, 2.3]}" (MUST include braces, 3 instead of [3] for default cutoff)
  -b NB                   --batch_size (starting). Default: 12
  -m MUL                  --num_batches. Default: 86
  -d DGF                  --diffusion_guidance_factor. Default: 2.0
  -u G                    diffusion_loss_weight first term. Default: 1.0
  -v K                    diffusion_loss_weight second term. Default: 1.0
  -c NORM                 diffusion_loss_weight third term (True/False). Default: True
  -r R                    --self_rec_steps. Default: 3
  -B B                    --back_step. Default: 2
  -a ALG                  --algo (e.g., False or an int). Default: False
  -M GPU_MEM_GB           --gpu_memory_gb (optional). Omit to skip.
  -F FRAC                 Extra `-f FRAC` flag to mattergen-generate (optional).
  -o OUTBASE              Output base directory. Default: results
  -l LOGFILE              Log file path. Default: (auto inside run dir)
  -e EXTRA                Extra args appended to mattergen-generate (quoted as one string).
  # OOM-handling knobs:
  -R OOM_RETRIES          Max retries on OOM per run. Default: 30
  -O OOM_BACKOFF_PCT      Backoff percent for batch size (integer, e.g., 70 means 70%). Default: 80
  -N MIN_BATCH_SIZE       Minimum allowed batch size before giving up. Default: 1
  -W WAIT_SEC             Seconds to wait between retries (cooldown). Default: 10
  -f FORCE_GPU            Index of the invoked gpu. Default: 0
  -h                      Show this help and exit.

Examples:
  ./multiple_run.sh -x 1 -s "Li-Co-O" -t dominant_environment -p "{'Co-O': 3}"
  ./multiple_run.sh -x 1 -s "Si-O"    -t environment           -p "{'mode':'huber','Si-O':[6, 0.5]}"
USAGE
}

# Defaults
RUNS=50
SYS="Li-Co-O"
GUIDTYPE="environment"
GUIDPARAM="{'Co-O':[3]}"
NB=12
MUL=1
DGF=2.0
G=1.0
K=1.0
NORM=True
R=3
B=2
ALG=False
GPU_MEM_GB=""
FRAC=""
OUTBASE="results"
LOGFILE=""
EXTRA=""
OOM_RETRIES=30
OOM_BACKOFF_PCT=80
MIN_BATCH_SIZE=1
WAIT_SEC=10
FORCE_GPU=0

while getopts ":x:s:t:p:b:m:d:u:v:c:r:B:a:M:F:o:l:e:R:O:N:W:f:h" opt; do
  case "$opt" in
    x) RUNS="$OPTARG" ;;
    s) SYS="$OPTARG" ;;
    t) GUIDTYPE="$OPTARG" ;;
    p) GUIDPARAM="$OPTARG" ;;
    b) NB="$OPTARG" ;;
    m) MUL="$OPTARG" ;;
    d) DGF="$OPTARG" ;;
    u) G="$OPTARG" ;;
    v) K="$OPTARG" ;;
    c) NORM="$OPTARG" ;;
    r) R="$OPTARG" ;;
    B) B="$OPTARG" ;;
    a) ALG="$OPTARG" ;;
    M) GPU_MEM_GB="$OPTARG" ;;
    F) FRAC="$OPTARG" ;;
    o) OUTBASE="$OPTARG" ;;
    l) LOGFILE="$OPTARG" ;;
    e) EXTRA="$OPTARG" ;;
    R) OOM_RETRIES="$OPTARG" ;;
    O) OOM_BACKOFF_PCT="$OPTARG" ;;
    N) MIN_BATCH_SIZE="$OPTARG" ;;
    W) WAIT_SEC="$OPTARG" ;;
    f) FORCE_GPU="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option -$OPTARG" >&2; usage; exit 2 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done
shift $((OPTIND-1))

# Sanity checks
if ! [[ "$OOM_BACKOFF_PCT" =~ ^[0-9]+$ ]] || [ "$OOM_BACKOFF_PCT" -le 0 ] || [ "$OOM_BACKOFF_PCT" -ge 100 ]; then
  echo "ERROR: -O OOM_BACKOFF_PCT must be an integer between 1 and 99 (got: $OOM_BACKOFF_PCT)"
  exit 2
fi
if ! [[ "$MIN_BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$MIN_BATCH_SIZE" -lt 1 ]; then
  echo "ERROR: -N MIN_BATCH_SIZE must be a positive integer (got: $MIN_BATCH_SIZE)"
  exit 2
fi

# --- Guidance type short tag ---
case "$GUIDTYPE" in
  dominant_environment) TYPE_TAG="domenv" ;;
  environment)          TYPE_TAG="env" ;;
  *)                    TYPE_TAG="$GUIDTYPE" ;;
esac
# Sanitize tag (letters/digits/._- only)
TYPE_TAG="$(printf '%s' "$TYPE_TAG" | tr -cd 'A-Za-z0-9._-')"

# --- Parameter tag: strip braces, quotes, commas, colons, spaces, brackets. Keep hyphen (e.g., Si-O) ---
PARAM_TAG="$(printf '%s' "$GUIDPARAM" | tr -d "{}[]'\":, " )"

# --- Compose output directories ---
OUTROOT="${OUTBASE%/}/${SYS}/${TYPE_TAG}/${PARAM_TAG}"
SET_TAG="g${G}_k${K}_${NORM}_${R}-${B}_alg${ALG}"
if [[ -n "$GPU_MEM_GB" ]]; then
  SET_TAG="${SET_TAG}_gpu${GPU_MEM_GB}"
fi
DIR="${OUTROOT}/${SET_TAG}"
mkdir -p "$DIR"

# Decide log file if not provided
if [[ -z "$LOGFILE" ]]; then
  LOGFILE="${DIR}/mattergen.log"
fi

echo ">>> multiple_run.sh starting"
echo ">>> System:            $SYS"
echo ">>> Guidance type:     $GUIDTYPE  (tag: $TYPE_TAG)"
echo ">>> Guidance param:    $GUIDPARAM  (tag: $PARAM_TAG)"
echo ">>> Output directory:  $DIR"
echo ">>> Log file:          $LOGFILE"
echo ">>> Runs:              $RUNS"
echo ">>> mattergen: batch_size=$NB num_batches=$MUL dgf=$DGF dlw=[$G,$K,$NORM] rec=$R back=$B algo=$ALG gpu_mem=$GPU_MEM_GB frac=$FRAC"
echo ">>> OOM: retries=$OOM_RETRIES backoff=${OOM_BACKOFF_PCT}% min_batch=$MIN_BATCH_SIZE wait=${WAIT_SEC}s"
echo "--------------------------------------------------------------------"

# Turn EXTRA into an array safely (word-splitting respecting quotes)
read -r -a EXTRA_ARR <<< "$EXTRA"

# Detect OOM in the last lines of the log (exit 0 if OOM-like found)
detect_oom() {
  local logfile="$1"
  # tail a generous chunk to catch messages
  tail -n 400 "$logfile" 2>/dev/null | grep -E -i \
    -e "CUDA out of memory" \
    -e "RuntimeError:.*out of memory" \
    -e "CUBLAS_STATUS_ALLOC_FAILED" \
    -e "cuMemAlloc" \
    -e "ResourceExhausted" \
    -e "std::bad_alloc" \
    -e "MemoryError" \
    -e "Killed process .* out of memory" \
    -e "OOM" \
    >/dev/null && return 0 || return 1
}

for i in $(seq 1 "$RUNS"); do
  RUN_PATH="${DIR}/run_${i}"
  mkdir -p "$RUN_PATH"

  # dynamic batch size per attempt
  cur_nb="$NB"
  attempt=0

  while : ; do
    attempt=$((attempt + 1))

    # Build command as an array to preserve quoting
    cmd=( mattergen-generate "$RUN_PATH"
          --pretrained-name=chemical_system
          --batch_size="$cur_nb"
          --num_batches="$MUL"
          --properties_to_condition_on="{'chemical_system':'${SYS}'}"
          --record_trajectories=False
          --diffusion_guidance_factor="$DGF"
          --guidance="$(printf "{'%s': %s}" "$GUIDTYPE" "$GUIDPARAM")"
          --diffusion_loss_weight="[$G,$K,$NORM]"
          --print_loss=False
          --self_rec_steps="$R"
          --back_step="$B"
          --algo="$ALG"
          -f "$FORCE_GPU"
    )

    if [[ -n "$GPU_MEM_GB" ]]; then
      cmd+=( --gpu_memory_gb="$GPU_MEM_GB" )
    fi
    if [[ -n "$FRAC" ]]; then
      cmd+=( -f "$FRAC" )
    fi
    if [[ ${#EXTRA_ARR[@]} -gt 0 ]]; then
      cmd+=( "${EXTRA_ARR[@]}" )
    fi

    {
      printf '\n# --- Run %d (attempt %d, batch_size=%s) ---\n' "$i" "$attempt" "$cur_nb"
      printf '(%s)\n' "$(date -Is)"
      printf 'CMD: '; printf '%q ' "${cmd[@]}"; printf '\n'
    } >> "$LOGFILE"

    echo ">>> [Run $i/$RUNS | Attempt $attempt] Launching with batch_size=$cur_nb → $RUN_PATH"
    set +e
    "${cmd[@]}" >> "$LOGFILE" 2>&1
    status=$?
    set -e

    if [[ $status -eq 0 ]]; then
      echo ">>> [Run $i] SUCCESS on attempt $attempt (batch_size=$cur_nb)"
      break
    fi

    # Non-zero exit: decide if it's OOM-like
    echo ">>> [Run $i] Non-zero exit ($status). Inspecting log for OOM…"
    oom_hint=false
    # Exit code 137 is often OOM-kill; 9 (SIGKILL) can be as well.
    if [[ $status -eq 137 || $status -eq 9 || $status -eq 143 ]]; then
      oom_hint=true
    elif detect_oom "$LOGFILE"; then
      oom_hint=true
    fi

    if $oom_hint; then
      if [[ $attempt -gt $OOM_RETRIES ]]; then
        echo ">>> [Run $i] OOM persisted after $OOM_RETRIES retries. Giving up. See $LOGFILE" >&2
        exit 1
      fi

      # compute new batch size with integer backoff
      new_nb=$(( (cur_nb * OOM_BACKOFF_PCT + 99) / 100 ))  # ceiling division
      if [[ $new_nb -lt $MIN_BATCH_SIZE ]]; then
        echo ">>> [Run $i] New batch_size=$new_nb would be < MIN_BATCH_SIZE=$MIN_BATCH_SIZE. Aborting." >&2
        exit 1
      fi
      if [[ $new_nb -ge $cur_nb ]]; then
        new_nb=$((cur_nb - 1))
        if [[ $new_nb -lt $MIN_BATCH_SIZE ]]; then
          echo ">>> [Run $i] Cannot reduce batch_size further (cur=$cur_nb, min=$MIN_BATCH_SIZE). Aborting." >&2
          exit 1
        fi
      fi

      echo ">>> [Run $i] OOM detected. Backing off batch_size: $cur_nb → $new_nb. Cooling down ${WAIT_SEC}s…"
      sleep "$WAIT_SEC"
      cur_nb="$new_nb"
      continue
    else
      echo ">>> [Run $i] Failure appears non-OOM. Stop. See $LOGFILE" >&2
      exit 1
    fi
  done
done

echo ">>> All runs completed. Output in: $DIR"
echo ">>> Log at: $LOGFILE"
