
if [[ "$1" == "--help" ]]; then
    echo "Usage:"
    echo "  ./copy.sh MUL BASE DIR"
    echo ""
    echo "Arguments:"
    echo "  MUL  : Number of runs (default: 50)"
    echo "  BASE : Base directory for results (default: /Data/auguste.de-lambilly/mattergenbis/)"
    echo "  DIR  : Output directory prefix (default: results/Li-Co-O_guided_env3_3-2_)"
    echo ""
    echo "Example:"
    echo "  ./copy.sh 50 /Data/auguste.de-lambilly/mattergenbis/ results/Li-Co-O_guided_env3_3-2_"
    exit 0
fi

# Default values for parameters
MUL=${1:-50}
BASE=${2:-/Data/auguste.de-lambilly/mattergenbis/}
SYS=${3:Li-Co-O}
SUF=${4:-_guided_env3_3-2}

main_file="${BASE}results/${SYS}/generated_crystals${SUF}.extxyz"
DIR="results/${SYS}${SUF}_"

for X in $(seq 1 "$MUL"); do
    src="${BASE}${DIR}${X}/generated_crystals.extxyz"
    if [ -f "$src" ]; then
        cat "$src" >> "$main_file"
    else
        echo "Warning: $src does not exist, skipping."
    fi
done

echo "Everything copied to $main_file"