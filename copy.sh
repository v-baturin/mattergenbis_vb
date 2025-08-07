
if [[ "$1" == "--help" ]]; then
    echo "Usage:"
    echo "  ./copy.sh MUL BASE DIR"
    echo ""
    echo "Arguments:"
    echo "  MUL  : Number of runs (default: 50)"
    echo "  BASE : Base directory for results (default: /Data/auguste.de-lambilly/mattergenbis/)"
    echo "  SYS  : System to generate (default: Li-Co-O)"
    echo "  SUF  : Suffix for the results directory (default: _guided_env3_3-2)"
    echo ""
    echo "Example:"
    echo "  ./copy.sh 50 /Data/auguste.de-lambilly/mattergenbis/ Li-Co-O _guided_env3_3-2_"
    exit 0
fi

# Default values for parameters
MUL=${1:-50}
BASE=${2:-/Data/auguste.de-lambilly/mattergenbis/}
SYS=${3:Li-Co-O}
SUF=${4:-_guided_env3_3-2}

main_file="${BASE}results/${SYS}_f/generated_crystals${SUF}.extxyz"
hard_save="/users/eleves-b/2021/auguste.de-lambilly/results/${SYS}_f/generated_crystals${SUF}.extxyz"

if [ ! -f "$main_file" ]; then
    echo "Creating main file $main_file."
    if [ ! -d "$(dirname "$main_file")" ]; then
        mkdir -p "$(dirname "$main_file")"
    fi
    touch "$main_file"
fi

if [ ! -f "$hard_save" ]; then
    echo "Creating hard save file $hard_save."
    if [ ! -d "$(dirname "$hard_save")" ]; then
        mkdir -p "$(dirname "$hard_save")"
    fi
    touch "$hard_save"
fi

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