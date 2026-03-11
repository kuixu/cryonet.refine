#!/bin/bash

input_pdb_path=$1
target_density=$2
resolution=$3
out_dir=$4

phenix_env="/opt/phenix-1.21.1-5286/phenix_env.sh"
chimerax_cmd="/usr/bin/chimerax"

if [ -z "$phenix_env" ] || [ -z "$chimerax_cmd" ]; then
    echo "Usage: $0 <input_pdb_path> <target_density> <resolution> <out_dir> <phenix_env> <chimerax_cmd>"
    exit 1
fi

if [ ! -d "$out_dir" ]; then 
    mkdir -p $out_dir
fi  

max_tokens=1000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Set PYTHONPATH to include project root so CryoNetRefine package can be found
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
echo "Starting CryoNet.Refine..."
echo "Input: $input_pdb_path"
echo "Target density: $target_density"
echo "Resolution: $resolution"
echo "Output: $out_dir"
echo "Max tokens: $max_tokens"
echo "Phenix env: $phenix_env"
echo "ChimeraX cmd: $chimerax_cmd"

export PHENIX_ENV="$phenix_env"
export CHIMERAX_CMD="$chimerax_cmd"

CUDA_VISIBLE_DEVICES=0 python main.py \
    $input_pdb_path \
    --target_density $target_density \
    --resolution $resolution \
    --out_dir $out_dir \
    --out_suffix CryoNet.Refine \
    --max_tokens $max_tokens \
    # --validate_output \
 
echo "CryoNet.Refine refinement completed!"

