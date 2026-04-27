#!/bin/bash

input_pdb_path=$1
target_density=$2
resolution=$3
out_dir=$4
restraints_file=$5

phenix_env="/opt/phenix-1.21.1-5286/phenix_env.sh"
chimerax_cmd="/usr/bin/chimerax"

if [ -z "$input_pdb_path" ] || [ -z "$target_density" ] || [ -z "$resolution" ] || [ -z "$out_dir" ]; then
    echo "Usage: $0 <input_pdb_path> <target_density> <resolution> <out_dir> [restraints_file]"
    exit 1
fi

if [ ! -d "$out_dir" ]; then 
    mkdir -p "$out_dir"
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
if [ -n "$restraints_file" ]; then
    echo "User restraints: $restraints_file"
fi
echo "Max tokens: $max_tokens"
echo "Phenix env: $phenix_env"
echo "ChimeraX cmd: $chimerax_cmd"

export PHENIX_ENV="$phenix_env"
export CHIMERAX_CMD="$chimerax_cmd"

restraint_flags=()
if [ -n "$restraints_file" ]; then
    restraint_flags+=(--use_user_restraints --restraints_file "$restraints_file")
fi

CUDA_VISIBLE_DEVICES=0 python main.py \
    "$input_pdb_path" \
    --target_density "$target_density" \
    --resolution "$resolution" \
    --out_dir "$out_dir" \
    --out_suffix CryoNet.Refine \
    --max_tokens "$max_tokens" \
    "${restraint_flags[@]}" \
    # --validate_output

echo "CryoNet.Refine refinement completed!"

