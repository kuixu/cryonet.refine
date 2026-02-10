#!/bin/bash

input_pdb_path=$1
target_density=$2
resolution=$3
out_dir=$4

if [ ! -d "$out_dir" ]; then 
    mkdir -p $out_dir
fi  

max_tokens=1000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Set PYTHONPATH to include project root so CryoNetRefine package can be found
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
checkpoint="${SCRIPT_DIR}/params/CryoNet.Refine_model.pt"
echo "Starting CryoNet.Refine..."
echo "Input: $input_pdb_path"
echo "Target density: $target_density"
echo "Resolution: $resolution"
echo "Output: $out_dir"
echo "Checkpoint: $checkpoint"
echo "Max tokens: $max_tokens"

CUDA_VISIBLE_DEVICES=0 python main.py \
    $input_pdb_path \
    --target_density $target_density \
    --resolution $resolution \
    --out_dir $out_dir \
    --out_suffix CryoNet.Refine \
    --checkpoint $checkpoint \
    --max_tokens $max_tokens \
    # --validate_output \
 
echo "CryoNet.Refine refinement completed!"

