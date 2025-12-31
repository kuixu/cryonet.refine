#!/bin/bash

input_pdb_path=$1
target_density1=$2
target_density2=$3
resolution1=$4
resolution2=$5
out_dir=$6

if [ ! -d "$out_dir" ]; then 
    mkdir -p $out_dir
fi  

max_tokens=1000
num_recycles=300
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# checkpoint="${SCRIPT_DIR}/params/cryonet.refine_model_checkpoint_best26.pt"
checkpoint="/home/huangfuyao/proj/cryonet.refine/params/cryonet.refine_model_checkpoint_best26.pt"
# checkpoint="/home/huangfuyao/data/benchmark_v2/refine_results/train_te43/model_checkpoint_epoch_1.pt"
echo "Input: $input_pdb_path"
echo "Target density: $target_density1, $target_density2"
echo "Resolution: $resolution1, $resolution2"
echo "Output: $out_dir"
echo "Checkpoint: $checkpoint"
echo "Max tokens: $max_tokens"
echo "Number of recycles: $num_recycles"


den=20.0
geometric=1.0
learning_rate=1.8e-4
gamma_0=-0.5
max_norm_sigmas_value=1.0

CUDA_VISIBLE_DEVICES=0 python main.py \
    $input_pdb_path \
    --target_density $target_density1 \
    --target_density $target_density2 \
    --resolution $resolution1 \
    --resolution $resolution2 \
    --out_dir $out_dir \
    --checkpoint $checkpoint \
    --max_tokens $max_tokens \
    --recycles $num_recycles \
    --den $den \
    --gamma_0 $gamma_0 \
    --learning_rate $learning_rate \
    --max_norm_sigmas_value $max_norm_sigmas_value 
echo "CryoNet.Refine refinement completed!"

