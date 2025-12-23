#!/bin/bash


j=$1

d=/data1/jobs/$j
name=$(cat $d/name.list)
out_dir=$d/${name}
map=${out_dir}.mrc
stg=$d/status

cif_filename=$(jq -r '.pdbfile | split("/")[-1]' $stg)
cif=${d}/${cif_filename}

# tem=${out_dir}_aem0.pdb
out=${out_dir}_CryoNet.Refine.cif
log=${out_dir}.log
# python cryofold.py -m $map -s $seq -t $tem 

res=$(jq '.resolution' $stg)
echo "-m $map -s $cif -r $res"
date >$log;
# CUDA_VISIBLE_DEVICES=0 python cryonet.fold.py -m $map -s $seq -r $res -o $out;



# input_pdb_path=$1
# target_density=$2
# resolution=$3
# out_dir=$4

# if [ ! -d "$out_dir" ]; then 
#     mkdir -p $out_dir
# fi  

max_tokens=1000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
checkpoint="${SCRIPT_DIR}/params/cryonet.refine_model_checkpoint_best26.pt"
echo "Starting CryoNet.Refine..."
echo "Input model   : $cif"
echo "Target density: $map"
echo "Resolution: $res"
echo "Output: $out_dir"
echo "Checkpoint: $checkpoint"
echo "Max tokens: $max_tokens"

CUDA_VISIBLE_DEVICES=0 python main.py \
    $cif \
    --target_density $map \
    --resolution $res \
    --out_dir $d \
    --checkpoint $checkpoint \
    --max_tokens $max_tokens \
 
echo "CryoNet.Refine refinement completed!"

date >> $log
