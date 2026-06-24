#!/bin/bash
set -eo pipefail


j=${1:?Usage: $0 <jobid>}

d=/data1/jobs/$j
name=$(cat $d/name.list)
out_dir=$d/${name}
map=${out_dir}.mrc
stg=$d/status
restraints_file=$d/user_restraints.json
cif_filename=$(jq -r '.pdbfile_local | split("/")[-1]' $stg)
cif=${d}/${cif_filename}

# tem=${out_dir}_aem0.pdb
# out=${out_dir}_CryoNet.Refine.pdb
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

max_tokens=${CRYONET_REFINE_MAX_TOKENS:-1000}
recycles=${CRYONET_REFINE_RECYCLES:-300}
restraint_flags=()
if [ -f "$restraints_file" ]; then
    restraint_flags+=(--use_user_restraints --restraints_file "$restraints_file")
fi
global_clash_flags=()
if [ "${CRYONET_REFINE_DISABLE_GLOBAL_CLASH:-0}" = "1" ]; then
    global_clash_flags+=(--no-use_global_clash)
fi
validation_flags=()
if [ "${CRYONET_REFINE_VALIDATE_OUTPUT:-1}" = "1" ]; then
    validation_flags+=(--validate_output)
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Starting CryoNet.Refine..."
echo "Input model   : $cif"
echo "Target density: $map"
echo "Resolution: $res"
echo "Output: $out_dir"
echo "Checkpoint: $checkpoint"
echo "Max tokens: $max_tokens"
echo "Recycles: $recycles"
echo "Validate output: ${CRYONET_REFINE_VALIDATE_OUTPUT:-1}"

refine_python=${CRYONET_REFINE_PYTHON:-python}

CUDA_VISIBLE_DEVICES=0 "$refine_python" main.py \
    $cif \
    --target_density $map \
    --resolution $res \
    --out_dir $d \
    --out_suffix CryoNet.Refine \
    --max_tokens $max_tokens \
    --recycles $recycles \
    "${validation_flags[@]}" \
    "${global_clash_flags[@]}" \
    "${restraint_flags[@]}"
 
echo "CryoNet.Refine refinement completed!"

date >> $log
