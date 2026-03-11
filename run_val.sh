#!/bin/bash

# run_val.sh - Run validation for input and output structures using val.py
# This script runs validation twice: once for the input structure, once for the refined output

j=$1
phenix_env="/opt/phenix-1.21.1-5286/phenix_env.sh"
chimerax_cmd="/usr/bin/chimerax"

if [ -z "$j" ] || [ -z "$phenix_env" ] || [ -z "$chimerax_cmd" ]; then
    echo "Usage: $0 <job_id> <phenix_env> <chimerax_cmd>"
    exit 1
fi

d=/data1/jobs/$j
name=$(cat $d/name.list)
out_dir=$d/${name}
map=${out_dir}.mrc
stg=$d/status

cif_filename=$(jq -r '.pdbfile | split("/")[-1]' $stg)
cif=${d}/${cif_filename}

out=${out_dir}_CryoNet.Refine.cif
log=${out_dir}_val.log

res=$(jq '.resolution' $stg)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Starting CryoNet.Refine validation..."
echo "Input model   : $cif"
echo "Target density: $map"
echo "Resolution    : $res"
echo "Output model  : $out"
echo "Log file      : $log"
echo "Phenix env    : $phenix_env"
echo "ChimeraX cmd  : $chimerax_cmd"

date > $log
rm  ${out_dir}.vc ${out_dir}.vcx
rm  ${out_dir}_CryoNet.Refine.vc ${out_dir}_CryoNet.Refine.vcx

# Run validation for both input and output structures using val.py
echo "Running validation..."
CUDA_VISIBLE_DEVICES=0 python val.py \
    "$map" \
    "$cif" \
    "$out" \
    --resolution "$res" \
    --output_dir "$d" \
    --phenix_env "$phenix_env" \
    --chimerax_cmd "$chimerax_cmd" \
    # >> $log 2>&1

# if [ $? -eq 0 ]; then
#     echo "Validation completed successfully!"
# else
#     echo "Validation failed with error code $?"
# fi

# date >> $log

echo "Validation log saved to: $log"
