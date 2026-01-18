#!/bin/bash

# Test script for CryoNet.Refine with recycle=2
# Downloads test data and runs refinement

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/examples"
OUTPUT_DIR="${EXAMPLES_DIR}/output"

# Create directories if they don't exist
mkdir -p "${EXAMPLES_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Download test data
echo "Downloading test data..."
wget https://cryonet.oss-cn-beijing.aliyuncs.com/cryonet.refine/0775_af3.cif -O "${EXAMPLES_DIR}/0775_af3.cif"
wget https://cryonet.oss-cn-beijing.aliyuncs.com/cryonet.refine/0775.mrc -O "${EXAMPLES_DIR}/0775.mrc"

# Check if downloads were successful
if [ ! -f "${EXAMPLES_DIR}/0775_af3.cif" ]; then
    echo "Error: Failed to download 0775_af3.cif"
    exit 1
fi

if [ ! -f "${EXAMPLES_DIR}/0775.mrc" ]; then
    echo "Error: Failed to download 0775.mrc"
    exit 1
fi

# Set default parameters
RESOLUTION=3.6
RECYCLES=2
MAX_TOKENS=1000

# Set PYTHONPATH to include project root
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Checkpoint path
checkpoint="${SCRIPT_DIR}/params/cryonet.refine_model_checkpoint_best26.pt"

# Input structure file
input_pdb="${EXAMPLES_DIR}/0775_af3.cif"
map_file="${EXAMPLES_DIR}/0775.mrc"

echo "Starting CryoNet.Refine test..."
echo "Input structure: ${input_pdb}"
echo "Target density: ${map_file}"
echo "Resolution: ${RESOLUTION}"
echo "Output: ${OUTPUT_DIR}"
echo "Checkpoint: ${checkpoint}"
echo "Recycles: ${RECYCLES}"
echo "Max tokens: ${MAX_TOKENS}"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/main.py" \
    "${input_pdb}" \
    --target_density "${map_file}" \
    --resolution ${RESOLUTION} \
    --out_dir "${OUTPUT_DIR}" \
    --out_suffix CryoNet.Refine \
    --checkpoint "${checkpoint}" \
    --max_tokens ${MAX_TOKENS} \
    --recycles ${RECYCLES}

echo "CryoNet.Refine test completed!"