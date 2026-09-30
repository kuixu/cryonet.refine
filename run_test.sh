#!/bin/bash
set -eu

# Test script for CryoNet.Refine with recycle=2
# Downloads test data and runs refinement

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/examples"
OUTPUT_DIR="${EXAMPLES_DIR}/output"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Create directories if they don't exist
mkdir -p "${EXAMPLES_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Download and verify test data, falling back to the next configured source.
python -m CryoNetRefine.assets 0775_af3.cif "${EXAMPLES_DIR}/0775_af3.cif"
python -m CryoNetRefine.assets 0775.mrc "${EXAMPLES_DIR}/0775.mrc"

# Set default parameters
RESOLUTION=3.6
RECYCLES=2
MAX_TOKENS=1000

# Input structure file
input_pdb="${EXAMPLES_DIR}/0775_af3.cif"
map_file="${EXAMPLES_DIR}/0775.mrc"
checkpoint="${SCRIPT_DIR}/params/CryoNet.Refine_model.pt"

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
    --checkpoint "${checkpoint}" \
    --target_density "${map_file}" \
    --resolution ${RESOLUTION} \
    --out_dir "${OUTPUT_DIR}" \
    --max_tokens ${MAX_TOKENS} \
    --recycles ${RECYCLES} \
    --validate_output \

echo "CryoNet.Refine test completed!"
