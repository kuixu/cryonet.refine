#!/bin/bash

dataset_dir="/data/huangfuyao/AF3DB_pre_train/AFDB_aligned_test/"

# output_dir="/home/huangfuyao/data/benchmark/modelangelo_results/proteins/train_results_n26_wo_den"
output_dir="/data/huangfuyao/AF3DB_pre_train/checkpoint_test"
map_db_path="/data/huangfuyao/AF3DB_pre_train/EMD_cropped_test"

# checkpoint="/home/huangfuyao/proj/cryonet.refine/params/cryonet.refine_model_checkpoint_best26.pt"
# checkpoint="/home/huangfuyao/data/benchmark_v2/refine_results/train_te42_debug/model_checkpoint_best.pt"
checkpoint="/home/huangfuyao/proj/cryonet.refine/params/boltz2_conf_clean.ckpt"

num_epochs=5
epoch_early_stop_patience=10
resume=False
recycles=10
den=20.0

# Number of GPUs to use
NUM_GPUS=3

# Use torchrun for distributed training
CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_v2.py \
    $dataset_dir \
    --map_db_path $map_db_path \
    --out_dir $output_dir \
    --checkpoint $checkpoint \
    --num_epochs $num_epochs \
    --recycles $recycles \
    --epoch_early_stop_patience $epoch_early_stop_patience \
    --resume $resume \
    --den $den 
    | tee $output_dir/train.log
