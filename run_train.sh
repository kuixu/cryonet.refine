# 修改 srun 命令，直接启动 torchrun
srun \
  -p priority \
  -J CryoNet.Refine \
  --nodes=2 \
  --ntasks-per-node=8 \
  --gres=gpu:8 \
  --mem=128G \
  --qos=priority_qos \
  bash -c '
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_STEP_NODELIST | head -n 1)
    export MASTER_PORT=$((29500 + SLURM_JOB_ID % 10000))
    export NODE_RANK=$SLURM_NODEID
    if [ "$SLURM_LOCALID" = "0" ]; then
        torchrun \
            --nnodes=$SLURM_NNODES \
            --nproc_per_node=$SLURM_NTASKS_PER_NODE \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            train_v2.py \
            /data/huangfuyao/AF3DB_pre_train/AFDB_aligned/ \
            --map_db_path /data/huangfuyao/AF3DB_pre_train/EMD_cropped \
            --out_dir /data/huangfuyao/AF3DB_pre_train/checkpoint_test1 \
            --checkpoint /home/huangfuyao/proj/cryonet.refine/params/boltz2_conf_clean.ckpt \
            --file_list /data/huangfuyao/AF3DB_pre_train/cc_mask_ge02_ids.list \
            --num_epochs 100 \
            --recycles 10 \
            --epoch_early_stop_patience 10 \
            --resume False \
            --den 20.0 \
            --use_global_clash \
            --max_tokens 1000
    else
        wait
    fi
  '