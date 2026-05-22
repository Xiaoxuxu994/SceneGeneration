export HCCL_BUFFSIZE=120
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_OP_STATS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export USE_CHECKPOINT=True
export ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_LAUNCH_BLOCKING=1
export pred_z_depth=True
torchrun \
    --standalone \
    --nproc_per_node npu \
    train_ds.py \
    --train_url outputs/pi3 \
    --config configs/pi3_dinov2/main.yaml \
    --ckpt_path ckpt/Pi3 \
    --no_load_optim \
    --iterations 300000 \
    --deepspeed \
    --deepspeed_config ./ds_config/ds_config_z2_ot.json \
    # --backbone_lr_ratio 0.01 \