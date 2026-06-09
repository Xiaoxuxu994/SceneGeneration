export HCCL_BUFFSIZE=120
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_OP_STATS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export USE_CHECKPOINT=True
export ASCEND_RT_VISIBLE_DEVICES=5,6,7
export ASCEND_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export pred_z_depth=True
torchrun \
    --standalone \
    --nproc_per_node npu \
    train_ds.py \
    --train_url outputs/pi3 \
    --config configs/pi3_dinov3_5Bv3/debug.yaml \
    --ckpt_path /efs-m3ga/c50041401/exp/pi3sup/outputs/exp0505_pi3_5B_datamix_all_v5_lowres_v8-m8_randres_stage2_lr5e-5/ckpts/checkpoint_39/ \
    --no_load_optim \
    --encoder_lr_ratio 0.01 \
    --decoder_lr_ratio 0.1 \
    --deepspeed \
    --deepspeed_config ./ds_config/ds_config_z2_ot.json \
    # --ckpt_path /efs-m3ga/c50041401/exp/pi3sup/outputs/exp0605_pi3_5B_datamix_all_v9-1_lowres_v8-m8_randres_stage2_lr5e-5_resume0601/ckpts/checkpoint_37/ \
    # --strict_load \
    # --ckpt_id 20000 \
