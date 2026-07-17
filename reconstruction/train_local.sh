export HCCL_BUFFSIZE=120
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_OP_STATS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export USE_CHECKPOINT=True
export ASCEND_RT_VISIBLE_DEVICES=6
export HYDRA_FULL_ERROR=1
export pred_z_depth=True
torchrun \
    --standalone \
    --nproc_per_node npu \
    train_ds.py \
    --train_url outputs/toy_gtintr \
    --config configs/toy/stage3.yaml \
    --learning_rate 1e-2 \
    --deepspeed \
    --deepspeed_config ./ds_config/ds_config_z2_ot.json \
    # --no_load_optim \
    # --no_load_sched \
    # --ckpt_path /efs-m3ga/c50041401/exp/pi3sup/outputs/exp0608_pi3_5Bv2_datamix_all_v9-5_randres_stage2_uploss/ckpts/checkpoint_39 \
    # --ckpt_path /efs-m3ga/y00959808/exp/reliev3r/0708-Pi3-5Bv2 \
    # --deepspeed_config configs/pi3_dinov3_stage1/ds_config_z2_ot.json \
    # --no_load_optim \
    # --ckpt_id 50000 \