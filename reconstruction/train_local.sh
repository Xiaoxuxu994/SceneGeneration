export HCCL_BUFFSIZE=120
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_OP_STATS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export USE_CHECKPOINT=True
export ASCEND_RT_VISIBLE_DEVICES=5,6
# export HYDRA_FULL_ERROR=1
export pred_z_depth=True

### Command
# # stage2
# torchrun \
#     --standalone \
#     --nproc_per_node npu \
#     train_ds.py \
#     --train_url outputs/pi3_dinov2_stage2 \
#     --config configs/pi3_dinov2_stage2/debug.yaml \
#     --ckpt_path /home/z84301856/proj/Relieve3R/SceneGeneration/reconstruction/ckpt/Pi3/model.safetensors \
#     --lr_scheduler cosine \
#     --learning_rate 1e-5 \
#     --no_load_optim \
#     --not_resume_step \
#     --deepspeed \
#     --deepspeed_config ./ds_config/ds_config_z2_ot.json 


# stage3
torchrun \
    --standalone \
    --nproc_per_node npu \
    train_ds.py \
    --train_url outputs/pi3_dinov2_stage3 \
    --config configs/pi3_dinov2_stage3/debug.yaml \
    --ckpt_path /efs-m3ga/c50041401/exp/ssff3d/Reliev3R-260608-1B-truePi3-baseline/ckpt \
    --ckpt_id 40000 \
    --lr_scheduler cosine \
    --learning_rate 1e-5 \
    --no_load_optim \
    --deepspeed \
    --deepspeed_config ./ds_config/ds_config_z2_ot.json 

