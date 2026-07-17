### Exp Configs
## Experiment Name
EXPNAME=debug
export USE_CHECKPOINT=True
export HYDRA_FULL_ERROR=1
export pred_z_depth=True
## CUDA Visible Devices
# export CUDA_VISIBLE_DEVICES="0"
## NPU Visible Devices
export ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7
export HCCL_BUFFSIZE=120
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_OP_STATS=1
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

### Command
torchrun \
    --standalone \
    --nproc_per_node npu \
    train_ds.py \
    --config configs/pi3_dinov3/main.yaml \
    --deepspeed \
    --deepspeed_config ds_config/ds_config_z2_ot.json
