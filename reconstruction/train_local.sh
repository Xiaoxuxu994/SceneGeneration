### DDP Configs
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12356
export WORLD_SIZE=2
export NODE_RANK=0

### Exp Configs
## Experiment Name
EXPNAME=debug
## CUDA Visible Devices
# export CUDA_VISIBLE_DEVICES="0"
## NPU Visible Devices
# export NPU_VISIBLE_DEVICES="0,1"
export ASCEND_RT_VISIBLE_DEVICES="2,3"
export pred_z_depth=True

### Command
torchrun --nnodes=1 --nproc_per_node=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
train_ds.py --config configs/reliev3r_full_v8_c32_s1_dinov3/main_single_new.yaml \
--iterations 19 --deepspeed --deepspeed_config ./ds_config/ds_config_z2_ot.json
  # --ckpt_path exp/new_pointsup/ckpt/10000.ckpt \
  # --iterations 20000
