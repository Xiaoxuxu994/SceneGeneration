### DDP Configs
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12357
export WORLD_SIZE=1
export NODE_RANK=0

### Exp Configs
## Experiment Name
EXPNAME=debug
## CUDA Visible Devices
export CUDA_VISIBLE_DEVICES="0"
## NPU Visible Devices
# export NPU_VISIBLE_DEVICES="0,1"
# export ASCEND_RT_VISIBLE_DEVICES="0,1"

### Command
torchrun --nnodes=1 --nproc_per_node=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
train.py \
  --config configs/reliev3r_overfit/main.yaml #\
  # --ckpt_path exp/debug_simpleLoss/ckpt/5000.ckpt \
  # --iterations 6000
