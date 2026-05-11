### DDP Configs
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12358
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
train_mixSup.py \
  --config configs/reliev3r_full_v8_c32_s1_dinov3_3B/main.yaml \
  --train_url /home/z84301856/proj/Relieve3R/exp/reliev3r_full_v8_c32_s1_dinov3_3B \
  --no_load_optim 
  # --iterations 80000
  # --ckpt_path exp/new_pointsup/ckpt/10000.ckpt \
  # --iterations 20000
    # --ckpt_path ckpt/Pi3 \
      # --freeze_backbone \

