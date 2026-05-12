### DDP Configs
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12358
export WORLD_SIZE=1
export NODE_RANK=0

## CUDA Visible Devices
# export CUDA_VISIBLE_DEVICES="3"
## NPU Visible Devices
export NPU_VISIBLE_DEVICES="3"
export ASCEND_RT_VISIBLE_DEVICES="3"

### Command
# torchrun --nnodes=1 --nproc_per_node=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
# train_mixSup.py \
#   --config configs/reliev3r_full_v8_c32_s1_dinov2_1B_camRegFalse/main.yaml \
#   --train_url /home/z84301856/proj/Relieve3R/exp/reliev3r_full_v8_c32_s1_dinov2_1B_camRegFalse \
#   --ckpt_path /efs-m3ga/c50041401/exp/ssff3d/Reliev3R-260421-mixSupervise-lr1e-5-All-SVallSynthReal-Pi3-freeze-newcode-addArkit/ckpt/60000.ckpt \
#   --freeze_backbone \
#   --no_load_optim 
#   # --iterations 20000
#   # --ckpt_path ckpt/Pi3 \


torchrun --nnodes=1 --nproc_per_node=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
train_ds_mixSup.py \
  --config configs/reliev3r_full_v8_c32_s1_dinov2_1B_camRegFalse/main.yaml \
  --train_url /home/z84301856/proj/Relieve3R/exp/reliev3r_full_v8_c32_s1_dinov2_1B_camRegFalse \
  --ckpt_path /efs-m3ga/c50041401/exp/ssff3d/Reliev3R-260421-mixSupervise-lr1e-5-All-SVallSynthReal-Pi3-freeze-newcode-addArkit/ckpt/60000.ckpt \
  --freeze_backbone \
  --no_load_optim \
  --deepspeed \
  --deepspeed_config ./ds_config/ds_config_z2_ot.json
  # --iterations 20000
  # --ckpt_path ckpt/Pi3 \