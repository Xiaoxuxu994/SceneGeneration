#ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 --master_addr="127.0.0.1" --master_port=29500 train_npu.py \
#     --config configs/exp0612_3D_geometry_vae.py

ASCEND_RT_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_addr="127.0.0.2" --master_port=29550 train_npu.py \
   --config configs/exp0612_3D_geometry_vae_sbms.py
