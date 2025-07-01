ASCEND_RT_VISIBLE_DEVICES=5 nohup python val_npu.py \
   --config configs/exp0701_3D_geometry_vae_releaseV1_val.py > val.log 2>&1 &