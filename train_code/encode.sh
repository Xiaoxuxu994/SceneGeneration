ASCEND_RT_VISIBLE_DEVICES=7 nohup python encode_pcd_to_latent.py \
   --config configs/exp0701_3D_geometry_vae_encode.py > latent-val.log 2>&1 &