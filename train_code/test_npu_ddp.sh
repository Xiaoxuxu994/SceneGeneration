# ASCEND_RT_VISIBLE_DEVICES=0 python test_npu_ctf_new_metric.py \
#     --config configs/val/exp0620_3D_geometry_vae_normv6_ctf_4level_debug.py

# ASCEND_RT_VISIBLE_DEVICES=0 python test_npu_ctf_new_metric_KL_sdf_singlevs.py \
#     --config configs/val/exp0725_4en_8de_KL64_norm_sdf_baseline_1e-4_w10000_tsdf.py

# ASCEND_RT_VISIBLE_DEVICES=0 python test_npu_ctf_new_metric_KL_sdf_singlevs.py \
#     --config configs/val/exp0722_4en_8de_KL64_norm_sdf_baseline_1e-4_w100_tsdf.py

ASCEND_RT_VISIBLE_DEVICES=0 python test_npu_ctf_new_metric_KL_sdf_singlevs.py \
    --config configs/val/exp0723_4en_8de_KL64_norm_sdf_baseline_1e-4_w10000_tsdf_vs256.py