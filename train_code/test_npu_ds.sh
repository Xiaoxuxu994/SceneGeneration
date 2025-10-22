# ASCEND_RT_VISIBLE_DEVICES=6 python test_npu_ctf_new_metric_KL_sdf_singlevs_ds.py \
#     --config configs/val/exp0822_004_4en_8de_KL64_norm_sdf_vs512_vsgd128_baseline_lr1e-4_w1e4_kl1e-3_32npu_8workers_sdfonly_lrschedulernone_ds_s3.py

# ASCEND_RT_VISIBLE_DEVICES=0 python test_npu_ctf_new_metric_KL_sdf_singlevs_ds.py \
#     --config configs/val/vs256_best_zt.py
#     --config configs/val/exp0818_010_4en_8de_KL64_norm_sdf_vs512_vsgd256_baseline_lr1e-4_w1e4_kl1e-3_8npu_8workers_sdfonly_lrschedulernone_ds_s3.py
    
        # --config configs/val/vs256_best_zt.py
# ASCEND_RT_VISIBLE_DEVICES=6 python test_npu_ctf_new_metric_KL_sdf_256-512_eval_ds.py \
#     --config configs/val/exp0911_3D_geometry_vae_ctf_vs256_vs512.py


export LD_LIBRARY_PATH=/home/z00533370/proj_3dgs/SceneGeneration_code_npu/SceneGeneration/train_code/src/models:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch_npu/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
# ASCEND_RT_VISIBLE_DEVICES=6 python test_npu_ctf_new_metric_KL_sdf_256-512_eval_ds.py \
#     --config configs/val/exp0911_014_3D_geometry_vae_ctf_vs256_vs512.py

# ASCEND_RT_VISIBLE_DEVICES=7 python test_npu_ctf_new_metric_KL_sdf_256-512_eval_ds.py \
#     --config configs/val/exp0911_018_3D_geometry_vae_ctf_vs256_vs512.py

# ASCEND_RT_VISIBLE_DEVICES=5 python test_npu_ctf_new_metric_KL_sdf_256-512_eval_ds.py \
#     --config configs/val/exp0911_019_3D_geometry_vae_ctf_vs256_vs512.py

# ASCEND_RT_VISIBLE_DEVICES=3 python test_npu_ctf_new_metric_KL_sdf_256-512_eval_ds.py \
#     --config configs/val/exp0911_020_3D_geometry_vae_ctf_vs256_vs512.py


### single level
# ASCEND_RT_VISIBLE_DEVICES=4 python test_npu_ctf_new_metric_KL_sdf_singlevs_ds.py \
#     --config configs/val/exp0829_000_512oct3x_vsgd128_baseline.py

# ASCEND_RT_VISIBLE_DEVICES=0 python test_npu_ctf_new_metric_KL_sdf_singlevs_ds.py \
#     --config configs/val/vs256_best_zt.py

# ASCEND_RT_VISIBLE_DEVICES=7 python test_npu_single_new_metricTM_datactf_eval_ds.py \
#     --config configs/val/vs256_best_zt_datactf.py

# ASCEND_RT_VISIBLE_DEVICES=6 python test_npu_single_new_metricTM_datactf_eval_ds.py \
#     --config configs/val/vs256_vs512_mix_zt_datactf.py

# # # ### data ctf
# ASCEND_RT_VISIBLE_DEVICES=3 python test_npu_single_new_metricTM_datactf_eval_ds.py \
#     --config configs/val/exp1015_008_vs256_vs512_fq1e7_mix_8enc_datactf.py

#ASCEND_RT_VISIBLE_DEVICES=3 python test_npu_single_new_metricTM_datactf_eval_ds.py \
#    --config configs/val/exp1015_009_vs256_vs512_fq1e7_mix_8enc_dim1024_datactf.py

ASCEND_RT_VISIBLE_DEVICES=1 python test_npu_single_new_metricTM_datactf_infer_ds.py \
    --config configs/test/exp1015_008_vs256_vs512_fq1e7_mix_8enc_datactf.py
