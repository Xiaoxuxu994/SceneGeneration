# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29509 --num_processes 4 train_npu_ctf_new_metric_KL_sdf_singlevs_ds.py --config configs/train/exp0815_4en_8de_KL64_norm_sdf_vs256_1e-4_w10000_tsdf_ds.py

# export ASCEND_RT_VISIBLE_DEVICES=0,1
# accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29509 --num_processes 2 train_npu_ctf_new_metric_KL_sdfgrad_singlevs_ds.py --config configs/train/exp0830_cafe_debug_sdf_grad.py


# export ASCEND_RT_VISIBLE_DEVICES=0,1
# accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29509 --num_processes 2 train_npu_ctf_new_metric_KL_sdfgradsign_singlevs_ds.py --config configs/train/exp0901_cafe_debug_sdf_grad_sign.py

# export ASCEND_RT_VISIBLE_DEVICES=0,1
# accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29509 --num_processes 2 train_npu_ctf_new_metric_KL_sdfgradsign_singlevs_ds.py --config configs/train/exp0903_cafe_debug_sdf_grad_sign_normalinput.py

# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29509 --num_processes 4 train_npu_ctf_new_metric_KL_sdf_ds.py --config configs/train/exp0911_3D_geometry_vae_ctf_vs256_vs512.py

# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29509 --num_processes 4 train_npu_ctf_new_metric_KL_sdf_ds.py --config configs/train/exp0911_3D_geometry_vae_ctf_vs64_vs512.py


# ## profiling
# export ASCEND_RT_VISIBLE_DEVICES=1,2,3,4
# accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29609 --num_processes 4 train_npu_ctf_new_metric_KL_sdf_ds_profiling.py --config configs/train/exp0911_3D_geometry_vae_ctf_vs256_vs512.py

# ## with fps new kernel
#export LD_LIBRARY_PATH=/home/z00533370/proj_3dgs/SceneGeneration_code_npu/SceneGeneration_dev/train_code/src/models:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch_npu/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=7
# accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29609 --num_processes 1 train_npu_ctf_new_metricTM_KL_sdf_ds.py --config configs/train/exp1013_3D_geometry_vae_ctf_vs256_vs512_overfitting.py
accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29609 --num_processes 1 train_npu_single_new_metricTM_KL_sdf_ds_v2.py --config configs/train/exp1014_mix_debug.py



