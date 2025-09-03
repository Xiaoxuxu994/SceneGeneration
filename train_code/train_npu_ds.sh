# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29509 --num_processes 4 train_npu_ctf_new_metric_KL_sdf_singlevs_ds.py --config configs/train/exp0815_4en_8de_KL64_norm_sdf_vs256_1e-4_w10000_tsdf_ds.py

# export ASCEND_RT_VISIBLE_DEVICES=0,1
# accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29509 --num_processes 2 train_npu_ctf_new_metric_KL_sdfgrad_singlevs_ds.py --config configs/train/exp0830_cafe_debug_sdf_grad.py


export ASCEND_RT_VISIBLE_DEVICES=0,1
accelerate launch --config_file ./accelerate_config.yaml --main_process_port 29509 --num_processes 2 train_npu_ctf_new_metric_KL_sdfgradsign_singlevs_ds.py --config configs/train/exp0901_cafe_debug_sdf_grad_sign.py
