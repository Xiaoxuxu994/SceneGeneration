### Exp Configs
## Experiment Name
EXPNAME=debug
## CUDA Visible Devices
# export CUDA_VISIBLE_DEVICES="0"
## NPU Visible Devices
# export NPU_VISIBLE_DEVICES="0,1"
# export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_RT_VISIBLE_DEVICES=6
# export LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

### Command
torchrun --standalone --nproc_per_node npu train.py \
  --config configs/reliev3r_anchorsplat.yaml \
  --ckpt_path /efs-mi-east4-3/c50041401/exp/ssff3d/Reliev3R_bs2v8_260330_ase_ss_lr3e-5_distl1_sigT/ckpt/130000.ckpt \
  --no_load_optim
  # --iterations 20000
