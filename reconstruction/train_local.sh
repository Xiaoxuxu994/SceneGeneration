### Exp Configs
## Experiment Name
EXPNAME=debug
## CUDA Visible Devices
# export CUDA_VISIBLE_DEVICES="0"
## NPU Visible Devices
export ASCEND_RT_VISIBLE_DEVICES=7
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

### Command
torchrun --standalone --nproc_per_node npu \
  train.py \
  --config configs/reliev3r_anchorsplat.yaml \
  --ckpt_path /efs-mi-east4-3/c50041401/exp/ssff3d/Reliev3R_bs2v8_260330_ase_ss_lr3e-5_distl1_sigT/ckpt/130000.ckpt \
  --no_load_optim \
  --train_url ./outputs/$EXPNAME \
  --verbose_interval 1 \
  # --iterations 20000
