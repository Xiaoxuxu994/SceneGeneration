#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
SCRIPT="scripts/run_infer.py"
EVAL_SCRIPT="scripts/run_eval.py"
VIS_SCRIPT="scripts/run_vis.py"

# ----- model ------
source ./scripts/config/re10k/512x288.sh

# MODEL="reliev3r" # ['vggt','pi3','reliev3r']
# MODEL_SIZE=""
# CKPT="experiments/VGGT-3B_260603_sub10k/ckpt/100000"
# CONFIG="experiments/VGGT-3B_260603_sub10k/config.json"
# PREFIX="VGGT-3B_260603_sub10k_ckpt100000"

# MODEL="reliev3r" # ['vggt','pi3','reliev3r']
# MODEL_SIZE=""
# CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0513_pi3_3B_datamix_all_v6_lowres_v8-m8_randres_stage2_lr5e-5/ckpts/model_ckpt79"
# CONFIG="experiments/Reliev3R-3B_260525_subv6ckpt79_womix_puresub/config.json"
# PREFIX="Reliev3R-3B_260525_subv6ckpt79_womix_puresub_ckpt0"

MODEL="pi3" # ['vggt','pi3','reliev3r']
MODEL_SIZE="dinov3_huge+_3Bv2" #"dinov3_huge+_5Bv2" #"dinov3_huge+_3Bv2"
CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0604_pi3_3B_datamix_all_v9-5_randresNew_scratch_uploss_pinmemTrue/ckpts/model_ckpt78" 
PREFIX="exp0604_pi3_3B_datamix_all_v9-5_randresNew_scratch_uploss_pinmemTrue_ckpt78"
CONFIG=""

# MODEL="pi3" # ['vggt','pi3','reliev3r']
# MODEL_SIZE="dinov3_huge+_5Bv2" #"dinov3_huge+_5Bv2" #"dinov3_huge+_3Bv2"
# CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0505_pi3_5B_datamix_all_v5_lowres_v8-m8_randres_stage2_lr5e-5/ckpts/model_ckpt39" 
# PREFIX="exp0505_pi3_5B_datamix_all_v5_lowres_v8-m8_randres_stage2_lr5e-5_ckpt39"
# CONFIG=""

# MODEL="pi3" # ['vggt','pi3','reliev3r']
# MODEL_SIZE="dinov3_huge+_3Bv2"
# CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0513_pi3_3B_datamix_all_v6_lowres_v8-m8_randres_stage2_lr5e-5/ckpts/model_ckpt79"
# CONFIG=""
# PREFIX="exp0513_pi3_3B_datamix_all_v6_lowres_v8-m8_randres_stage2_lr5e-5_ckpt79"

# MODEL="pi3" # ['vggt','pi3','reliev3r']
# MODEL_SIZE="dinov3_huge+_3Bv2"
# CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0521_pi3_3B_datamix_all_v9-1_lowres_v8-m8_randres_stage3_preexp02/ckpts/model_ckpt3" 
# PREFIX="exp0521_pi3_3B_datamix_all_v9-1_lowres_v8-m8_randres_stage3_preexp02_ckpt3"
# CONFIG=""

# MODEL="vggt_omega"
# CKPT="/efs-mi-east4-2/w60114355/checkpoints/modelscope.cn/models/sanqilu376/VGGT-Omega/resolve/master/vggt_omega_1b_512.pt"
# PREFIX="vggt_omega_1B_official"
# CONFIG=""
# MODEL_SIZE=""

## ------ 224x224 --------
# source ./scripts/config/re10k/224x224.sh

# MODEL="reliev3r" # ['vggt','pi3','reliev3r']
# MODEL_SIZE=""
# CKPT="experiments/VGGT-3B_260603_sub10k/ckpt/100000"
# CONFIG="experiments/VGGT-3B_260603_sub10k/config.json"
# PREFIX="VGGT-3B_260603_sub10k_ckpt100000"

# MODEL="pi3" # ['vggt','pi3','reliev3r']
# MODEL_SIZE="dinov2_large_0.95B"
# CKPT="./ckpts/Pi3/model.safetensors"
# PREFIX="pi3_official"
# CONFIG=""

# MODEL="vggt_omega"
# CKPT="/efs-mi-east4-2/w60114355/checkpoints/modelscope.cn/models/sanqilu376/VGGT-Omega/resolve/master/vggt_omega_1b_512.pt"
# PREFIX="vggt_omega_1B_official"
# CONFIG=""
# MODEL_SIZE=""

NPU_ID="1"
DEVICE="npu:0"
DTYPE="fp32"

OUT_ROOT="outputs"  # "/mnt/nvme3/z00533370/3drecon/benchmark/"

# ---- eval common params ----
POSE_THRESHOLD=30
DEPTH_ALIGN="scale_shift"
EVAL_DEVICE="${DEVICE}"

# ---- vis params ----
DEPTH_VMIN=0.1
DEPTH_VMAX=10.0

# ---- behavior ----
REUSE_PRED_IF_EXISTS=1   # 1: if pred exists, skip infer; 0: always run infer

USE_GT_POSE=0 # 1 if using GT extrinsics; 0 predicted
USE_GT_K=0 # 1 if using GT intrinsics; 0 predicted

# ---- run benchmark ----
source ./scripts/lib/engine.sh
