#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
SCRIPT="scripts/run_infer.py"
EVAL_SCRIPT="scripts/run_eval.py"
VIS_SCRIPT="scripts/run_vis.py"

# ----- model ------
# source ./scripts/config/eth3d/512x336.sh

# MODEL="reliev3r" # ['vggt','pi3','reliev3r']
# MODEL_SIZE=""
# CKPT="experiments/Reliev3R-3B_260525_subv6ckpt79_womix_puresub/ckpt/15000"
# CONFIG="experiments/Reliev3R-3B_260525_subv6ckpt79_womix_puresub/config.json"
# PREFIX="Reliev3R-3B_260525_subv6ckpt79_womix_puresub_ckpt15000"

# MODEL="reliev3r" # ['vggt','pi3','reliev3r']
# MODEL_SIZE=""
# CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0513_pi3_3B_datamix_all_v6_lowres_v8-m8_randres_stage2_lr5e-5/ckpts/model_ckpt79"
# CONFIG="experiments/Reliev3R-3B_260522_subPi3-3B_v6ckpt79/config.json"
# PREFIX="Reliev3R-3B_260522_subPi3-3B_v6ckpt79_ckpt0"

# MODEL="pi3" # ['vggt','pi3','reliev3r']
# MODEL_SIZE="dinov3_huge+_3Bv2"
# CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0521_pi3_3B_datamix_all_v9-1_lowres_v8-m8_randres_stage3_preexp02/ckpts/model_ckpt3" 
# PREFIX="exp0521_pi3_3B_datamix_all_v9-1_lowres_v8-m8_randres_stage3_preexp02_ckpt3"
# CONFIG=""

source ./scripts/config/eth3d/224x224.sh
MODEL="pi3" # ['vggt','pi3','reliev3r']
MODEL_SIZE="dinov3_huge+_3Bv2"
CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0428_pi3_3B_datamix_all_v5_lowres_v8-m8_224x224_lr1e4_scratch/ckpts/model_ckpt79" 
PREFIX="exp0428_pi3_3B_datamix_all_v5_lowres_v8-m8_224x224_lr1e4_scratch_ckpt79"
CONFIG=""

# MODEL="pi3" # ['vggt','pi3','reliev3r']
# MODEL_SIZE="dinov2_large_0.95B"
# CKPT="./ckpts/Pi3/model.safetensors"
# PREFIX="pi3_official"
# CONFIG=""

NPU_ID="4"
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
