#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
SCRIPT="scripts/run_infer.py"
EVAL_SCRIPT="scripts/run_eval.py"
VIS_SCRIPT="scripts/run_vis.py"

# ----- model ------
source ./scripts/config/7scenes/518x392.sh

# MODEL="reliev3r" # ['vggt','pi3','reliev3r']
# MODEL_SIZE=""
# CKPT="experiments/Reliev3R-260421-mixSupervise-lr1e-5-All-SVallSynthReal-Pi3-freeze-newcode-addArkit/ckpt/60000.ckpt"
# CONFIG="experiments/Reliev3R-260421-mixSupervise-lr1e-5-All-SVallSynthReal-Pi3-freeze-newcode-addArkit/config.json"
# PREFIX="Reliev3R-260421-mixSupervise-lr1e-5-All-SVallSynthReal-Pi3-freeze-newcode-addArkit_ckpt60k"

# MODEL="pi3" # ['vggt','pi3','reliev3r']
# MODEL_SIZE="dinov3_huge+_3Bv2"
# CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0503_pi3_3B_datamix_all_v5_lowres_v8-m8_randres_stage2_lr5e-5/ckpts/model_ckpt39" 
# PREFIX="exp0503_pi3_3B_datamix_all_v5_lowres_v8-m8_randres_stage2_lr5e-5_ckpt39"
# CONFIG=""

# MODEL_SIZE="dinov2_large_0.95B"
# CKPT="./ckpts/Pi3/model.safetensors"
# PREFIX="pi3_official"
# CONFIG=""
# source ./scripts/config/7scenes/518x392.sh

MODEL="vggt" # ['vggt','pi3','reliev3r']

MODEL_SIZE=""
CONFIG="./ckpts/VGGT-1B/config.json"
CKPT="./ckpts/VGGT-1B/model.pt"
PREFIX="vggt_official"
source ./scripts/config/7scenes/518x392.sh

# MODEL="vggt" # ['vggt','pi3','reliev3r']
# CONFIG=""
# MODEL_SIZE=""
# CKPT="/mnt/nvme2/share/chenyouyu/codespace/vggt_sz_ft/epoch021-loss_valid0.1170-metric_valid0.0528.ckpt/pytorch_model.bin"
# PREFIX="vggt_sz_final"

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
