#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
SCRIPT="scripts/run_infer.py"
EVAL_SCRIPT="scripts/run_eval.py"
VIS_SCRIPT="scripts/run_vis.py"

# ----- model ------
# source ./scripts/config/scannet/512x384.sh
# source ./scripts/config/scannet/518x392.sh

MODEL="reliev3r" # ['vggt','pi3','reliev3r']
MODEL_SIZE=""
CKPT="experiments/Reliev3R-3B_260525_subv6ckpt79_womix_puresub/ckpt/5000"
CONFIG="experiments/Reliev3R-3B_260525_subv6ckpt79_womix_puresub/config.json"
PREFIX="Reliev3R-3B_260525_subv6ckpt79_womix_puresub_ckpt5000"

# MODEL="reliev3r" # ['vggt','pi3','reliev3r']
# MODEL_SIZE=""
# CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0513_pi3_3B_datamix_all_v6_lowres_v8-m8_randres_stage2_lr5e-5/ckpts/model_ckpt79"
# CONFIG="experiments/Reliev3R-3B_260525_subv6ckpt79_womix_puresub/config.json"
# PREFIX="Reliev3R-3B_260525_subv6ckpt79_womix_puresub_ckpt0"

# MODEL="pi3" # ['vggt','pi3','reliev3r']
# MODEL_SIZE="dinov3_huge+_3Bv2"
# CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0513_pi3_3B_datamix_all_v6_lowres_v8-m8_randres_stage2_lr5e-5/ckpts/model_ckpt79" 
# PREFIX="exp0513_pi3_3B_datamix_all_v6_lowres_v8-m8_randres_stage2_lr5e-5_ckpt79"
# CONFIG=""

## ------ 224x224 --------
source ./scripts/config/scannet/224x224.sh
MODEL="pi3" # ['vggt','pi3','reliev3r']
MODEL_SIZE="dinov3_huge+_5Bv2" #"dinov3_huge+_3Bv2"
CKPT="/efs-m3ga/c50041401/exp/pi3sup/outputs/exp0429_pi3_5B_datamix_all_v5_lowres_v8-m8_224x224_lr1e4_scratch/ckpts/model_ckpt79" 
PREFIX="exp0429_pi3_5B_datamix_all_v5_lowres_v8-m8_224x224_lr1e4_scratch_ckpt79"
CONFIG=""

# MODEL="pi3" # ['vggt','pi3','reliev3r']
# MODEL_SIZE="dinov2_large_0.95B"
# CKPT="./ckpts/Pi3/model.safetensors"
# PREFIX="pi3_official"
# CONFIG=""

# MODEL="pi3" # ['vggt','pi3','reliev3r']
# MODEL_SIZE="dinov3_huge+_3Bv2"
# CKPT="experiments/Reliev3R-3B_260519_subPi3-3B_v6ckpt63/ckpt/5000"
# CONFIG="experiments/Reliev3R-3B_260519_subPi3-3B_v6ckpt63/config.json"
# PREFIX="Reliev3R-3B_260519_subPi3-3B_v6ckpt63_ckpt5000"

MODEL="vggt_omega"
CKPT="/efs-mi-east4-2/w60114355/checkpoints/modelscope.cn/models/sanqilu376/VGGT-Omega/resolve/master/vggt_omega_1b_512.pt"
PREFIX="vggt_omega_1B_official"
CONFIG=""
MODEL_SIZE=""

# MODEL_SIZE="dinov2_large_0.95B"
# CKPT="./ckpts/Pi3/model.safetensors"
# PREFIX="pi3_official"
# CONFIG=""
# source ./scripts/config/scannet/518x392.sh

# MODEL="vggt" # ['vggt','pi3','reliev3r']

# MODEL_SIZE=""
# CONFIG="./ckpts/VGGT-1B/config.json"
# CKPT="./ckpts/VGGT-1B/model.pt"
# PREFIX="vggt_official"
# source ./scripts/config/ase/518x336.sh

# MODEL="vggt" # ['vggt','pi3','reliev3r']
# CONFIG=""
# MODEL_SIZE=""
# CKPT="/mnt/nvme2/share/chenyouyu/codespace/vggt_sz/epoch036-loss_valid0.1726-metric_valid0.1199.ckpt/checkpoint/mp_rank_00_model_states.pt"
# PREFIX="vggt_sz_ep36"

USE_GT_POSE=0 # 1 if using GT extrinsics; 0 predicted
USE_GT_K=0 # 1 if using GT intrinsics; 0 predicted

SKIP_POINTMAP=0 # 1 if skip pointmap evaluation
SKIP_NORMAL=1 # 1 if skip point normal evaluation

# MODEL="vggt" # ['vggt','pi3','reliev3r']
# MODEL_SIZE=""
# CONFIG="./ckpts/VGGT-1B/config.json"
# CKPT="./ckpts/VGGT-1B/model.pt"
# PREFIX="vggt_official"

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
