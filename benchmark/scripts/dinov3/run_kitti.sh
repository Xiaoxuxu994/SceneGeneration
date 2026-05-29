#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
SCRIPT="scripts/run_infer.py"
EVAL_SCRIPT="scripts/run_eval.py"
VIS_SCRIPT="scripts/run_vis.py"

# ----- model ------
source ./scripts/config/kitti/512x144.sh

MODEL=$1 # ['vggt','pi3','reliev3r']
MODEL_SIZE=$2
CONFIG=$3
CKPT=$4
PREFIX=$5
NPU_ID=$6

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
