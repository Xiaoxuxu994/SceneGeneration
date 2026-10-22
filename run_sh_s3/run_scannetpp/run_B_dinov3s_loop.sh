#!/usr/bin/env bash
# B (loop): DINOv3 ViT-S/16 + looped decoder (1 frame + 1 global block, K in [12,18], infer K=18),
# same heads / data / recipe as A. ScanNet++ only, 32 NPUs, 30k steps.
# Smoke test: SMOKE=1 bash run_sh_s3/run_scannetpp/run_B_dinov3s_loop.sh
set -euo pipefail
export NAME="exp_scannetpp_B_dinov3s_loop_k12-18"
export MODEL="pi3_dinov3s_loop"
bash "$(cd "$(dirname "$0")" && pwd)/common.sh"
