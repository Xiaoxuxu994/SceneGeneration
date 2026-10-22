#!/usr/bin/env bash
# B (loop): DINOv3 ViT-S/16 + looped decoder at width 384 (1 frame + 1 global block, K in [12,18], infer K=18)
# + 2-block heads; same data / recipe as A. ScanNet++ only, 32 NPUs, 30k steps.
# Smoke test: SMOKE=1 bash run_sh_s3/run_scannetpp/run_exp1022_00_B_dinov3s_loop384.sh
set -euo pipefail
export NAME="exp1022_00_scannetpp_B_dinov3s_loop384_k12-18_head2"
export MODEL="pi3_dinov3s_loop_w384"
bash "$(cd "$(dirname "$0")" && pwd)/common.sh"
