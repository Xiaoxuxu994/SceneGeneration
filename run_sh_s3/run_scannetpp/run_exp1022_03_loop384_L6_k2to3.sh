#!/usr/bin/env bash
# exp1022_03: loop body 6 pairs, training K in [2, 3], inference K = 3.
# ScanNet++ only, 16 NPUs, fixed 8 views, 32 samples / NPU, 7.5k steps (configs/train/train_scannetpp_dinov3.yaml).
# Smoke test: SMOKE=1 bash run_sh_s3/run_scannetpp/run_exp1022_03_loop384_L6_k2to3.sh
set -euo pipefail
export NAME="exp1022_03_scannetpp_loop384_L6_k2to3_head2"
export MODEL="pi3_dinov3s_loop_w384_L6_k2to3"
bash "$(cd "$(dirname "$0")" && pwd)/common.sh"
