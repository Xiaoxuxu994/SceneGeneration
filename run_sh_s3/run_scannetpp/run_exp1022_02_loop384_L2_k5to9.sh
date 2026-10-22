#!/usr/bin/env bash
# exp1022_02: loop body 2 pairs, training K in [5, 9], inference K = 9.
# ScanNet++ only, 16 NPUs, fixed 8 views, 16 samples / NPU, 15k steps (configs/train/train_scannetpp_dinov3.yaml).
# Smoke test: SMOKE=1 bash run_sh_s3/run_scannetpp/run_exp1022_02_loop384_L2_k5to9.sh
set -euo pipefail
export NAME="exp1022_02_scannetpp_loop384_L2_k5to9_head2"
export MODEL="pi3_dinov3s_loop_w384_L2_k5to9"
bash "$(cd "$(dirname "$0")" && pwd)/common.sh"
