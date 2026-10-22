#!/usr/bin/env bash
# exp1022_01: loop body 1 pair, training K in [14, 18], inference K = 18.
# ScanNet++ only, 16 NPUs, fixed 8 views, 32 samples / NPU, 7.5k steps (configs/train/train_scannetpp_dinov3.yaml).
# Smoke test: SMOKE=1 bash run_sh_s3/run_scannetpp/run_exp1022_01_loop384_L1_k14to18.sh
set -euo pipefail
export NAME="exp1022_01_scannetpp_loop384_L1_k14to18_head2"
export MODEL="pi3_dinov3s_loop_w384_L1_k14to18"
bash "$(cd "$(dirname "$0")" && pwd)/common.sh"
