#!/usr/bin/env bash
# exp1022_00 baseline: DINOv3-S/16 + 36 unshared decoder blocks at width 384 (18 pairs, K = 1) + 2-block heads.
# ScanNet++ only, 16 NPUs, fixed 8 views, 16 samples / NPU, 15k steps (configs/train/train_scannetpp_dinov3.yaml).
# Smoke test: SMOKE=1 bash run_sh_s3/run_scannetpp/run_exp1022_00_base_dec384x36.sh
set -euo pipefail
export NAME="exp1022_00_scannetpp_base_dec384x36_k1_head2"
export MODEL="pi3_dinov3s_w384"
bash "$(cd "$(dirname "$0")" && pwd)/common.sh"
