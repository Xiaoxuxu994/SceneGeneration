#!/usr/bin/env bash
# A (baseline): DINOv3 ViT-S/16 + 36 unshared decoder blocks at width 384 + 2-block heads, ScanNet++ only,
# 32 NPUs, 30k steps.
# Smoke test: SMOKE=1 bash run_sh_s3/run_scannetpp/run_exp1022_00_A_dinov3s_dec384.sh
set -euo pipefail
export NAME="exp1022_00_scannetpp_A_dinov3s_dec384x36_head2"
export MODEL="pi3_dinov3s_w384"
bash "$(cd "$(dirname "$0")" && pwd)/common.sh"
