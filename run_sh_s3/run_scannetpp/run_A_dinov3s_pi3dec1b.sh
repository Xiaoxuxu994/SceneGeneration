#!/usr/bin/env bash
# A (baseline): DINOv3 ViT-S/16 + Pi3-1B decoder (36 unshared blocks), ScanNet++ only, 32 NPUs, 30k steps.
# Smoke test: SMOKE=1 bash run_sh_s3/run_scannetpp/run_A_dinov3s_pi3dec1b.sh
set -euo pipefail
export NAME="exp_scannetpp_A_dinov3s_pi3dec1b"
export MODEL="pi3_dinov3s_1b"
bash "$(cd "$(dirname "$0")" && pwd)/common.sh"
