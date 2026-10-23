#!/usr/bin/env bash
# exp1022_05 control: DINOv3-S/16 + 12 unshared decoder blocks at width 384 (6 pairs, K = 1) + 2-block heads.
# Same unique blocks / parameters as exp1022_03 (L6 loop body) but run once: is the loop gain from repetition
# or from having fewer parameters than the 36-block baseline? Recipe identical to exp1022_00 .. 04.
# Smoke test: SMOKE=1 bash run_sh_s3/run_scannetpp/run_exp1022_05_base_dec384x12.sh
set -euo pipefail
export NAME="exp1022_05_scannetpp_base_dec384x12_k1_head2"
export MODEL="pi3_dinov3s_w384x12"
bash "$(cd "$(dirname "$0")" && pwd)/common.sh"
