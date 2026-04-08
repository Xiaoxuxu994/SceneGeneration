#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="$(pwd)"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ASCEND_DEVICE_ID

NUM_PROCESSES=8

export USE_CHECKPOINT=False

accelerate launch \
  --config_file configs/accelerate/ddp.yaml \
  --num_processes ${NUM_PROCESSES} \
  --num_machines 1 \
  scripts/train_pi3.py \
  train=train_pi3_highres \
  data=blendedmvs \
  name=exp0402_pi3_3B_blendedmvs_highres_v2-16_512x336_maxdist8_debug