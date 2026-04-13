#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="$(pwd)"
export ASCEND_RT_VISIBLE_DEVICES=4
unset ASCEND_DEVICE_ID

NUM_PROCESSES=1
export USE_CHECKPOINT=True

accelerate launch \
  --config_file configs/accelerate/ddp.yaml \
  --num_processes ${NUM_PROCESSES} \
  --num_machines 1 \
  scripts/train_pi3.py \
  train=train_pi3_lowres \
  data=datamix_sync_v1 \
  name=exp0408_pi3_3B_datamix_sync_v1_highres_v2-16_512x336_debug