#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/efs-m3ga/z00533370/reliev3r_data/taskonomy"
JOBS=16
BATCH=1000

echo "ROOT_DIR=$ROOT_DIR"
echo "JOBS=$JOBS"
echo "BATCH=$BATCH"

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "[ERROR] directory not found: $ROOT_DIR" >&2
  exit 1
fi

chmod a+rx "$ROOT_DIR"

echo "[1/2] chmod directories to a+rx ..."
find "$ROOT_DIR" -type d -print0 | xargs -0 -n "$BATCH" -P "$JOBS" chmod a+rx

echo "[2/2] chmod files to a+r ..."
find "$ROOT_DIR" -type f -print0 | xargs -0 -n "$BATCH" -P "$JOBS" chmod a+r

echo "[DONE] all directories are a+rx, all files are a+r"