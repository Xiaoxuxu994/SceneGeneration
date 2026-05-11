#!/usr/bin/env bash
set -euo pipefail

SRC_FILE="/efs-m3ga/z00533370/reliev3r_data/GibsonGibson/gibson_habitat_trainval.zip"
DST_DIR="/efs-m3ga/z00533370/reliev3r_data/Gibson_fullplus_decode"

mkdir -p "${DST_DIR}"

echo "[INFO] SRC_FILE=${SRC_FILE}"
echo "[INFO] DST_DIR=${DST_DIR}"

if [[ ! -f "${SRC_FILE}" ]]; then
  echo "[ERROR] source file not found: ${SRC_FILE}" >&2
  exit 1
fi

echo "[INFO] extracting gibson_fullplus.tar.gz ..."
tar -xzf "${SRC_FILE}" -C "${DST_DIR}"

echo "[INFO] extraction done, start chmod..."

chmod a+rx "${DST_DIR}"
find "${DST_DIR}" -type d -print0 | xargs -0 -r chmod a+rx
find "${DST_DIR}" -type f -print0 | xargs -0 -r chmod a+r

echo "[INFO] done"