#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/efs-m3ga/z00533370/reliev3r_data/Omniobject3d-New/raw/blender_renders"
DST_DIR="/efs-m3ga/z00533370/reliev3r_data/Omniobject3dNew/blender_renders_decode"
JOBS=8

mkdir -p "${DST_DIR}"

echo "[INFO] SRC_DIR=${SRC_DIR}"
echo "[INFO] DST_DIR=${DST_DIR}"
echo "[INFO] JOBS=${JOBS}"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "[ERROR] source directory not found: ${SRC_DIR}" >&2
  exit 1
fi

extract_one() {
  local tar_path="$1"
  local tar_name
  tar_name="$(basename "${tar_path}")"

  echo "[INFO] extracting ${tar_name}"
  tar -xzf "${tar_path}" -C "${DST_DIR}"
}

export -f extract_one

find "${SRC_DIR}" -maxdepth 1 -type f -name '*.tar.gz' -print0 \
  | sort -z \
  | xargs -0 -n 1 -P "${JOBS}" bash -c 'extract_one "$0"'

echo "[INFO] extraction done, start chmod..."

chmod a+rx "${DST_DIR}"
find "${DST_DIR}" -type d -print0 | xargs -0 -r chmod a+rx
find "${DST_DIR}" -type f -print0 | xargs -0 -r chmod a+r

echo "[INFO] done"