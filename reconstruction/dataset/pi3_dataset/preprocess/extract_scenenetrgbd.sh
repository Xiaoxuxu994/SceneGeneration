#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/efs-m3ga/z00533370/reliev3r_data/SceneNet-RGB-D"
DST_DIR="/efs-m3ga/z00533370/reliev3r_data/SceneNet-RGB-D_decode"
JOBS=2

mkdir -p "${DST_DIR}"

echo "[INFO] SRC_DIR=${SRC_DIR}"
echo "[INFO] DST_DIR=${DST_DIR}"
echo "[INFO] JOBS=${JOBS}"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "[ERROR] source directory not found: ${SRC_DIR}" >&2
  exit 1
fi

export DST_DIR

find "${SRC_DIR}" -maxdepth 1 -type f \( -name '*.tar.gz' -o -name '*.tar' \) -print0 \
  | sort -z \
  | xargs -0 -n 1 -P "${JOBS}" bash -c '
      archive="$1"
      name="$(basename "$archive")"

      echo "[INFO] extracting ${name}"
      tar -xf "$archive" -C "${DST_DIR}"
    ' _

if [[ -f "${SRC_DIR}/scenenet_rgbd_val.pb" ]]; then
  echo "[INFO] copying scenenet_rgbd_val.pb"
  cp -f "${SRC_DIR}/scenenet_rgbd_val.pb" "${DST_DIR}/"
fi

echo "[INFO] extraction done, start chmod..."

chmod a+rx "${DST_DIR}"
find "${DST_DIR}" -type d -print0 | xargs -0 -r chmod a+rx
find "${DST_DIR}" -type f -print0 | xargs -0 -r chmod a+r

echo "[INFO] done"