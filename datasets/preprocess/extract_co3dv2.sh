#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/efs-m3ga/z00533370/reliev3r_data/CO3D-v2CO3D-v2"
DST_DIR="/efs-m3ga/z00533370/reliev3r_data/CO3Dv2_decode"
JOBS=8

mkdir -p "${DST_DIR}"

echo "[INFO] SRC_DIR=${SRC_DIR}"
echo "[INFO] DST_DIR=${DST_DIR}"
echo "[INFO] JOBS=${JOBS}"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "[ERROR] source directory not found: ${SRC_DIR}" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "[ERROR] unzip not found" >&2
  exit 1
fi

export DST_DIR

extract_one() {
  local zip_path="$1"
  local zip_name stem category out_dir

  zip_name="$(basename "${zip_path}")"
  stem="${zip_name%.zip}"

  if [[ "${stem}" == *_* ]]; then
    category="${stem%_*}"
  else
    category="${stem}"
  fi

  out_dir="${DST_DIR}/${category}"
  mkdir -p "${out_dir}"

  echo "[INFO] extracting ${zip_name} -> ${out_dir}"
  unzip -oq "${zip_path}" -d "${out_dir}"
}

export -f extract_one

find "${SRC_DIR}" -maxdepth 1 -type f -name '*.zip' -print0 \
  | sort -z \
  | xargs -0 -n 1 -P "${JOBS}" bash -c 'extract_one "$0"' 

echo "[INFO] extraction done, start chmod..."

chmod a+rx "${DST_DIR}"
find "${DST_DIR}" -type d -print0 | xargs -0 -r chmod a+rx
find "${DST_DIR}" -type f -print0 | xargs -0 -r chmod a+r

echo "[INFO] done"