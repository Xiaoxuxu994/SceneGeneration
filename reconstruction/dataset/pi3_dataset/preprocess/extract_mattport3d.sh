#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/efs-m3ga/z00533370/reliev3r_data/Mattport3D/scans"
OUT_ROOT="/efs-m3ga/z00533370/reliev3r_data/Matterport3D_decode"
JOBS=4

mkdir -p "${OUT_ROOT}"

echo "[INFO] SRC_ROOT=${SRC_ROOT}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] JOBS=${JOBS}"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "[ERROR] source scans directory not found: ${SRC_ROOT}" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "[ERROR] unzip not found" >&2
  exit 1
fi

export OUT_ROOT

find "${SRC_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 \
  | sort -z \
  | xargs -0 -n 1 -P "${JOBS}" bash -c '
      scene_dir="$1"
      scene_id="$(basename "$scene_dir")"
      out_dir="${OUT_ROOT}/${scene_id}"

      mkdir -p "${out_dir}"

      needed=(
        "undistorted_color_images.zip"
        "undistorted_depth_images.zip"
        "undistorted_camera_parameters.zip"
        "undistorted_normal_images.zip"
      )

      echo "[INFO] scene=${scene_id}"

      for f in "${needed[@]}"; do
        if [[ ! -f "${scene_dir}/${f}" ]]; then
          echo "[WARN] skip ${scene_id}: missing ${f}"
          exit 0
        fi
      done

      for f in "${needed[@]}"; do
        echo "[INFO] extracting ${scene_id}/${f}"
        unzip -oq "${scene_dir}/${f}" -d "${out_dir}"
      done
    ' _

echo "[INFO] extraction done, start chmod..."

chmod a+rx "${OUT_ROOT}"
find "${OUT_ROOT}" -type d -print0 | xargs -0 -r chmod a+rx
find "${OUT_ROOT}" -type f -print0 | xargs -0 -r chmod a+r

echo "[INFO] done"#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/efs-m3ga/z00533370/reliev3r_data/Mattport3D/scans"
OUT_ROOT="/efs-m3ga/z00533370/reliev3r_data/Matterport3D_decode"
JOBS=4

mkdir -p "${OUT_ROOT}"

echo "[INFO] SRC_ROOT=${SRC_ROOT}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] JOBS=${JOBS}"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "[ERROR] source scans directory not found: ${SRC_ROOT}" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "[ERROR] unzip not found" >&2
  exit 1
fi

export OUT_ROOT

find "${SRC_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 \
  | sort -z \
  | xargs -0 -n 1 -P "${JOBS}" bash -c '
      scene_dir="$1"
      scene_id="$(basename "$scene_dir")"
      out_dir="${OUT_ROOT}/${scene_id}"

      mkdir -p "${out_dir}"

      needed=(
        "undistorted_color_images.zip"
        "undistorted_depth_images.zip"
        "undistorted_camera_parameters.zip"
        "undistorted_normal_images.zip"
      )

      echo "[INFO] scene=${scene_id}"

      for f in "${needed[@]}"; do
        if [[ ! -f "${scene_dir}/${f}" ]]; then
          echo "[WARN] skip ${scene_id}: missing ${f}"
          exit 0
        fi
      done

      for f in "${needed[@]}"; do
        echo "[INFO] extracting ${scene_id}/${f}"
        unzip -oq "${scene_dir}/${f}" -d "${out_dir}"
      done
    ' _

echo "[INFO] extraction done, start chmod..."

chmod a+rx "${OUT_ROOT}"
find "${OUT_ROOT}" -type d -print0 | xargs -0 -r chmod a+rx
find "${OUT_ROOT}" -type f -print0 | xargs -0 -r chmod a+r

echo "[INFO] done"