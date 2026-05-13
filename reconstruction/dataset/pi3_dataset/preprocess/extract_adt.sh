#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/efs-m3ga/z00533370/reliev3r_data/ADT"
OUT_DIR="/efs-m3ga/z00533370/reliev3r_data/ADT_decode"
JOBS="${1:-8}"

# 0: 已存在则跳过
# 1: 强制覆盖
OVERWRITE="${OVERWRITE:-0}"

# 0: 只给新解压目录 chmod
# 1: 最后对整个 OUT_DIR 再统一 chmod 一遍
CHMOD_OUT_ROOT="${CHMOD_OUT_ROOT:-1}"

export ROOT_DIR OUT_DIR JOBS OVERWRITE CHMOD_OUT_ROOT

echo "ROOT_DIR=$ROOT_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "JOBS=$JOBS"
echo "OVERWRITE=$OVERWRITE"
echo "CHMOD_OUT_ROOT=$CHMOD_OUT_ROOT"

mkdir -p "$OUT_DIR"

ensure_readable_tree() {
  local target="$1"
  find "$target" -type d -exec chmod a+rx {} + 2>/dev/null || true
  find "$target" -type f -exec chmod a+r {} + 2>/dev/null || true
}

process_one_zip() {
  local zipf="$1"
  local seq_dir seq_name zip_base out_seq_dir out_dir unzip_flag

  seq_dir="$(dirname "$zipf")"
  seq_name="$(basename "$seq_dir")"
  zip_base="$(basename "$zipf" .zip)"

  out_seq_dir="$OUT_DIR/$seq_name"
  out_dir="$out_seq_dir/$zip_base"

  mkdir -p "$out_seq_dir"

  if [[ -d "$out_dir" ]] && find "$out_dir" -mindepth 1 -print -quit | grep -q . && [[ "$OVERWRITE" != "1" ]]; then
    echo "[SKIP] already extracted: $out_dir"
    return 0
  fi

  mkdir -p "$out_dir"

  if [[ "$OVERWRITE" == "1" ]]; then
    unzip_flag="-o"
  else
    unzip_flag="-n"
  fi

  echo "[UNZIP] $(basename "$zipf") -> $out_dir"
  unzip -q "$unzip_flag" "$zipf" -d "$out_dir"

  ensure_readable_tree "$out_dir"
  echo "[DONE ] $out_dir"
}

export -f ensure_readable_tree
export -f process_one_zip

mapfile -d '' ZIP_LIST < <(
  find "$ROOT_DIR" -mindepth 2 -maxdepth 2 -type f \
    \( \
      -name '*_depth.zip' -o \
      -name '*_main_groundtruth.zip' -o \
      -name '*_mps_slam_calibration.zip' -o \
      -name '*_synthetic.zip' \
    \) \
    -print0 | sort -z
)

ZIP_COUNT="${#ZIP_LIST[@]}"
echo "Found $ZIP_COUNT target zip files"

if [[ "$ZIP_COUNT" -eq 0 ]]; then
  echo "No target zip files found."
  exit 0
fi

printf '%s\0' "${ZIP_LIST[@]}" | xargs -0 -n 1 -P "$JOBS" bash -c '
  process_one_zip "$0"
'

if [[ "$CHMOD_OUT_ROOT" == "1" ]]; then
  echo "[CHMOD] applying readable permissions to whole OUT_DIR..."
  ensure_readable_tree "$OUT_DIR"
fi

echo "All done."
