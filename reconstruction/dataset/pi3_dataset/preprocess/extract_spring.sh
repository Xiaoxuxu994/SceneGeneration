#!/usr/bin/env bash
set -Eeuo pipefail

# ==========================================
# Usage:
#   bash extract_and_chmod_spring.sh [SRC_DIR] [DST_DIR] [JOBS]
#
# Example:
#   bash extract_and_chmod_spring.sh \
#     /efs-m3ga/h00548084/relive3r_raw/Spring \
#     /efs-m3ga/h00548084/relive3r_raw/Spring_extracted \
#     4
# ==========================================

SRC_DIR="${1:-/efs-m3ga/h00548084/relive3r_raw/Spring}"
DST_DIR="${2:-/efs-m3ga/z00533370/reliev3r_data/Spring_decode}"
JOBS="${3:-4}"   # HDD建议2~3，SSD/NVMe可试4~6

DONE_DIR="$DST_DIR/.extract_done"
LOG_DIR="$DST_DIR/.extract_logs"

mkdir -p "$DST_DIR" "$DONE_DIR" "$LOG_DIR"

echo "=========================================="
echo "SRC_DIR  = $SRC_DIR"
echo "DST_DIR  = $DST_DIR"
echo "JOBS     = $JOBS"
echo "DONE_DIR = $DONE_DIR"
echo "LOG_DIR  = $LOG_DIR"
echo "=========================================="
echo

extract_one() {
    local zip_path="$1"
    local zip_name
    zip_name="$(basename "$zip_path")"

    local done_flag="$DONE_DIR/${zip_name}.done"
    local log_file="$LOG_DIR/${zip_name}.log"

    if [[ -f "$done_flag" ]]; then
        echo "[SKIP] $zip_name already extracted"
        return 0
    fi

    echo "[START] $zip_name"

    # -q: quiet
    # -o: overwrite existing files if needed
    if unzip -q -o "$zip_path" -d "$DST_DIR" >"$log_file" 2>&1; then
        touch "$done_flag"
        echo "[ OK ] $zip_name"
    else
        echo "[FAIL] $zip_name   (see $log_file)"
        return 1
    fi
}

export DST_DIR DONE_DIR LOG_DIR
export -f extract_one

echo "[INFO] Start parallel extraction..."
find "$SRC_DIR" -maxdepth 1 -type f -name '*.zip' | sort | \
    xargs -I{} -n1 -P "$JOBS" bash -c 'extract_one "$@"' _ {}

echo
echo "[INFO] Extraction stage finished."
echo "[INFO] Applying permissions..."

# 所有目录可读可进入
find "$DST_DIR" -type d -exec chmod a+rx {} +

# 所有普通文件可读（包括 jpg/png/npy/txt/json 等）
find "$DST_DIR" -type f -exec chmod a+r {} +

echo "[INFO] Permission update finished."

echo
echo "[INFO] Extracted zip count:"
find "$DONE_DIR" -maxdepth 1 -type f -name '*.done' | wc -l

echo
echo "[INFO] Example top-level entries:"
find "$DST_DIR" -maxdepth 2 | head -50

echo
echo "[DONE] All finished."