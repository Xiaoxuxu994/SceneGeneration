#!/usr/bin/env bash
set -Eeuo pipefail

# ==========================================
# Usage:
#   bash extract_and_chmod_unrealstereo4k_480p.sh [SRC_DIR] [DST_DIR] [JOBS]
#
# Example:
#   bash extract_and_chmod_unrealstereo4k_480p.sh \
#     /efs-m3ga/h00548084/relive3r_raw/UnrealStereo4K-480P \
#     /efs-m3ga/h00548084/relive3r_raw/UnrealStereo4K-480P_extracted \
#     4
# ==========================================

SRC_DIR="${1:-/efs-m3ga/h00548084/relive3r_raw/UnrealStereo4K}"
DST_DIR="${2:-/efs-m3ga/z00533370/reliev3r_data/UnrealStereo4K_decode}"
JOBS="${3:-4}"   # 建议 HDD: 2~4, SSD/NVMe: 4~8

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
echo "[INFO] Applying read permissions recursively to extracted data..."

# 给所有文件加读权限；给目录加可进入权限
chmod -R a+rX "$DST_DIR"

echo "[INFO] Permission update finished."

echo
echo "[INFO] Summary:"
echo "  Extracted zip count:"
ls -1 "$DONE_DIR"/*.done 2>/dev/null | wc -l || true

echo
echo "[INFO] Example checks:"
echo "  ls -l $DST_DIR | head"
echo "  find $DST_DIR -maxdepth 2 | head -50"

echo
echo "[DONE] All finished."