#!/usr/bin/env bash
set -euo pipefail

########################################
# 用法:
#   bash sync_matrixcity.sh <src_dir> <dst_dir> [num_jobs]
#
# 示例:
#   bash sync_matrixcity.sh \
#       /efs-large-spatial-model/dataset/MatrixCity \
#       /your/target/path/MatrixCity \
#       8
#
# 说明:
#   1. 默认同步 scene_*_wo_anti_alias 目录
#   2. 使用 rsync 增量同步
#   3. 多进程并行加速
########################################

SRC_DIR="${1:-/efs-large-spatial-model/dataset/MatrixCity}"
DST_DIR="${2:-/efs-m3ga/z00533370/reliev3r_data/MatrixCity}"
NUM_JOBS="${3:-8}"

mkdir -p "${DST_DIR}"

echo "SRC_DIR   = ${SRC_DIR}"
echo "DST_DIR   = ${DST_DIR}"
echo "NUM_JOBS  = ${NUM_JOBS}"
echo

# 找到所有 scene_xxx_x_wo_anti_alias 目录
mapfile -t SCENES < <(find "${SRC_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'scene_*_wo_anti_alias' | sort)

if [ "${#SCENES[@]}" -eq 0 ]; then
    echo "没有找到匹配的目录: ${SRC_DIR}/scene_*_wo_anti_alias"
    exit 1
fi

echo "共找到 ${#SCENES[@]} 个场景目录"
echo

sync_one() {
    local scene_path="$1"
    local scene_name
    scene_name="$(basename "${scene_path}")"

    echo "[START] ${scene_name}"

    rsync -a \
        --info=progress2 \
        --human-readable \
        --partial \
        --append-verify \
        "${scene_path}/" "${DST_DIR}/${scene_name}/"

    echo "[DONE ] ${scene_name}"
}

export DST_DIR
export -f sync_one

printf '%s\n' "${SCENES[@]}" | xargs -I {} -P "${NUM_JOBS}" bash -c 'sync_one "$@"' _ {}

echo
echo "全部同步完成。"