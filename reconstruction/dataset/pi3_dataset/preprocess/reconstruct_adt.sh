#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/efs-m3ga/z00533370/reliev3r_data/ADT_decode"
OVERWRITE="${OVERWRITE:-0}"   # 0: 已存在则跳过, 1: 覆盖
JOBS="${1:-8}"

export ROOT_DIR OVERWRITE

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "[ERROR] ROOT_DIR not found: $ROOT_DIR"
  exit 1
fi

echo "ROOT_DIR=$ROOT_DIR"
echo "OVERWRITE=$OVERWRITE"
echo "JOBS=$JOBS"

move_one_scene() {
  local scene_dir="$1"
  local subdir item base target

  shopt -s dotglob nullglob

  echo "[SCENE] $scene_dir"

  for subdir in "$scene_dir"/*/; do
    [[ -d "$subdir" ]] || continue

    for item in "$subdir"*; do
      [[ -e "$item" ]] || continue

      # 只移动文件；如果有更深层目录，这里跳过
      if [[ -d "$item" ]]; then
        echo "[SKIP-DIR] $item"
        continue
      fi

      base="$(basename "$item")"
      target="$scene_dir/$base"

      if [[ -e "$target" ]]; then
        if [[ "$OVERWRITE" == "1" ]]; then
          mv -f "$item" "$target"
          echo "[OVERWRITE] $item -> $target"
        else
          echo "[SKIP-EXIST] $target"
        fi
      else
        mv "$item" "$target"
        echo "[MOVE] $item -> $target"
      fi
    done
  done

  # 删除空的一级子目录
  find "$scene_dir" -mindepth 1 -maxdepth 1 -type d -empty -print -delete
}

export -f move_one_scene

mapfile -d '' SCENE_LIST < <(
  find "$ROOT_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)

SCENE_COUNT="${#SCENE_LIST[@]}"
echo "Found $SCENE_COUNT scene folders"

if [[ "$SCENE_COUNT" -eq 0 ]]; then
  echo "No scene folders found."
  exit 0
fi

printf '%s\0' "${SCENE_LIST[@]}" | xargs -0 -n 1 -P "$JOBS" bash -c '
  move_one_scene "$0"
'

echo "All done."