#!/usr/bin/env bash
# Download checkpoints + tensorboard logs of ModelArts runs from OBS to EFS with moxing.
# Paths follow common.sh: OBS  ${OBS_ROOT}/${NAME}/outputs/${NAME}/{ckpts/<ckpt>, ${NAME}}
#                         EFS  ${EFS_ROOT}/${NAME}/{ckpts/<ckpt>, ${NAME}}
#
# Usage (run where moxing is available, e.g. a ModelArts notebook / job):
#   bash mox_download.sh                                   # the five exp1022 runs: latest ckpt + log
#   bash mox_download.sh exp1022_03_scannetpp_loop384_L6_k2to3_head2 [more names ...]
#   CKPT=checkpoint_9 bash mox_download.sh <name>          # a specific checkpoint
#   CKPT=best_model   bash mox_download.sh <name>          # best checkpoint by val loss
#   CKPT=none         bash mox_download.sh                 # logs only
#   CKPT=all          bash mox_download.sh <name>          # every checkpoint (large: DeepSpeed states)
#   LOG=0             bash mox_download.sh <name>          # checkpoint only
set -uo pipefail

OBS_ROOT="${OBS_ROOT:-obs://bucket-mi-east4-sf/z00533370/project/proj_3dgs/exp_pi3loop}"
EFS_ROOT="${EFS_ROOT:-/efs-m3ga/z00533370/exp/pi3sup/outputs}"
CKPT="${CKPT:-latest}"
LOG="${LOG:-1}"

if [ "$#" -gt 0 ]; then
    NAMES=("$@")
else
    NAMES=(
        exp1022_00_scannetpp_base_dec384x36_k1_head2
        exp1022_01_scannetpp_loop384_L1_k14to18_head2
        exp1022_02_scannetpp_loop384_L2_k5to9_head2
        exp1022_03_scannetpp_loop384_L6_k2to3_head2
        exp1022_04_scannetpp_loop384_L1_k1to18_head2
    )
fi

# mox_py <mode> <src> [dst]: mode = latest (print latest checkpoint_N under src) | list (print all) | copy
mox_py () {
    python - "$@" <<'PY'
import re, sys
import moxing as mox
mode, src = sys.argv[1], sys.argv[2].rstrip("/")
if mode == "copy":
    dst = sys.argv[3]
    if not mox.file.exists(src):
        print(f"[mox] not found: {src}"); sys.exit(2)
    mox.file.copy_parallel(src, dst)
    print(f"[mox] copy done: {src} -> {dst}")
    sys.exit(0)
if not mox.file.exists(src):
    sys.exit(2)
ckpts = []
for item in mox.file.list_directory(src):
    m = re.match(r"checkpoint_(\d+)$", item.rstrip("/").split("/")[-1])
    if m:
        ckpts.append((int(m.group(1)), f"checkpoint_{m.group(1)}"))
ckpts.sort()
if not ckpts:
    sys.exit(3)
print(ckpts[-1][1] if mode == "latest" else " ".join(c for _, c in ckpts))
PY
}

copy_by_mox () {
    local SRC="$1" DST="$2"
    mkdir -p "$(dirname "${DST}")"
    echo "==> mox copy: ${SRC} -> ${DST}"
    mox_py copy "${SRC}" "${DST}"
}

FAILED=()
for NAME in "${NAMES[@]}"; do
    echo "================ ${NAME} ================"
    OBS_RUN="${OBS_ROOT}/${NAME}/outputs/${NAME}"
    EFS_RUN="${EFS_ROOT}/${NAME}"
    ok=1

    # 1) checkpoint(s)
    case "${CKPT}" in
        none) CKPT_LIST=() ;;
        latest)
            c="$(mox_py latest "${OBS_RUN}/ckpts" 2>/dev/null | tail -1)"
            if [ -n "${c}" ]; then CKPT_LIST=("${c}"); else echo "[WARN] no checkpoint_N under ${OBS_RUN}/ckpts"; CKPT_LIST=(); ok=0; fi ;;
        all)
            read -r -a CKPT_LIST <<< "$(mox_py list "${OBS_RUN}/ckpts" 2>/dev/null | tail -1)"
            [ "${#CKPT_LIST[@]}" -gt 0 ] || { echo "[WARN] no checkpoint_N under ${OBS_RUN}/ckpts"; ok=0; } ;;
        *) CKPT_LIST=("${CKPT}") ;;
    esac
    for c in ${CKPT_LIST[@]+"${CKPT_LIST[@]}"}; do
        echo "==> checkpoint: ${c}"
        copy_by_mox "${OBS_RUN}/ckpts/${c}" "${EFS_RUN}/ckpts/${c}" || ok=0
    done

    # 2) tensorboard / text logs
    if [ "${LOG}" = "1" ]; then
        echo "==> log"
        copy_by_mox "${OBS_RUN}/${NAME}" "${EFS_RUN}/${NAME}" || ok=0
    fi

    [ "${ok}" = "1" ] || FAILED+=("${NAME}")
done

echo "================ summary ================"
echo "EFS: ${EFS_ROOT}"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "[WARN] incomplete: ${FAILED[*]}"
    exit 1
fi
echo "==> Done"
