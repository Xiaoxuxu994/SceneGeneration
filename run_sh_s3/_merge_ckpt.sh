# DeepSpeed ZeRO checkpoint -> 单个 fp32 模型文件：用 checkpoint 目录里 DeepSpeed 自带的 zero_to_fp32.py（与保存时的版本一致）合并，
# 再用 tools/slim_ckpt.py 把结果存成一个文件。被 run_utils/mox_download.sh 与 run_datamix_all/_train_pi3_obs_common.sh source。
#
#   merge_zero_ckpt <checkpoint_N 目录> <输出所在目录> [等待的分片数]
# 输出：<输出所在目录>/model_ckpt_N.safetensors（编号与 checkpoint_N 一致；best_model -> model_ckpt_best_model），
#   fp32、只含模型参数；没有 safetensors 包时为 model_ckpt_N.pt。可直接作训练的 model.ckpt=<文件>。
# 为什么还要过一遍 slim_ckpt.py：zero_to_fp32 的各参数是同一块 fp32 缓冲区的切片，torch.save 会把整块缓冲区写进每个分片，
#   3B 模型输出 3 个 7.8GB 的 .bin（共 26.8GB，实际参数 11.2GB；2026-11-11 实测）。slim 逐个 clone 后再存，就是 11.2GB。
# 第三个参数 > 0 时先等 *_optim_states.pt 数量达到它（多节点训练各节点把自己的分片传到同一个 efs 目录，到齐了才能合并），
# 最多等 MERGE_WAIT_MAX 秒（默认 7200）。目录里没有 zero_to_fp32.py 时退回 tools/zero_ckpt_to_model.py --mode fp32。
merge_zero_ckpt () {
    local CK="$1" OUT_PARENT="$2" EXPECT="${3:-0}"
    local NAME_CK; NAME_CK="$(basename "${CK}")"
    local OUT_BASE="${OUT_PARENT}/model_ckpt_${NAME_CK#checkpoint_}"
    local RAW="${OUT_BASE}.raw"
    local CODE_ROOT; CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    mkdir -p "${OUT_PARENT}"

    if [ "${EXPECT}" -gt 0 ] 2>/dev/null; then
        local waited=0 have=0
        while :; do
            have=$(find "${CK}" -name '*_optim_states.pt' 2>/dev/null | wc -l)
            [ "${have}" -ge "${EXPECT}" ] && break
            if [ "${waited}" -ge "${MERGE_WAIT_MAX:-7200}" ]; then
                echo "[merge] WARNING: ${CK} 只有 ${have}/${EXPECT} 个优化器分片，等了 ${waited}s，放弃合并"
                return 1
            fi
            sleep 30; waited=$((waited + 30))
        done
        echo "[merge] ${have}/${EXPECT} 个分片已到齐（等了 ${waited}s）"
    fi

    local TAG=()
    [ -f "${CK}/latest" ] || TAG=(-t "$(basename "$(dirname "$(find "${CK}" -name 'mp_rank_00_model_states.pt' | head -1)")")")
    local t0=$SECONDS rc
    rm -rf "${RAW}"
    if [ -f "${CK}/zero_to_fp32.py" ]; then
        # 第二个参数：新版是输出目录（output_dir，分片），旧版是输出文件（output_file）。不用管道：grep -q 提前退出会让 pipefail 误判
        local HELP; HELP="$(cd "${CK}" && python zero_to_fp32.py -h 2>&1 || true)"
        local RAW_OUT="${RAW}"
        [[ "${HELP}" == *output_dir* ]] || RAW_OUT="${RAW}/model.bin"
        mkdir -p "${RAW}"
        echo "[merge] zero_to_fp32.py: ${CK} -> ${RAW_OUT}"
        (cd "${CK}" && python zero_to_fp32.py . "${RAW_OUT}" ${TAG[@]+"${TAG[@]}"}); rc=$?
    else
        echo "[merge] ${CK} 里没有 zero_to_fp32.py，用 tools/zero_ckpt_to_model.py"
        mkdir -p "${RAW}"
        python "${CODE_ROOT}/tools/zero_ckpt_to_model.py" "${CK}" --mode fp32 --out "${RAW}/model.bin"; rc=$?
    fi
    if [ "${rc}" -eq 0 ]; then
        local EXT=safetensors
        python -c "import safetensors" 2>/dev/null || EXT=pt
        echo "[merge] slim_ckpt.py: ${RAW} -> ${OUT_BASE}.${EXT}"
        python "${CODE_ROOT}/tools/slim_ckpt.py" "${RAW}" --out "${OUT_BASE}.${EXT}"; rc=$?
    fi
    if [ "${rc}" -eq 0 ]; then
        rm -rf "${RAW}"
        echo "[merge] done in $((SECONDS - t0))s: $(du -sh "${OUT_BASE}".* 2>/dev/null | tr '\n' ' ')"
    else
        echo "[merge] FAILED (rc=${rc}): ${CK}（中间结果留在 ${RAW}）"
    fi
    return "${rc}"
}
