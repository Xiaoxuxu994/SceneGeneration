# DeepSpeed ZeRO checkpoint -> 单个 fp32 模型：用 checkpoint 目录里 DeepSpeed 自带的 zero_to_fp32.py（与保存时的版本一致）。
# 被 run_utils/mox_download.sh 与 run_datamix_all/_train_pi3_obs_common.sh source。
#
#   merge_zero_ckpt <checkpoint_N 目录> <输出所在目录> [等待的分片数]
# 输出命名 model_ckpt_<epoch>，epoch = N + 1（checkpoint_N 的 N 从 0 计；55 个 epoch 的 checkpoint_54 -> model_ckpt_55）：
#   新版 zero_to_fp32（有 --safe_serialization）-> 目录 model_ckpt_<epoch>/（safetensors，大模型会分片）
#   旧版                                         -> 文件 model_ckpt_<epoch>.bin
# 两种都可直接作训练的 model.ckpt=<路径>（pi3_training.py 目录走 load_sharded_checkpoint，文件走 torch.load）。
# 第三个参数 > 0 时先等 *_optim_states.pt 数量达到它（多节点训练各节点把自己的分片传到同一个 efs 目录，到齐了才能合并），
# 最多等 MERGE_WAIT_MAX 秒（默认 7200）。目录里没有 zero_to_fp32.py 时退回 tools/zero_ckpt_to_model.py --mode fp32。
merge_zero_ckpt () {
    local CK="$1" OUT_PARENT="$2" EXPECT="${3:-0}"
    local NAME_CK; NAME_CK="$(basename "${CK}")"
    local N="${NAME_CK#checkpoint_}"
    local EPOCH
    if [[ "${N}" =~ ^[0-9]+$ ]]; then EPOCH=$((N + 1)); else EPOCH="${NAME_CK}"; fi
    local OUT_BASE="${OUT_PARENT}/model_ckpt_${EPOCH}"
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
    if [ -f "${CK}/zero_to_fp32.py" ]; then
        local HELP; HELP="$(cd "${CK}" && python zero_to_fp32.py -h 2>&1 || true)"   # 不用管道：grep -q 提前退出会让 pipefail 误判
        if [[ "${HELP}" == *safe_serialization* ]]; then
            echo "[merge] ${CK} -> ${OUT_BASE}/ (zero_to_fp32.py, safetensors)"
            (cd "${CK}" && python zero_to_fp32.py . "${OUT_BASE}" --safe_serialization ${TAG[@]+"${TAG[@]}"}); rc=$?
        else
            echo "[merge] ${CK} -> ${OUT_BASE}.bin (zero_to_fp32.py)"
            (cd "${CK}" && python zero_to_fp32.py . "${OUT_BASE}.bin" ${TAG[@]+"${TAG[@]}"}); rc=$?
        fi
    else
        local CODE_ROOT; CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        echo "[merge] ${CK} 里没有 zero_to_fp32.py，用 tools/zero_ckpt_to_model.py -> ${OUT_BASE}.bin"
        python "${CODE_ROOT}/tools/zero_ckpt_to_model.py" "${CK}" --mode fp32 --out "${OUT_BASE}.bin"; rc=$?
    fi
    if [ "${rc}" -eq 0 ]; then
        echo "[merge] done in $((SECONDS - t0))s: $(du -sh "${OUT_BASE}" "${OUT_BASE}.bin" 2>/dev/null | tr '\n' ' ')"
    else
        echo "[merge] FAILED (rc=${rc}): ${CK}"
    fi
    return "${rc}"
}
