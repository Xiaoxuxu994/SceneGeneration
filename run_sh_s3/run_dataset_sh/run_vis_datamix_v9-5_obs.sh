#!/usr/bin/env bash
# 数据读取与 GT 验证：按 datamix 配置（默认 v9-5 的贵阳 OBS 版）逐个数据集取样本，
#   1) check：构造数据集 + 读取样本，打印每帧的图像尺寸、深度范围、有效像素比例、位姿，汇总成 check_summary.tsv
#   2) vis  ：导出点云 + 相机 + 逐帧图（tools/gt_vis_dump.py），用 MeshLab / CloudCompare 打开看 GT 是否正确
#
# 用法（ModelArts 或 devserver 均可）：
#   bash run_sh_s3/run_dataset_sh/run_vis_datamix_v9-5_obs.sh                       # 全部数据集，check + vis
#   STAGE=check bash run_sh_s3/run_dataset_sh/run_vis_datamix_v9-5_obs.sh           # 只做读取检查（快）
#   DATASETS="Hypersim CO3DV2" N=3 bash run_sh_s3/run_dataset_sh/run_vis_datamix_v9-5_obs.sh
#   OBS_OUT=obs://bucket-471/z00533370/vis/datamix_v9-5 bash run_sh_s3/run_dataset_sh/run_vis_datamix_v9-5_obs.sh  # 另外再传一份到 OBS
#
# 可用环境变量：
#   CODE_DIR   代码目录（默认本脚本所在仓库；ModelArts 上为 /home/ma-user/modelarts/user-job-dir/Pi3）
#   CONFIG     configs/data/datamix_all_v9-5_obs.yaml（加修复开关看效果：datamix_all_v9-5_fix_obs.yaml）
#   DATASETS   ALL = 配置 weights 里的全部数据集
#   STAGE      all | check | vis
#   N=2  FRAMES=8  RES_W=512  RES_H=336  PARALLEL=4（vis 阶段同时跑几个数据集）
#   VIS_ROOT   输出根目录（默认 /efs-mi-pay-gy1/z00533370/data_vis）
#   OUT_DIR    输出目录（默认 ${VIS_ROOT}/<配置名>，例如 .../data_vis/datamix_all_v9-5_obs）
#   OBS_OUT    非空时把 OUT_DIR 整个复制到这个 obs:// 目录
#
# 输出：
#   ${OUT_DIR}/check/check_summary.tsv              每个数据集：状态、样本数、成功/失败数、耗时
#   ${OUT_DIR}/vis/<数据集>_vis/<idx>_<label>_f8_512x336/
#       current.ply   训练实际用到的点（RGB）+ 相机        diag.ply    诊断着色：品红=多视角违例 黄=被 98 分位截掉 红=边缘
#       filtered.ply  建议过滤后的点                      cameras.ply 相机视锥与轨迹（看视差）
#       frames.png    每帧一行：RGB | 深度 | 诊断图        stats.json  基线/深度比、各类像素比例
#   ${OUT_DIR}/logs/<数据集>.log
set -euo pipefail

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1   # 日志实时写出（否则输出经 tee / 重定向时 print 被缓冲，看不到进度）

CONFIG="${CONFIG:-configs/data/datamix_all_v9-5_obs.yaml}"
DATASETS="${DATASETS:-ALL}"
STAGE="${STAGE:-all}"
N="${N:-2}"
FRAMES="${FRAMES:-8}"
RES_W="${RES_W:-512}"
RES_H="${RES_H:-336}"
PARALLEL="${PARALLEL:-4}"
VIS_ROOT="${VIS_ROOT:-/efs-mi-pay-gy1/z00533370/data_vis}"
OUT_DIR="${OUT_DIR:-${VIS_ROOT}/$(basename "${CONFIG}" .yaml)}"
mkdir -p "${OUT_DIR}/logs"

# 依赖：缺的包才安装（新镜像常缺 hydra 等）
bash run_sh_s3/ensure_deps.sh

if [ "${DATASETS}" = "ALL" ]; then
    DATASETS="$(python -c "from omegaconf import OmegaConf; print(' '.join(OmegaConf.load('${CONFIG}').train_dataset.weights.keys()))")"
fi

echo "==> CODE_DIR=${CODE_DIR}"
echo "==> CONFIG=${CONFIG}  STAGE=${STAGE}  N=${N}  FRAMES=${FRAMES}  RES=${RES_W}x${RES_H}"
echo "==> OUT_DIR=${OUT_DIR}"
echo "==> DATASETS=${DATASETS}"

# -----------------------------
# 1) 读取检查
# -----------------------------
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "check" ]; then
    echo "==> [check] start"
    # shellcheck disable=SC2086
    python tools/check_datasets.py --config "${CONFIG}" --datasets ${DATASETS} --n "${N}" --frames "${FRAMES}" \
        --res "${RES_W}" "${RES_H}" --out "${OUT_DIR}/check" 2>&1 | tee "${OUT_DIR}/logs/check.log" || true
    echo "==> [check] summary: ${OUT_DIR}/check/check_summary.tsv"
fi

# -----------------------------
# 2) 可视化（每个数据集一个进程）
# -----------------------------
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "vis" ]; then
    echo "==> [vis] start (parallel=${PARALLEL})"
    for ds in ${DATASETS}; do
        while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]; do sleep 2; done
        echo "==> [vis] ${ds}  (log: ${OUT_DIR}/logs/${ds}.log)"
        python tools/gt_vis_dump.py --config "${CONFIG}" --datasets "${ds}" --n "${N}" --frames "${FRAMES}" \
            --res "${RES_W}" "${RES_H}" --out-root "${OUT_DIR}/vis" > "${OUT_DIR}/logs/${ds}.log" 2>&1 \
            && echo "==> [vis] done ${ds}" || echo "==> [vis] FAIL ${ds}: see ${OUT_DIR}/logs/${ds}.log" &
    done
    wait
    echo
    echo "==> [vis] exported samples per dataset:"
    for ds in ${DATASETS}; do
        d="${OUT_DIR}/vis/${ds}_vis"
        n=$( [ -d "${d}" ] && find "${d}" -mindepth 1 -maxdepth 1 -type d ! -name '.tmp_*' | wc -l || echo 0 )
        printf "    %-18s %s\n" "${ds}" "${n}"
    done
fi

# -----------------------------
# 3) 可选：传回 OBS
# -----------------------------
if [ -n "${OBS_OUT:-}" ]; then
    echo "==> upload ${OUT_DIR} -> ${OBS_OUT}"
    python - <<PY
import moxing as mox
mox.file.copy_parallel(r"""${OUT_DIR}""", r"""${OBS_OUT}""")
print("[mox] upload done")
PY
fi

echo "==> finished: ${OUT_DIR}"
