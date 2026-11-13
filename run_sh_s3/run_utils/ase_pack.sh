#!/usr/bin/env bash
# 从 OBS 读取评测要用的 ASE 文件，打成一个 tar.gz 放到贵阳 efs，方便传出去。
#   源：  obs://bucket-471/y00959808/data/sr_datasets/processed/ASE/<场景>/{undistort_rgb,undistort_depth}/...
#         （= 训练配置里 /efs-m3ga/c50041401/datasets/reliev3r_synth_data/ASE 的 path_remap 目标）
#   输出：${OUT_DIR}/ase_subset.tar.gz（tar 内路径相对 ASE/，即 <场景>/undistort_rgb/vignette0000000.jpg）
#
# 两种取法：
#   LIST=<清单文件>  按清单取（ase_fetch.txt：每行一个相对 ASE/ 的路径，# 开头的行忽略），正好 640 个文件
#   不给 LIST        按场景整目录取 SCENES 里每个场景的 undistort_rgb/ 和 undistort_depth/（比清单多，但不怕漏帧）
#
# ModelArts（云道）作业，单节点即可，作业要挂载 /efs-mi-pay-gy1：
#   cd /home/ma-user/modelarts/user-job-dir/SceneGeneration/; source /home/ma-user/.bashrc; LIST=/efs-mi-pay-gy1/z00533370/transfer/ase_fetch.txt bash run_sh_s3/run_utils/ase_pack.sh
# 可用环境变量：
#   SRC_ROOT  默认 obs://bucket-471/y00959808/data/sr_datasets/processed/ASE
#   OUT_DIR   默认 /efs-mi-pay-gy1/z00533370/transfer（tar 包和日志写这里）
#   SCENES    默认 "10000 10001 10004 10006 10014 10018 10025 10026 10028 1003"（不给 LIST 时用）
#   SUBDIRS   默认 "undistort_rgb undistort_depth"
#   WORK      本地临时目录，默认 /cache/ase_pack（结束后删除）
#   DRY=1     只列出要取的文件数和 OBS 上是否存在，不下载不打包
# 以 - 开头的参数忽略（ModelArts 会追加 --HCCL_* 等超参数）。最后一行 "==> finished rc=0" 表示成功。
set -uo pipefail
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
[ -f "${CODE_DIR}/run_sh_s3/_pip_deps.sh" ] && source "${CODE_DIR}/run_sh_s3/_pip_deps.sh"
for arg in "$@"; do case "${arg}" in -*) echo "[INFO] ignoring argument ${arg}" ;; esac; done

SRC_ROOT="${SRC_ROOT:-obs://bucket-471/y00959808/data/sr_datasets/processed/ASE}"
OUT_DIR="${OUT_DIR:-/efs-mi-pay-gy1/z00533370/transfer}"
SCENES="${SCENES:-10000 10001 10004 10006 10014 10018 10025 10026 10028 1003}"
SUBDIRS="${SUBDIRS:-undistort_rgb undistort_depth}"
WORK="${WORK:-/cache/ase_pack}"
LIST="${LIST:-}"
DRY="${DRY:-0}"
mkdir -p "${OUT_DIR}" "${WORK}" || { echo "[ERROR] cannot create ${OUT_DIR} or ${WORK}"; exit 1; }
LOGFILE="${OUT_DIR}/ase_pack_$(date +%m%d_%H%M%S).log"
ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%F %T')" "${line}"; done; }

python - "${SRC_ROOT}" "${WORK}/ASE" "${LIST}" "${SCENES}" "${SUBDIRS}" "${DRY}" <<'PY' 2>&1 | grep -a --line-buffered -vE "^INFO:root" | ts | tee -a "${LOGFILE}"
import os, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import moxing as mox

src, dst, lst, scenes, subdirs, dry = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4].split(), sys.argv[5].split(), sys.argv[6] == "1"
t0 = time.time()
if lst:
    rel = [l.strip() for l in open(lst) if l.strip() and not l.startswith("#")]
    print(f"[list] {lst}: {len(rel)} files")
else:
    rel = []
    for sc in scenes:
        for sub in subdirs:
            d = f"{src}/{sc}/{sub}"
            if not mox.file.exists(d):
                print(f"[WARN] missing on OBS: {d}"); continue
            names = sorted(x.rstrip("/").split("/")[-1] for x in mox.file.list_directory(d))
            rel += [f"{sc}/{sub}/{n}" for n in names]
    print(f"[scenes] {len(scenes)} scenes x {subdirs}: {len(rel)} files")
per_scene = Counter(r.split("/")[0] for r in rel)
print("  per scene: " + ", ".join(f"{k} {v}" for k, v in sorted(per_scene.items())))

def fetch(r):
    s, d = f"{src}/{r}", os.path.join(dst, r)
    err = None
    for attempt in range(3):                     # moxing 并发时偶发内部异常（如 KeyError），重试
        try:
            if dry:
                if mox.file.exists(s):
                    return r, True
                err = "not found on OBS"
                break
            if os.path.exists(d) and os.path.getsize(d) == mox.file.get_size(s):
                return r, True
            os.makedirs(os.path.dirname(d), exist_ok=True)
            mox.file.copy(s, d)
            return r, True
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(1 + attempt)
    print(f"  fail {r}: {err}")
    return r, False

mox.file.exists(src)                             # 先单线程初始化 moxing，再开多线程（32 线程同时首次调用会触发 KeyError: '253'）

with ThreadPoolExecutor(32) as ex:
    res = list(ex.map(fetch, rel))
bad = [r for r, ok in res if not ok]
if not lst:                                      # 整目录模式：场景目录在 OBS 上不存在也算失败
    bad += [f"{sc}/{sub}/" for sc in scenes for sub in subdirs if not any(r.startswith(f"{sc}/{sub}/") for r in rel)]
print(f"[fetch] {'checked' if dry else 'downloaded'} {len(res) - len(bad)}/{len(res)} files in {time.time() - t0:.0f}s"
      + (f"; missing/failed {len(bad)}, e.g. {bad[:5]}" if bad else ""))
sys.exit(1 if bad else 0)
PY
rc=${PIPESTATUS[0]}
if [ "${rc}" -ne 0 ] || [ "${DRY}" = "1" ]; then
    echo "==> finished rc=${rc}$( [ "${DRY}" = "1" ] && echo ' (DRY)')" | ts | tee -a "${LOGFILE}"
    exit "${rc}"
fi

{
    echo "==> tar ${WORK}/ASE -> ${OUT_DIR}/ase_subset.tar.gz"
    tar czf "${OUT_DIR}/ase_subset.tar.gz" -C "${WORK}/ASE" . && \
    echo "==> $(du -h "${OUT_DIR}/ase_subset.tar.gz" | cut -f1)  $(tar tzf "${OUT_DIR}/ase_subset.tar.gz" | grep -vc '/$') files  md5 $(md5sum "${OUT_DIR}/ase_subset.tar.gz" | cut -d' ' -f1)"
} 2>&1 | ts | tee -a "${LOGFILE}"
rc=${PIPESTATUS[0]}
[ "${rc}" -eq 0 ] && rm -rf "${WORK:?}/ASE"
echo "==> finished rc=${rc}" | ts | tee -a "${LOGFILE}"
exit "${rc}"
