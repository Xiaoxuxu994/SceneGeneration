#!/usr/bin/env bash
# 从 RE10K_tar（tools/re10k_to_tar.py 的输出）里随机抽 N 个场景，解开成目录后打成一个 tar 放到 efs，方便传出去。
#   源：  ${SRC}/<key>.tar + ${SRC}/_index.json（默认 obs://bucket-471/z00533370/dataset/RE10K_tar/test）
#   输出：${OUT_DIR}/re10k_<split>_sample<N>_seed<SEED>.tar，包内：
#           <key>/images/000000.jpg ...   <key>/meta.json（像素内参 intrinsics、OpenCV w2c、timestamps、url ...）
#           sample_list.txt               抽到的 key 及帧数 / 宽高（同一 SEED 抽样结果固定，可复现）
#
# 开发机直接跑（nohup），或 ModelArts（云道）作业，单节点即可：
#   cd /home/ma-user/modelarts/user-job-dir/SceneGeneration/; source /home/ma-user/.bashrc; bash run_sh_s3/run_utils/re10k_sample_pack.sh
# 可用环境变量：
#   SRC         默认 obs://bucket-471/z00533370/dataset/RE10K_tar/test
#   N=40        抽几个场景
#   SEED=0      随机种子
#   MIN_FRAMES=32  只从帧数 >= 这个值的场景里抽（RE10K 有只有 1 帧的场景）
#   OUT_DIR     默认 /efs-mi-pay-gy1/z00533370/transfer
#   WORK        本地临时目录，默认 /cache/re10k_sample（结束后删除）
#   EXTRACT=0   不解开，直接把 <key>.tar 原样打包
# 以 - 开头的参数忽略（ModelArts 会追加 --HCCL_* 等超参数）。最后一行 "==> finished rc=0" 表示成功。
set -uo pipefail
for arg in "$@"; do case "${arg}" in -*) echo "[INFO] ignoring argument ${arg}" ;; esac; done

SRC="${SRC:-obs://bucket-471/z00533370/dataset/RE10K_tar/test}"
N="${N:-40}"
SEED="${SEED:-0}"
MIN_FRAMES="${MIN_FRAMES:-32}"
OUT_DIR="${OUT_DIR:-/efs-mi-pay-gy1/z00533370/transfer}"
WORK="${WORK:-/cache/re10k_sample}"
EXTRACT="${EXTRACT:-1}"
SPLIT="$(basename "${SRC%/}")"
NAME="re10k_${SPLIT}_sample${N}_seed${SEED}"
mkdir -p "${OUT_DIR}" "${WORK}/${NAME}" || { echo "[ERROR] cannot create ${OUT_DIR} or ${WORK}"; exit 1; }
LOGFILE="${OUT_DIR}/${NAME}_$(date +%m%d_%H%M%S).log"
ts() { while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%F %T')" "${line}"; done; }

python - "${SRC%/}" "${WORK}/${NAME}" "${N}" "${SEED}" "${MIN_FRAMES}" "${EXTRACT}" <<'PY' 2>&1 | grep --line-buffered -vE "^INFO:root" | ts | tee -a "${LOGFILE}"
import io, json, os, random, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor
import moxing as mox

src, dst, n, seed, min_frames, extract = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] == "1"
t0 = time.time()
index = json.loads(mox.file.read(f"{src}/_index.json"))       # {key: [帧数, 宽, 高]}
pool = sorted(k for k, v in index.items() if v[0] >= min_frames)
print(f"[index] {src}: {len(index)} clips, {len(pool)} with >= {min_frames} frames")
keys = sorted(random.Random(seed).sample(pool, min(n, len(pool))))
if not keys:
    print("[ERROR] no clip to sample (check SRC / MIN_FRAMES)")
    sys.exit(1)
with open(os.path.join(dst, "sample_list.txt"), "w") as f:
    f.write(f"# {src}  seed={seed}  min_frames={min_frames}\n# key frames width height\n")
    f.writelines(f"{k} {index[k][0]} {index[k][1]} {index[k][2]}\n" for k in keys)

def fetch(k):
    for i in range(3):
        try:
            if extract:
                with tarfile.open(fileobj=io.BytesIO(mox.file.read(f"{src}/{k}.tar", binary=True))) as t:
                    t.extractall(os.path.join(dst, k))
            else:
                mox.file.copy(f"{src}/{k}.tar", os.path.join(dst, f"{k}.tar"))
            return k, None
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(1 + i)
    return k, err

with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(fetch, keys))
bad = [(k, e) for k, e in res if e]
for k, e in bad:
    print(f"  fail {k}: {e}")
print(f"[fetch] {len(keys) - len(bad)}/{len(keys)} clips, {sum(index[k][0] for k in keys)} frames in {time.time() - t0:.0f}s")
sys.exit(1 if bad else 0)
PY
rc=${PIPESTATUS[0]}
if [ "${rc}" -ne 0 ]; then
    echo "==> finished rc=${rc}" | ts | tee -a "${LOGFILE}"
    exit "${rc}"
fi

{
    # jpg 已经是压缩过的，再 gzip 基本不变小，直接打 tar
    echo "==> tar ${WORK}/${NAME} -> ${OUT_DIR}/${NAME}.tar"
    tar cf "${OUT_DIR}/${NAME}.tar" -C "${WORK}" "${NAME}" && chmod 777 "${OUT_DIR}/${NAME}.tar" && \
    echo "==> $(du -h "${OUT_DIR}/${NAME}.tar" | cut -f1)  $(tar tf "${OUT_DIR}/${NAME}.tar" | grep -vc '/$') files  md5 $(md5sum "${OUT_DIR}/${NAME}.tar" | cut -d' ' -f1)"
} 2>&1 | ts | tee -a "${LOGFILE}"
rc=${PIPESTATUS[0]}
[ "${rc}" -eq 0 ] && rm -rf "${WORK:?}/${NAME}"
echo "==> finished rc=${rc}" | ts | tee -a "${LOGFILE}"
exit "${rc}"
