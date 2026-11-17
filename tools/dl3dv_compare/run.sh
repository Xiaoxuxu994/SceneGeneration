#!/usr/bin/env bash
# DL3DV 新旧标签对比。
#
# 一、6 个已渲染场景 × 12 帧的细看（scenes_s256_view6.txt / frames_s256_view6.json），结束后各打一个 tar.gz：
#   bash tools/dl3dv_compare/run.sh old    # 旧标签（Pi3 训练读的 DL3DV_WAI）：能挂 /efs-mi-east4-2 的机器、训练用的 python 环境
#   bash tools/dl3dv_compare/run.sh new    # 新标签（GLR-v3 结果 tar）：能读 obs://bucket-471 的机器（ROMA，镜像带 moxing）
#
# 二、dl3dv_s256 全部 256 个场景的扫描：找「旧 mvsanywhere 标签疑似失败、新管线成功」的场景。
#    三步按顺序跑，OUT 默认 <仓库>/dl3dv_cmp_s256_all（两台机器都能看到的共享盘上，中间不用传文件）：
#   bash tools/dl3dv_compare/run.sh scan-new   # ROMA：抽新标签，并决定每场景抽哪 12 帧（写 OUT/frames.json）；GLR 没打完的场景记为失败
#   bash tools/dl3dv_compare/run.sh scan-old   # devserver：按同一份 frames.json 抽旧标签，外加每帧的第 +4 帧旧深度（查旧深度自身一致性）
#   bash tools/dl3dv_compare/run.sh scan       # devserver：打分排序，前 24 个疑似场景 + 4 个对照出图，打包 OUT/scan
#
# 可选环境变量：PY（默认 python）、OUT、WAI_ROOT、RGB_ROOT、RESULTS（结果 tar 前缀）、
#              WORK（new 下载 tar 的临时目录，单个 tar 约 1.5 GB，用完即删）、SHARDS（scan-new/scan-old 并行进程数，默认 4）、
#              TOP（scan 出图的场景数，默认 24）
set -euo pipefail
what=${1:?用法: run.sh old|new|scan-new|scan-old|scan}
cd "$(dirname "$0")/../.."
PY=${PY:-python}
D=tools/dl3dv_compare
T=tools/dl3dv_label_compare.py
RESULTS=${RESULTS:-obs://bucket-471/z00533370/dataset/DL3DV-test/glr/dl3dv_s256/results}
WAI=(--wai-root "${WAI_ROOT:-/efs-mi-east4-2/datasets/DL3DV_WAI}" --rgb-root "${RGB_ROOT:-/efs-mi-east4-2/datasets/DL3DV}")
case "$what" in
  old|new) OUT=${OUT:-$PWD/dl3dv_cmp_s256_view6} ;;
  *)       OUT=${OUT:-$PWD/dl3dv_cmp_s256_all} ;;
esac
mkdir -p "$OUT"
echo "code $(git rev-parse --short HEAD 2>/dev/null || echo '?')  host $(hostname)  out $OUT"

status() {   # 汇总 status_<what>*.json（分片时每片一个文件）
  "$PY" - "$OUT" "$1" <<'PY'
import json, sys, glob, os
out, what = sys.argv[1:]
st = {}
for f in sorted(glob.glob(os.path.join(out, f'status_{what}*.json'))):
    for r in json.load(open(f)):
        st[r['sid']] = r
bad = [s for s in st.values() if s['status'] == 'error']
print(f"{len(st) - len(bad)}/{len(st)} 个场景成功（含已存在跳过的）")
for s in bad[:30]:
    print('  失败', s['sid'], s['error'][:160])
if len(bad) > 30:
    print(f'  ……共 {len(bad)} 个失败')
PY
}

sharded() {  # sharded <tag> <python 参数...>：开 SHARDS 个进程并行，各写各的日志，全部结束后返回
  local tag=$1; shift
  local n=${SHARDS:-4} pids=()
  for i in $(seq 0 $((n - 1))); do
    "$PY" "$T" "$@" --shard "$i/$n" > "$OUT/log_${tag}_s${i}.txt" 2>&1 &
    pids+=($!)
  done
  echo "已启动 $n 个进程，日志 $OUT/log_${tag}_s*.txt"
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc -eq 0 ] || echo "有分片进程非零退出，看日志" >&2
}

case "$what" in
  old)
    "$PY" "$T" old --scenes "$D/scenes_s256_view6.txt" --frames-from "$D/frames_s256_view6.json" --out "$OUT" "${WAI[@]}"
    status old ;;
  new)
    "$PY" "$T" new --scenes "$D/scenes_s256_view6.txt" --frames-from "$D/frames_s256_view6.json" --out "$OUT" \
      --results "$RESULTS" ${WORK:+--work "$WORK"}
    status new ;;
  scan-new)
    sharded new new --scenes "$D/scenes_s256_all.txt" --out "$OUT" --n-frames 12 --results "$RESULTS" ${WORK:+--work "$WORK"}
    status new
    echo "下一步（devserver）：bash $D/run.sh scan-old" ;;
  scan-old)
    [ -f "$OUT/frames.json" ] || { echo "没有 $OUT/frames.json：先在 ROMA 上跑 scan-new（两边要抽同一组帧）" >&2; exit 2; }
    sharded old old --scenes "$D/scenes_s256_all.txt" --out "$OUT" --neighbor-offset 4 "${WAI[@]}"
    status old
    echo "下一步（devserver）：bash $D/run.sh scan" ;;
  scan)
    "$PY" "$T" scan --out "$OUT" --top "${TOP:-24}"
    pkg="$(dirname "$OUT")/dl3dv_scan_$(hostname -s)_$(date +%m%d_%H%M).tar.gz"
    tar -czf "$pkg" -C "$OUT" scan frames.json $(cd "$OUT" && ls status_*.json)
    echo "打包完成：$pkg  $(du -h "$pkg" | cut -f1)" ;;
  *) echo "用法: run.sh old|new|scan-new|scan-old|scan" >&2; exit 2 ;;
esac
if [ "$what" = old ] || [ "$what" = new ]; then
  pkg="$(dirname "$OUT")/dl3dv_cmp_${what}_$(hostname -s)_$(date +%m%d_%H%M).tar.gz"
  tar -czf "$pkg" -C "$(dirname "$OUT")" "$(basename "$OUT")/$what" "$(basename "$OUT")/frames.json" "$(basename "$OUT")/status_$what.json"
  echo "打包完成：$pkg  $(du -h "$pkg" | cut -f1)"
fi
