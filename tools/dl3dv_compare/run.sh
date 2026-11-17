#!/usr/bin/env bash
# DL3DV 新旧标签对比：dl3dv_s256 里 6 个已渲染场景、每场景固定 12 帧（scenes_s256_view6.txt / frames_s256_view6.json）。
#
#   bash tools/dl3dv_compare/run.sh old    # 旧标签（Pi3 训练读的 DL3DV_WAI）：在能挂 /efs-mi-east4-2 的机器、训练用的 python 环境里
#   bash tools/dl3dv_compare/run.sh new    # 新标签（GLR-v3 结果 tar）：在能读 obs://bucket-471 的机器（ROMA，镜像带 moxing）
#
# 结束后打一个 tar.gz（旧约 50-100 MB，新约 50-150 MB），路径打印在最后一行，传回来即可。
# 可选环境变量：PY（默认 python）、OUT（默认 <仓库>/dl3dv_cmp_s256_view6）、WAI_ROOT、RGB_ROOT、
#              RESULTS（结果 tar 前缀）、WORK（new 下载 tar 的临时目录，单个 tar 约 1.5 GB，用完即删）
set -euo pipefail
what=${1:?用法: run.sh old|new}
cd "$(dirname "$0")/../.."
PY=${PY:-python}
OUT=${OUT:-$PWD/dl3dv_cmp_s256_view6}
D=tools/dl3dv_compare
common=(--scenes "$D/scenes_s256_view6.txt" --frames-from "$D/frames_s256_view6.json" --out "$OUT")
echo "code $(git rev-parse --short HEAD 2>/dev/null || echo '?')  host $(hostname)  out $OUT"
case "$what" in
  old)
    "$PY" tools/dl3dv_label_compare.py old "${common[@]}" \
      --wai-root "${WAI_ROOT:-/efs-mi-east4-2/datasets/DL3DV_WAI}" --rgb-root "${RGB_ROOT:-/efs-mi-east4-2/datasets/DL3DV}" ;;
  new)
    "$PY" tools/dl3dv_label_compare.py new "${common[@]}" \
      --results "${RESULTS:-obs://bucket-471/z00533370/dataset/DL3DV-test/glr/dl3dv_s256/results}" ${WORK:+--work "$WORK"} ;;
  *) echo "用法: run.sh old|new" >&2; exit 2 ;;
esac
"$PY" - "$OUT/status_$what.json" <<'EOF'
import json, sys
st = json.load(open(sys.argv[1]))
bad = [s for s in st if s['status'] == 'error']
print(f"{len(st) - len(bad)}/{len(st)} 个场景成功")
for s in bad:
    print('  失败', s['sid'], s['error'])
EOF
pkg="$(dirname "$OUT")/dl3dv_cmp_${what}_$(hostname -s)_$(date +%m%d_%H%M).tar.gz"
tar -czf "$pkg" -C "$(dirname "$OUT")" "$(basename "$OUT")/$what" "$(basename "$OUT")/frames.json" "$(basename "$OUT")/status_$what.json"
echo "打包完成：$pkg  $(du -h "$pkg" | cut -f1)"
