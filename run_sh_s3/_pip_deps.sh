# 公共依赖检查：所有 run_sh_s3 下的启动脚本在 cd 到代码目录后第一件事 source 它。
#   source "${CODE_DIR}/run_sh_s3/_pip_deps.sh"
# 只安装 import 不到的包（已有的不动，不会升级或替换镜像里的版本）；torch / torchvision / torch_npu / moxing
# 由镜像提供，这里只检查、不安装（装错版本会破坏 NPU 环境）。新增依赖时加到下面的 PIP_DEPS（模块名:pip 包名）。
# 额外的包可在启动命令前用 EXTRA_PIP_DEPS="模块:包 ..." 追加。
PIP_DEPS="hydra:hydra-core omegaconf:omegaconf hydra_colorlog:hydra_colorlog evo:evo pyquaternion:pyquaternion \
yaml:pyyaml cv2:opencv-python-headless PIL:pillow scipy:scipy pandas:pandas h5py:h5py imageio:imageio \
tqdm:tqdm matplotlib:matplotlib OpenEXR:OpenEXR ${EXTRA_PIP_DEPS:-}"

python - ${PIP_DEPS} <<'PY' || echo "[deps] WARNING: 依赖安装失败，见上方输出（缺的包会在 import 时报错）"
import importlib, subprocess, sys
pairs = [a.split(":", 1) for a in sys.argv[1:] if ":" in a]
missing = []
for mod, pkg in pairs:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pkg)
for mod in ("torch", "torchvision", "moxing"):
    try:
        importlib.import_module(mod)
    except Exception as e:
        print(f"[deps] WARNING: {mod} 不可用（由镜像提供，不自动安装）：{type(e).__name__}: {e}", flush=True)
if missing:
    print(f"[deps] pip install {' '.join(missing)}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("[deps] all present", flush=True)
PY
