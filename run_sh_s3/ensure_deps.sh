#!/usr/bin/env bash
# 检查训练 / 数据检查需要的 Python 包，缺的才 pip 安装（已装的不动，不升级）。
# 由 run_sh_s3 下的启动脚本 source 调用；也可单独运行：bash run_sh_s3/ensure_deps.sh
# 新镜像里常缺 hydra 等包（No module named 'hydra'），在这里统一补齐。
python - <<'PY'
import importlib, subprocess, sys

# 导入名 -> pip 包名
PKGS = [
    ("hydra", "hydra-core"),
    ("hydra_colorlog", "hydra_colorlog"),
    ("omegaconf", "omegaconf"),
    ("yaml", "pyyaml"),
    ("evo", "evo"),
    ("pyquaternion", "pyquaternion"),
    ("rich", "rich"),
    ("einops", "einops"),
    ("easydict", "easydict"),
    ("accelerate", "accelerate"),
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("timm", "timm"),
    ("h5py", "h5py"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("imageio", "imageio"),
    ("plyfile", "plyfile"),
    ("prettytable", "prettytable"),
    ("matplotlib", "matplotlib"),
]

missing = []
for mod, pkg in PKGS:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pkg)

if missing:
    print(f"[deps] installing: {' '.join(missing)}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("[deps] all present", flush=True)
PY
