# scripts/_bootstrap.py
from __future__ import annotations
import sys
from pathlib import Path

# benchmark/
BENCH_ROOT = Path(__file__).resolve().parents[1]

def setup_pythonpath() -> None:
    """
    Make project imports work no matter where you run the script from.
    """
    candidates = [
        BENCH_ROOT,                              # allow import from benchmark/*
        BENCH_ROOT / "python",                   # if you have local python packages here
        BENCH_ROOT / "lib",                      # optional local libs
        BENCH_ROOT / "models" / "vggt_official", # `import vggt` 
        BENCH_ROOT / "models" / "reliev3r", # `import reliev3r` 
        BENCH_ROOT / "models" / "pi3_official", # `import pi3` 
        BENCH_ROOT / "models" / "depth_anything_3", # `import da3` 
    ]
    for p in candidates:
        if p.exists():
            sys.path.insert(0, str(p))

setup_pythonpath()
