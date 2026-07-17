from omegaconf import OmegaConf

from _bootstrap import BENCH_ROOT 

DATA_CONFIGS = {
    "ase": BENCH_ROOT / "config" / "ase.yaml",
    "dl3dv": BENCH_ROOT / "config" / "dl3dv.yaml",
}