import numpy as np

from pathlib import Path

ROOT = Path("/efs-m3ga/z00533370/reliev3r_data/Spring_decode/spring/train/")

scene_list = list(ROOT.iterdir())

for scene_path in scene_list:
    focaldist_path = scene_path / "cam_data" / "focaldistance.txt"
    focaldist = np.loadtxt(focaldist_path)
    intr_path = scene_path / "cam_data" / "intrinsics.txt"
    intr = np.loadtxt(intr_path)
    print(focaldist)
    print(intr)
    exit()
    