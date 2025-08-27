import os
import argparse

import open3d as o3d

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", type=str, required=True)
parser.add_argument("--out_path", type=str, required=True)
args = parser.parse_args()

mesh = o3d.io.read_triangle_mesh(args.in_path)
print(mesh)
# o3d.io.write_point_cloud(args.out_path, mesh)