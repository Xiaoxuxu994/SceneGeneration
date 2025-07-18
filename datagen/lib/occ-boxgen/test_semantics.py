import numpy as np
import marching_cubes as mc
from pathlib import Path
import math

highres_dim = 64
lowres_dim = 8
padding_highres = math.ceil(highres_dim / lowres_dim)
padding_lowres = 1
highres_voxel_size = 2.6 / (highres_dim - 2 * padding_highres)
lowres_voxel_size = 2.6 / (lowres_dim - 2 * padding_lowres)

print(f"HighresVoxelRes: {highres_voxel_size}")
print(f"LowresVoxelRes: {lowres_voxel_size}")

def visualize_highres_colored():
    samples = [x for x in list(Path("/cluster/gondor/ysiddiqui/3DFrontDistanceFields/chunk_semantics").iterdir()) if Path("/rhome/ysiddiqui/repatch/data/sdf_064/3DFront", x.name).exists()]
    for sample in samples[:10]:
        df = np.load(str("/rhome/ysiddiqui/repatch/data/sdf_064/3DFront/")+sample.name)
        idf = np.load(str("/cluster/gondor/ysiddiqui/3DFrontDistanceFields/chunk_semantics/")+sample.name)
        colors = np.zeros([idf.shape[0], idf.shape[1], idf.shape[2], 3])
        colors[:, :, :, 0] = (idf[:, :, :] % 256) / 255
        colors[:, :, :, 1] = ((idf[:, :, :] // 256) % 256) / 255
        colors[:, :, :, 2] = ((idf[:, :, :] // 256 // 256) % 256) / 255
        vertices, triangles = mc.marching_cubes_color(df, colors, highres_voxel_size * 0.75)
        mc.export_obj(vertices, triangles, sample.name + ".obj")


if __name__ == "__main__":
    visualize_highres_colored()
