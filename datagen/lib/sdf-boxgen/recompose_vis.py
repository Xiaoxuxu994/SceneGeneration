import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm 
from collections import defaultdict

def get_scenes_chunk_dict(base_path, suffix):
    all_chunks = [(x.name.split(suffix)[0], "__".join(x.name.split("__")[:2])) for x in base_path.iterdir() if x.name.endswith(suffix)]
    scenes_chunk_dict = defaultdict(list)
    for chunk in all_chunks:
        scenes_chunk_dict[chunk[1]].append(chunk[0])
    return scenes_chunk_dict


def recompose_scene(base_path, chunks, suffix, shift):
    xyz = np.array([[int(y) for y in x.split("__")[-1].split("_")] for x in chunks])
    meshes = [trimesh.load(base_path / (chunk + suffix), force='mesh') for chunk in chunks]
    non_empty_meshes = [type(x) == trimesh.Trimesh for x in meshes]
    joining_shift = [np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]
    for i in range(len(meshes)):
        if non_empty_meshes[i]:
            meshes[i].apply_translation(xyz[i, :])
            for j in range(3):
                if xyz[i, j] != 0:
                    meshes[i].apply_translation(joining_shift[j] * (xyz[i, j] // 64)) 
    if np.array(non_empty_meshes).any():
        try:
            meshes = [m for m in meshes if type(m) == trimesh.Trimesh]
            concat_mesh = trimesh.util.concatenate(meshes)
            concat_mesh.apply_translation(shift)
            return concat_mesh
        except Exception as e:
            print("Exception: ", e)
            return None
    else:
        return None


def recompose_chunks_to_scenes(base_path, suffix, shift):
    scenes_chunk_dict = get_scenes_chunk_dict(base_path, suffix)
    for scene in tqdm(sorted(scenes_chunk_dict.keys())):
        rescene = recompose_scene(base_path, scenes_chunk_dict[scene], suffix, shift)
        if rescene is not None:
            rescene.export(base_path / (scene + ".obj"))
            print('exported')



if __name__ == "__main__":
    recompose_chunks_to_scenes(Path("chunk_highres"), "_vis.obj", np.zeros(3))
