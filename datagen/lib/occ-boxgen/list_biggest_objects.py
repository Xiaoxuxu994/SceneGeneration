from pathlib import Path
import trimesh
from tqdm import tqdm
import multiprocessing
import numpy as np

def get_valid_objects(mesh_dir, limit=None):
    list_of_scenes = Path(mesh_dir).iterdir()
    list_of_objects = []
    for scene in list_of_scenes:
        for room in scene.iterdir():
            objects = [Path(str(x).split(".")[0]) for x in room.iterdir() if x.name.endswith(".obj") and x.name != "mesh.obj"]
            list_of_objects.extend(objects)
    return list_of_objects[:limit]


def worker(area_list, items):
    for room in items:
        try:
            changed = False
            mesh = trimesh.load(str(room)+".obj")
            bbox = mesh.bounding_box.bounds
            scale = bbox[1][1] - bbox[0][1]
            if scale > 2.6:
                origin = bbox[0]
                mesh.apply_translation(-(bbox[0] + bbox[1]) / 2)
                mesh.apply_scale(2.6 / scale)
                mesh.apply_translation(origin - mesh.bounding_box.bounds[0])
                changed = True
            # ground
            bbox = mesh.bounding_box.bounds
            y_bounds = bbox[0][1]
            if y_bounds < 0:
                mesh.apply_translation(np.array([0, -y_bounds, 0]))
                changed = True
            if changed:
                mesh.export(str(room)+".obj")
                print(f"changed {room}")
            bbox = mesh.bounding_box.bounds
        except Exception as e:
            print(f"Error with {room}: {e}")
            continue
        area = (bbox[1] - bbox[0])[0] * (bbox[1] - bbox[0])[2]
        area_list.append((room, area))


if __name__ == '__main__':
    import sys, os
    from shutil import rmtree
    delete_top_paths = False
    valid_rooms = get_valid_objects(sys.argv[1])
    num_processes = 24
    items_per_worker = len(valid_rooms) // num_processes + 1
    process = []
    manager = multiprocessing.Manager()
    worker_results = manager.list()
    for pid in range(num_processes):
        worker_items = valid_rooms[pid * items_per_worker: (pid + 1) * items_per_worker]
        process.append(multiprocessing.Process(target=worker, args=(worker_results, worker_items)))
    for p in process:
        p.start()
    for p in process:
        p.join()
    area_list = []
    for wr in worker_results:
        area_list.append(wr)
    area_sorted = list(reversed(sorted(area_list, key=lambda x: x[1])))
    for m in area_sorted[:10]:
        print(str(m[0]), m[1], ", removing = ", delete_top_paths)