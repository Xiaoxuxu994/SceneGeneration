from pathlib import Path
import trimesh
from tqdm import tqdm
import multiprocessing

def get_valid_rooms(mesh_dir, limit=None):
    list_of_scenes = Path(mesh_dir).iterdir()
    list_of_rooms = []
    for scene in list_of_scenes:
        for room in scene.iterdir():
            if (room / "mesh.obj").exists():
                list_of_rooms.append(room)
    return list_of_rooms[:limit]


def worker(area_list, items):
    for room in items:
        try:
            bbox = trimesh.load(room / "mesh.obj").bounding_box.bounds
        except Exception as e:
            print(f"Error with {room}: {e}")
            continue
        area = (bbox[1] - bbox[0])[0] * (bbox[1] - bbox[0])[2]
        area_list.append((room, area))


if __name__ == '__main__':
    import sys
    from shutil import rmtree
    delete_top_paths = False
    valid_rooms = get_valid_rooms(sys.argv[1])
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
        if delete_top_paths:
            rmtree(m[0])