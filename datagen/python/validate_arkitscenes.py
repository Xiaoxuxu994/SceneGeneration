import os
import json
import argparse
import itertools

import multiprocessing as mp

from pathlib import Path
from tqdm import tqdm

GRID_256 = "vs_256.npy"
OCTREE = "vs256_to_vs512_th3.npz"
FREE = "queries_10000000.npz"

parser = argparse.ArgumentParser()
parser.add_argument("--mesh_dir", type=str, required=True)
parser.add_argument("--volume_dir", type=str, required=True)
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--num_processes", type=int, default=32)

args = parser.parse_args()

in_dir = Path(args.mesh_dir)
volume_dir = Path(args.volume_dir)

in_file_list = list(in_dir.iterdir())

def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]

def worker_job(job):
    valid_list = []
    for job_dir in job:
        scene_id = str(job_dir.stem)
        grid_256 = volume_dir / "udf" / scene_id / GRID_256
        octree = volume_dir / "octree" / scene_id / OCTREE
        free = volume_dir / "udf" / scene_id / FREE
        if (grid_256.exists() and octree.exists() and free.exists()) and \
            (grid_256.stat().st_size > 1024 and octree.stat().st_size > 1024 and free.stat().st_size > 1024):
            valid_list.append(scene_id)
            
    return valid_list

job_list = chunks(in_file_list, args.num_processes)
with mp.Pool(processes=32) as pool:
    results = pool.map(worker_job, job_list)
    results = list(itertools.chain.from_iterable(results))

with open(args.out_path, 'w') as f:
    json.dump(results, f, indent=4)
    
print("validated {}/{} data samples".format(len(results), len(in_file_list)))
