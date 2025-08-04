import os
import shutil
import argparse

import multiprocessing as mp

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--filename", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--force", action="store_true", help="generate octree even if target exists.")
args = parser.parse_args()

in_dir = Path(args.in_dir)

def worker_job(input_dir):
    input_path = input_dir / args.filename
    
    if input_path.exists():
        os.remove(input_path)

def main():
    oct_dir_list = list(sorted(list(in_dir.iterdir())))
    assert len(oct_dir_list) > 0, "empty input data list"
    
    if args.num_workers > 1:
        pool = mp.Pool(processes=args.num_workers)
        pool.map(worker_job, oct_dir_list)
    else:
        for oct_dir in oct_dir_list:
            # for debugging
            worker_job(oct_dir)
            exit()
        
main()