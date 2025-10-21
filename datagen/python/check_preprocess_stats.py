import os
import argparse

import multiprocessing as mp

from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--out_file", type=str, required=True)
parser.add_argument("--num_processes", type=int, default=32)

args = parser.parse_args()

in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir)

in_file_list = list(in_dir.iterdir())
out_file_list = list(out_dir.iterdir())

def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]

def worker_job(job):
    num_processed = 0
    for sub_dir in job:
        if (sub_dir / args.out_file).exists():
            num_processed += 1
            
    return num_processed

job_list = chunks(out_file_list, args.num_processes)
with mp.Pool(processes=32) as pool:
    results = pool.map(worker_job, job_list)

print("current preprocessing stats: {}/{}".format(sum(results), len(in_file_list)))