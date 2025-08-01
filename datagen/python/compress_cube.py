import os
import argparse

import numpy as np
import multiprocessing as mp

from pathlib import Path

VOLUME_FILE = "vs_{}.npy"
OCTREE_FILE = "vs{}_to_vs{}_th{}.npy"

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--src_size", type=int, required=True)
parser.add_argument("--tar_size", type=int, required=True)
parser.add_argument("--thres", type=int, required=True)
parser.add_argument("--hash", type=int, default=1000)
parser.add_argument("--num_workers", type=int, default=1)
args = parser.parse_args()

in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir)

out_dir.mkdir(exist_ok=True, parents=True)

assert args.tar_size > args.src_size, \
    "{} must be greater than {}".format(args.tar_size, args.src_size)

def worker_job(input_dir):
    print("processing {}...".format(str(input_dir)))
    src_path = input_dir / VOLUME_FILE.format(args.src_size)
    src_vol = np.load(src_path)
    
    tar_path = input_dir / VOLUME_FILE.format(args.tar_size)
    tar_vol = np.load(tar_path)
    
    src_ids = np.where(src_vol <= (2 / args.src_size) * args.thres)
    num_src_ids = src_ids[0].shape[0]
    
    rate = args.tar_size // args.src_size    
    offsets = np.meshgrid(np.arange(rate), np.arange(rate), np.arange(rate))
    num_offsets = offsets[0].shape[0]
    
    # lazy implementation
    compressed = []
    for row in range(num_src_ids):
        origin_i, origin_j, origin_k = src_ids[0][row], src_ids[1][row], src_ids[2][row]
        i_expanded = offsets[0].reshape(-1) + origin_i * rate
        j_expanded = offsets[1].reshape(-1) + origin_j * rate
        k_expanded = offsets[2].reshape(-1) + origin_k * rate
        for p in range(num_offsets):
            src_coords = origin_i + origin_j * args.hash + origin_k * args.hash * args.hash
            
            tar_coords = i_expanded[p] + j_expanded[p] * args.hash + k_expanded[p] * args.hash * args.hash
            tar_value = tar_vol[i_expanded[p], j_expanded[p], k_expanded[p]]

            compressed.append([src_coords, tar_coords, tar_value])
            
    compressed = np.array(compressed)
    
    output_path = input_dir / OCTREE_FILE.format(args.src_size, args.tar_size, args.thres)
    np.save(output_path, compressed)
    print("process complete.")

if __name__ == "__main__":
    in_dir_list = list(in_dir.iterdir())
    assert len(in_dir_list) > 0, "empty input data list"
    
    if args.num_workers > 1:
        pool = mp.Pool(processes=args.num_workers)
        pool.map(worker_job, in_dir_list)
    else:
        for in_dir in in_dir_list:
            # for debugging
            worker_job(in_dir)
            exit()

