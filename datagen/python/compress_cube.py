import os
import shutil
import argparse

import numpy as np
import multiprocessing as mp

from pathlib import Path

import sys
sys.path.append(os.path.dirname(__file__))
from lib.utils.hash import encode_coords, decode_coords

VOLUME_FILE = "vs_{}.npy"
OCTREE_FILE = "vs{}_to_vs{}_th{}.npy"

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--out_dir_sec", type=str, default="", help="secondary output directory.")
parser.add_argument("--src_size", type=int, required=True)
parser.add_argument("--tar_size", type=int, required=True)
parser.add_argument("--thres", type=int, required=True)
parser.add_argument("--check", action="store_true", help="check octree.")
parser.add_argument("--stats", action="store_true", help="check compression stats.")
parser.add_argument("--force", action="store_true", help="generate octree even if target exists.")
parser.add_argument("--num_workers", type=int, default=1)
args = parser.parse_args()

in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir)

out_dir.mkdir(exist_ok=True, parents=True)

assert args.tar_size > args.src_size, \
    "{} must be greater than {}".format(args.tar_size, args.src_size)

def worker_job(input_dir):
    src_path = input_dir / VOLUME_FILE.format(args.src_size)
    tar_path = input_dir / VOLUME_FILE.format(args.tar_size)
    
    if not src_path.exists():
        print("{} doesn't exist.".format(str(src_path)))
    
    if not tar_path.exists():
        print("{} doesn't exist.".format(str(tar_path)))
    
    output_dir = out_dir / input_dir.name
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / OCTREE_FILE.format(args.src_size, args.tar_size, args.thres)
    
    if output_path.exists() and not args.force:
        print("found existing octree file {}, skipping...".format(str(output_path)))
    else:
        print("processing {}...".format(str(input_dir)))
        
        src_vol = np.load(src_path)
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
                src_coords = encode_coords(origin_i, origin_j, origin_k) # coordinates in low-res volume
                tar_coords = encode_coords(i_expanded[p], j_expanded[p], k_expanded[p]) # coordinates in high-res volume
                tar_value = tar_vol[i_expanded[p], j_expanded[p], k_expanded[p]] # value in high-res volume
                
                compressed.append([src_coords, tar_coords, tar_value])
                
                if args.check:
                    _, coords, value = compressed[-1]
                    i, j, k = decode_coords(int(coords))
                    original_value = tar_vol[i, j, k]
                    
                    if original_value != value:
                        print("the original value {} and the compressed value {} doesn't match at {}, {}, {}".format(
                            original_value, value, i, j, k
                        ))
                        break
                
        compressed = np.array(compressed)
        np.save(output_path, compressed)
        print("save to {}".format(str(output_path)))
        
    if len(args.out_dir_sec) > 0:
        out_dir_sec = Path(args.out_dir_sec)
        output_dir_sec = out_dir_sec / input_dir.name
        output_dir_sec.mkdir(exist_ok=True, parents=True)
        output_path_sec = output_dir_sec / OCTREE_FILE.format(args.src_size, args.tar_size, args.thres)
        if not output_path_sec.exists():
            print("copy to secondary output directory {}".format(args.out_dir_sec))
            shutil.copy(output_path, output_path_sec)

if __name__ == "__main__":
    in_dir_list = list(in_dir.iterdir())
    assert len(in_dir_list) > 0, "empty input data list"
    
    if args.num_workers > 1:
        pool = mp.Pool(processes=args.num_workers)
        pool.map(worker_job, in_dir_list)
    else:
        for in_dir in in_dir_list:
            # for debugging
            # in_dir = Path("data/nvme2/scannetpp_volume_norm_-1_1_0.95/sdf/836363121f")
            worker_job(in_dir)
            # exit()
            
    if args.stats:
        sizes = []
        for in_dir in in_dir_list:
            tar_path = in_dir / VOLUME_FILE.format(args.tar_size)
            output_dir = out_dir / in_dir.name
            output_path = output_dir / OCTREE_FILE.format(args.src_size, args.tar_size, args.thres)
            
            original_size = os.path.getsize(tar_path)
            compressed_size = os.path.getsize(output_path)
            sizes.append([original_size, compressed_size])
            
        sizes = np.array(sizes) / (1024*1024) # convert to MB
        rates = (sizes[:, 0] - sizes[:, 1]) / sizes[:, 0]
        
        print("\nstats:")
        print("original size: min -> {:.2f} MB, max -> {:.2f} MB, avg -> {:.2f} MB".format(sizes[:, 0].min(), sizes[:, 0].max(), sizes[:, 0].mean()))
        print("compress size: min -> {:.2f} MB, max -> {:.2f} MB, avg -> {:.2f} MB".format(sizes[:, 1].min(), sizes[:, 1].max(), sizes[:, 1].mean()))
        print("compress rate: min -> {:.2f} %, max -> {:.2f} %, avg -> {:.2f} %".format(rates.min(), rates.max(), rates.mean()))
