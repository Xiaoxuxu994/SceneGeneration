#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="/mnt/nvme3/z00533370/dataset/scannetpp/data/",
        help="Directory containing scene folders",
    )
    ap.add_argument(
        "--out_txt",
        type=str,
        default="/mnt/nvme3/z00533370/3drecon/datalist/scannetpp_all.txt",
        help="Output txt path (one folder name per line)",
    )
    ap.add_argument(
        "--abs",
        action="store_true",
        help="Write absolute paths instead of folder names",
    )
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise FileNotFoundError(root)

    names = []
    for n in os.listdir(root):
        p = os.path.join(root, n)
        if os.path.isdir(p):
            names.append(n)

    names.sort()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_txt)), exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        for n in names:
            f.write((os.path.join(root, n) if args.abs else n) + "\n")

    print(f"[INFO] root: {root}")
    print(f"[INFO] folders: {len(names)}")
    print(f"[INFO] wrote: {args.out_txt}")


if __name__ == "__main__":
    main()
