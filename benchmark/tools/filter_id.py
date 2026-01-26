#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse


def read_ids(txt_path: str) -> list[str]:
    ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def write_ids(txt_path: str, ids: list[str]) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        for x in ids:
            f.write(x + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="/mnt/nvme3/z00533370/3drecon/datalist/scannetpp_val.txt",
                    help="txt A (source)")
    ap.add_argument("--b", default="/mnt/nvme3/z00533370/3drecon/datalist/nvs_test.txt",
                    help="txt B (filter set; remove these ids from A)")
    ap.add_argument("--out", default="/mnt/nvme3/z00533370/3drecon/datalist/scannetpp_val_final.txt",
                    help="output txt for A \\ B (keep A order)")
    ap.add_argument("--out_intersection", default=None,
                    help="optional output for A ∩ B (keep A order)")
    ap.add_argument("--dedup", action="store_true",
                    help="deduplicate A before filtering (keep first-seen order)")
    args = ap.parse_args()

    a_list = read_ids(args.a)
    b_set = set(read_ids(args.b))

    if args.dedup:
        seen = set()
        a_list2 = []
        for x in a_list:
            if x not in seen:
                a_list2.append(x)
                seen.add(x)
        a_list = a_list2

    filtered = [x for x in a_list if x not in b_set]
    inter = [x for x in a_list if x in b_set]

    write_ids(args.out, filtered)
    print(f"[INFO] A lines: {len(a_list)}")
    print(f"[INFO] B unique: {len(b_set)}")
    print(f"[INFO] A \\ B: {len(filtered)} -> {args.out}")

    if args.out_intersection:
        write_ids(args.out_intersection, inter)
        print(f"[INFO] A ∩ B: {len(inter)} -> {args.out_intersection}")


if __name__ == "__main__":
    main()
