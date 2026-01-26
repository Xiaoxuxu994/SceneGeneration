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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default='/mnt/nvme3/z00533370/3drecon/datalist/scannetpp_train.txt', help="txt A (treat as source to check extras)")
    ap.add_argument("--b", default='/mnt/nvme3/z00533370/3drecon/datalist/scannetpp_all.txt', help="txt B (treat as reference)")
    ap.add_argument("--out_extra", default='/mnt/nvme3/z00533370/3drecon/datalist/_scannetpp_all_missing.txt', help="output txt for ids that are in A but NOT in B")
    ap.add_argument("--out_missing", default='/mnt/nvme3/z00533370/3drecon/datalist/scannetpp_val.txt',
                    help="optional: output txt for ids that are in B but NOT in A (should be empty if A contains all B)")
    args = ap.parse_args()

    a_list = read_ids(args.a)
    b_list = read_ids(args.b)

    a_set = set(a_list)
    b_set = set(b_list)

    extra = sorted(a_set - b_set)     # A has but B doesn't
    missing = sorted(b_set - a_set)   # B has but A doesn't

    with open(args.out_extra, "w", encoding="utf-8") as f:
        for x in extra:
            f.write(x + "\n")

    if args.out_missing is not None:
        with open(args.out_missing, "w", encoding="utf-8") as f:
            for x in missing:
                f.write(x + "\n")

    print(f"[INFO] A: {args.a}  ({len(a_list)} lines, {len(a_set)} unique)")
    print(f"[INFO] B: {args.b}  ({len(b_list)} lines, {len(b_set)} unique)")
    print(f"[INFO] Extra (A \\ B): {len(extra)} -> {args.out_extra}")

    if len(missing) == 0:
        print("[OK] All ids in B are contained in A (B \\ A is empty).")
    else:
        print(f"[WARN] Missing (B \\ A): {len(missing)}")
        if args.out_missing is None:
            # print a small preview
            preview = missing[:20]
            print("[WARN] First missing ids:")
            for x in preview:
                print("  " + x)
            print("      (use --out_missing to dump all)")
        else:
            print(f"[INFO] Missing list written to: {args.out_missing}")


if __name__ == "__main__":
    main()
