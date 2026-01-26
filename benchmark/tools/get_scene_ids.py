#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import numpy as np


def extract_scene_id_from_item(s: str) -> str:
    """
    Example item:
      /efs.../sub_000000/00777c41d4_00000
    -> scene_id = 00777c41d4
    """
    base = os.path.basename(s.rstrip("/"))
    # take substring before the last "_<digits>" suffix
    m = re.match(r"^(?P<sid>.+?)_\d+$", base)
    if m:
        return m.group("sid")
    # fallback: split by '_' and take first chunk
    return base.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_path", type=str, default="/mnt/nvme3/z00533370/3drecon/datalist/reliev3r_sample_list_c32.npz",help="Path to .npz (must contain key 'data')")
    ap.add_argument("--key", type=str, default="data", help="Key in npz to read (default: data)")
    ap.add_argument("--out_txt", type=str, default='/mnt/nvme3/z00533370/3drecon/datalist/scannetpp_train.txt', help="Optional output txt path (one scene_id per line)")
    ap.add_argument("--keep_order", action="store_true", help="Keep first-seen order (no sorting)")
    args = ap.parse_args()

    obj = np.load(args.npz_path, allow_pickle=True)
    if args.key not in obj:
        raise KeyError(f"Key '{args.key}' not found in {args.npz_path}. Keys: {list(obj.keys())}")

    arr = obj[args.key]
    # ensure iterable of strings
    items = [str(x) for x in arr.tolist()]

    if args.keep_order:
        seen = set()
        scene_ids = []
        for it in items:
            sid = extract_scene_id_from_item(it)
            if sid not in seen:
                scene_ids.append(sid)
                seen.add(sid)
    else:
        scene_ids = sorted({extract_scene_id_from_item(it) for it in items})

    print(f"[INFO] total items: {len(items)}")
    print(f"[INFO] unique scene_ids: {len(scene_ids)}")
    for sid in scene_ids:
        print(sid)

    if args.out_txt:
        with open(args.out_txt, "w", encoding="utf-8") as f:
            for sid in scene_ids:
                f.write(sid + "\n")
        print(f"[INFO] wrote: {args.out_txt}")


if __name__ == "__main__":
    main()
