#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path


def sanitize_folder_name(name: str) -> str:
    """
    清理目录名：
    1. 去掉 Windows 回车符 \\r
    2. 去掉首尾空白
    """
    new_name = name.replace("\r", "").strip()
    return new_name


def get_non_conflicting_name(parent: Path, target_name: str, original_path: Path) -> str:
    """
    如果 target_name 已存在且不是原目录本身，则自动追加后缀避免冲突。
    """
    candidate = parent / target_name

    if not candidate.exists() or candidate.resolve() == original_path.resolve():
        return target_name

    idx = 1
    while True:
        new_name = f"{target_name}_re{idx}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_name
        idx += 1


def rename_top_level_folders(root: Path, dry_run: bool = False) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    entries = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

    total = 0
    changed = 0
    skipped = 0

    for folder in entries:
        total += 1
        old_name = folder.name
        sanitized = sanitize_folder_name(old_name)

        # 清理后为空，给一个兜底名字
        if sanitized == "":
            sanitized = "unnamed_folder"

        final_name = get_non_conflicting_name(folder.parent, sanitized, folder)

        if final_name == old_name:
            print(f"[SKIP] {old_name!r}")
            skipped += 1
            continue

        old_path = folder
        new_path = folder.parent / final_name

        print(f"[RENAME] {old_name!r} -> {final_name!r}")
        if not dry_run:
            old_path.rename(new_path)

        changed += 1

    print("\nDone.")
    print(f"Total folders : {total}")
    print(f"Renamed       : {changed}")
    print(f"Skipped       : {skipped}")
    if dry_run:
        print("Mode          : dry-run (no changes made)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename top-level folders by removing CR/trailing spaces and resolving name conflicts."
    )
    parser.add_argument(
        "--root",
        type=str,
        default='/efs-m3ga/z00533370/reliev3r_data/ADT/',
        help="Root directory containing the folders to rename.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually renaming.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    rename_top_level_folders(root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()