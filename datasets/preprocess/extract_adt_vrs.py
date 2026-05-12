#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_INPUT_ROOT = (
    "/home/data/y00391175/project_xiaoxu/efs/efs-m3ga/z00533370/reliev3r_data/ADT_decode"
)


def import_pyvrs():
    """
    pyvrs docs/examples are inconsistent about import names:
    - docs examples often show: from pyvrs2 import SyncVRSReader
    - pip package is published as: vrs
    - some environments expose module name: pyvrs
    So we try a few.
    """
    errors = []

    try:
        from pyvrs2 import SyncVRSReader, ImageConversion  # type: ignore
        return SyncVRSReader, ImageConversion, "pyvrs2"
    except Exception as e:
        errors.append(f"pyvrs2: {repr(e)}")

    try:
        from pyvrs import SyncVRSReader, ImageConversion  # type: ignore
        return SyncVRSReader, ImageConversion, "pyvrs"
    except Exception as e:
        errors.append(f"pyvrs: {repr(e)}")

    try:
        import pyvrs  # type: ignore

        SyncVRSReader = getattr(pyvrs, "SyncVRSReader")
        ImageConversion = getattr(pyvrs, "ImageConversion")
        return SyncVRSReader, ImageConversion, "pyvrs(module)"
    except Exception as e:
        errors.append(f"pyvrs(module): {repr(e)}")

    raise ImportError("Could not import pyvrs bindings. Tried: " + " | ".join(errors))


def sanitize_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^0-9a-zA-Z._-]+", "_", name)
    return name[:200] if len(name) > 200 else name


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_preview(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    if x.ndim == 3 and x.shape[-1] in (3, 4):
        if x.dtype == np.uint8:
            return x
        x = x.astype(np.float32)
        finite = np.isfinite(x)
        if not finite.any():
            return np.zeros_like(x, dtype=np.uint8)
        vals = x[finite]
        lo = np.percentile(vals, 1.0)
        hi = np.percentile(vals, 99.0)
        if hi <= lo:
            hi = lo + 1e-6
        x = np.clip((x - lo) / (hi - lo), 0, 1)
        return (x * 255.0).astype(np.uint8)

    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape[:2], dtype=np.uint8)

    vals = x[finite]
    lo = np.percentile(vals, 1.0)
    hi = np.percentile(vals, 99.0)
    if hi <= lo:
        hi = lo + 1e-6

    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (x * 255.0).astype(np.uint8)


def save_array(base_path: Path, arr: np.ndarray) -> Dict[str, str]:
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    info: Dict[str, str] = {}

    try:
        if arr.dtype == np.uint8:
            png_path = base_path.with_suffix(".png")
            Image.fromarray(arr).save(png_path)
            info["image"] = str(png_path)
            return info

        if arr.dtype == np.uint16 and arr.ndim == 2:
            png_path = base_path.with_suffix(".png")
            Image.fromarray(arr).save(png_path)
            info["image"] = str(png_path)
            return info
    except Exception:
        pass

    npy_path = base_path.with_suffix(".npy")
    np.save(npy_path, arr)
    info["array"] = str(npy_path)

    preview = normalize_preview(arr)
    preview_path = base_path.with_name(base_path.name + "_preview").with_suffix(".png")
    Image.fromarray(preview).save(preview_path)
    info["preview"] = str(preview_path)

    return info


def discover_jobs(input_root: Path, modalities: List[str]) -> List[Tuple[str, Path]]:
    jobs: List[Tuple[str, Path]] = []

    if "depth" in modalities:
        for p in input_root.rglob("depth_images.vrs"):
            jobs.append(("depth", p))

    if "synthetic" in modalities:
        for p in input_root.rglob("synthetic_video.vrs"):
            jobs.append(("synthetic", p))

    return sorted(jobs, key=lambda x: str(x[1]))


def get_vrs_out_dir(vrs_path: Path, input_root: Path, output_root: Path) -> Path:
    rel_parent = vrs_path.parent.relative_to(input_root)
    vrs_tag = sanitize_name(vrs_path.stem)
    return output_root / rel_parent / vrs_tag


def is_vrs_done(vrs_path: Path, input_root: Path, output_root: Path) -> bool:
    out_dir = get_vrs_out_dir(vrs_path, input_root, output_root)
    return (out_dir / "_SUCCESS").exists()


def get_reader_stream_ids(reader) -> List[str]:
    stream_ids = getattr(reader, "stream_ids", None)
    if callable(stream_ids):
        vals = stream_ids()
    else:
        vals = stream_ids
    if vals is None:
        return []
    return [str(x) for x in vals]


def extract_array_from_image_block(block: Any) -> np.ndarray:
    """
    Robust conversion because pyvrs objects can differ a bit across builds.
    """
    if isinstance(block, np.ndarray):
        return block

    # common methods
    for name in ("to_numpy", "to_numpy_array", "numpy", "as_numpy", "toArray"):
        fn = getattr(block, name, None)
        if callable(fn):
            arr = fn()
            arr = np.asarray(arr)
            if arr.size > 0:
                return arr

    # common attributes
    for name in ("buffer", "data", "array"):
        val = getattr(block, name, None)
        if val is not None:
            arr = np.asarray(val)
            if arr.size > 0 and arr.dtype != object:
                return arr

    # final try
    arr = np.asarray(block)
    if arr.size > 0 and arr.dtype != object:
        return arr

    raise TypeError(f"Unsupported image block type: {type(block)}")


def decode_one_vrs(
    modality: str,
    vrs_path_str: str,
    input_root_str: str,
    output_root_str: str,
    stride: int,
    overwrite: bool,
    max_frames_per_stream: int,
) -> Dict[str, Any]:
    SyncVRSReader, ImageConversion, import_name = import_pyvrs()

    vrs_path = Path(vrs_path_str)
    input_root = Path(input_root_str)
    output_root = Path(output_root_str)

    out_dir = get_vrs_out_dir(vrs_path, input_root, output_root)
    ensure_dir(out_dir)

    success_flag = out_dir / "_SUCCESS"
    if success_flag.exists():
        success_flag.unlink()

    summary: Dict[str, Any] = {
        "modality": modality,
        "vrs": str(vrs_path),
        "out_dir": str(out_dir),
        "pyvrs_import": import_name,
        "num_streams": 0,
        "streams": [],
    }

    with SyncVRSReader(str(vrs_path)) as reader:
        # Ask pyvrs to decompress image blocks to numpy-friendly buffers.
        try:
            reader.set_image_conversion(ImageConversion.DECOMPRESS)
        except Exception:
            pass

        stream_ids = get_reader_stream_ids(reader)
        summary["num_streams"] = len(stream_ids)

        for sid_str in stream_ids:
            stream_dir = out_dir / sanitize_name(sid_str)
            ensure_dir(stream_dir)

            meta: Dict[str, Any] = {
                "stream_id": sid_str,
                "saved": 0,
                "skipped_existing": 0,
                "errors": 0,
                "records_seen": 0,
                "records_with_images": 0,
                "sample_errors": [],
            }

            try:
                filtered = reader.filtered_by_fields(
                    stream_ids={sid_str},
                    record_types={"data"},
                )
            except Exception as e:
                meta["errors"] += 1
                meta["sample_errors"].append(
                    {"stage": "filtered_by_fields", "error": repr(e)}
                )
                summary["streams"].append(meta)
                continue

            saved_count = 0
            for idx, record in enumerate(filtered):
                meta["records_seen"] += 1

                if stride > 1 and (idx % stride != 0):
                    continue

                image_blocks = getattr(record, "image_blocks", None)
                if not image_blocks:
                    continue

                meta["records_with_images"] += 1

                if max_frames_per_stream > 0 and saved_count >= max_frames_per_stream:
                    break

                try:
                    arr = extract_array_from_image_block(image_blocks[0])

                    # pyvrs record.timestamp is documented
                    ts = getattr(record, "timestamp", None)
                    frame_name = f"{saved_count:06d}"
                    if ts is not None:
                        try:
                            ts_us = int(float(ts) * 1e6)
                            frame_name += f"_{ts_us}"
                        except Exception:
                            frame_name += f"_{sanitize_name(str(ts))}"

                    base_path = stream_dir / frame_name

                    exists_already = (
                        base_path.with_suffix(".png").exists()
                        or base_path.with_suffix(".npy").exists()
                    )

                    if exists_already and not overwrite:
                        meta["saved"] += 1
                        meta["skipped_existing"] += 1
                        saved_count += 1
                        continue

                    save_array(base_path, arr)
                    meta["saved"] += 1
                    saved_count += 1

                except Exception as e:
                    meta["errors"] += 1
                    if len(meta["sample_errors"]) < 10:
                        meta["sample_errors"].append(
                            {
                                "record_index": idx,
                                "error": repr(e),
                            }
                        )

            with open(stream_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            summary["streams"].append(meta)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(success_flag, "w") as f:
        f.write("done\n")

    return summary


def build_report_dict(
    input_root: Path,
    output_root: Path,
    modalities: List[str],
    workers: int,
    stride: int,
    max_frames_per_stream: int,
    overwrite: bool,
    resume: bool,
    discovered: int,
    pending: int,
    already_done: int,
    ok: int,
    failed: int,
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "modalities": modalities,
        "workers": workers,
        "stride": stride,
        "max_frames_per_stream": max_frames_per_stream,
        "overwrite": overwrite,
        "resume": resume,
        "discovered_vrs": discovered,
        "pending_vrs": pending,
        "already_done_vrs": already_done,
        "ok": ok,
        "failed": failed,
        "failures": failures,
    }


def dump_report(report_path: Path, report: Dict[str, Any]) -> None:
    tmp = report_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    tmp.replace(report_path)


def main():
    parser = argparse.ArgumentParser(
        "Extract depth_images.vrs / synthetic_video.vrs using low-level pyvrs"
    )

    parser.add_argument(
        "--input_root",
        type=str,
        default=DEFAULT_INPUT_ROOT,
        help=f"ADT root directory. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Decoded output root; default: write under input_root",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        default=["depth", "synthetic"],
        choices=["depth", "synthetic"],
        help="Which VRS modalities to export",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker count, per VRS file",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth data record with images",
    )
    parser.add_argument(
        "--max_frames_per_stream",
        type=int,
        default=-1,
        help="Cap exported frames per stream; -1 means all",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume and force all matching VRS to run again",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else input_root
    ensure_dir(output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")

    report_path = output_root / "decode_report.json"
    partial_report_path = output_root / "decode_report_partial.json"

    print(f"[INFO] python={sys.executable}", flush=True)
    print(f"[INFO] input_root={input_root}", flush=True)
    print(f"[INFO] output_root={output_root}", flush=True)
    print(f"[INFO] modalities={args.modalities}", flush=True)
    print(f"[INFO] Start scanning VRS files...", flush=True)

    jobs = discover_jobs(input_root, args.modalities)
    print(f"[INFO] Found {len(jobs)} target VRS files under {input_root}", flush=True)

    if not jobs:
        print("[WARN] No matching VRS files found.", flush=True)
        return

    resume = not args.no_resume

    pending_jobs = []
    skipped_done = []

    for modality, vrs_path in jobs:
        if resume and (not args.overwrite) and is_vrs_done(vrs_path, input_root, output_root):
            skipped_done.append((modality, vrs_path))
        else:
            pending_jobs.append((modality, vrs_path))

    print(
        f"[INFO] resume={resume} | pending={len(pending_jobs)} | already_done={len(skipped_done)}",
        flush=True,
    )

    if not pending_jobs:
        report = build_report_dict(
            input_root=input_root,
            output_root=output_root,
            modalities=args.modalities,
            workers=args.workers,
            stride=args.stride,
            max_frames_per_stream=args.max_frames_per_stream,
            overwrite=args.overwrite,
            resume=resume,
            discovered=len(jobs),
            pending=0,
            already_done=len(skipped_done),
            ok=0,
            failed=0,
            failures=[],
        )
        dump_report(report_path, report)
        print("[DONE] Nothing to do. All matching VRS are already decoded.", flush=True)
        return

    job_args = [
        (
            modality,
            str(vrs_path),
            str(input_root),
            str(output_root),
            args.stride,
            args.overwrite,
            args.max_frames_per_stream,
        )
        for modality, vrs_path in pending_jobs
    ]

    ok = 0
    failed = 0
    failures: List[Dict[str, Any]] = []

    def current_report() -> Dict[str, Any]:
        return build_report_dict(
            input_root=input_root,
            output_root=output_root,
            modalities=args.modalities,
            workers=args.workers,
            stride=args.stride,
            max_frames_per_stream=args.max_frames_per_stream,
            overwrite=args.overwrite,
            resume=resume,
            discovered=len(jobs),
            pending=len(pending_jobs),
            already_done=len(skipped_done),
            ok=ok,
            failed=failed,
            failures=failures,
        )

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(decode_one_vrs, *ja) for ja in job_args]

            with tqdm(total=len(futures), desc="Decoding VRS via pyvrs", ncols=110) as pbar:
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                        ok += 1
                        saved = sum(s.get("saved", 0) for s in res.get("streams", []))
                        pbar.set_postfix_str(f"ok={ok} failed={failed} saved={saved}")
                    except Exception as e:
                        failed += 1
                        tb = traceback.format_exc()
                        failures.append(
                            {
                                "error": repr(e),
                                "traceback": tb,
                            }
                        )
                        pbar.set_postfix_str(f"ok={ok} failed={failed}")
                    finally:
                        dump_report(partial_report_path, current_report())
                        pbar.update(1)

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Writing partial report...", flush=True)
        dump_report(partial_report_path, current_report())
        raise

    final_report = current_report()
    dump_report(report_path, final_report)

    print("=" * 110, flush=True)
    print(f"[DONE] ok={ok}, failed={failed}", flush=True)
    print(f"[DONE] final report: {report_path}", flush=True)
    print(f"[DONE] partial report: {partial_report_path}", flush=True)


if __name__ == "__main__":
    main()