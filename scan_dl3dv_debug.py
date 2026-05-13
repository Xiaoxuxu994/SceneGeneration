import os
import os.path as osp
import json
import csv
from collections import defaultdict, Counter
from tqdm import tqdm


# ======================
# Default config
# ======================
DATA_ROOT = "/efs-mi-east4-2/datasets/DL3DV_WAI/"
RGB_ROOT = "/efs-mi-east4-2/datasets/DL3DV"
OUT_DIR = "./log/dl3dv_debug"

MIN_NUM_IMAGES = 24
MAX_EXAMPLES = 5
LIMIT = -1   # -1 means scan all scenes


def safe_get(d, key, default=None):
    return d[key] if isinstance(d, dict) and key in d else default


def build_rgb_path(scene_name, frame_info):
    scene_seg = scene_name.split("-")
    if len(scene_seg) < 2:
        return None, f"bad_scene_name_split: {scene_name}"

    file_path = safe_get(frame_info, "file_path")
    if file_path is None:
        return None, "missing_file_path_in_frame"

    rgb_path = osp.join(
        RGB_ROOT,
        scene_seg[0],
        scene_seg[1],
        "images_8",
        osp.basename(file_path),
    ).replace(".jpg", ".png")

    return rgb_path, None


def check_transform_matrix(frame_info):
    mat = safe_get(frame_info, "transform_matrix")
    if mat is None:
        return False, "missing_transform_matrix"

    try:
        if len(mat) != 4:
            return False, f"bad_transform_matrix_rows={len(mat)}"
        for row in mat:
            if len(row) != 4:
                return False, "bad_transform_matrix_cols"
    except Exception:
        return False, "bad_transform_matrix_format"

    return True, None


def check_intrinsic(cam_info):
    required = ["fl_x", "fl_y", "cx", "cy"]
    missing = [k for k in required if k not in cam_info]
    if missing:
        return False, "missing_intrinsic_" + ",".join(missing)

    try:
        float(cam_info["fl_x"])
        float(cam_info["fl_y"])
        float(cam_info["cx"])
        float(cam_info["cy"])
    except Exception as e:
        return False, f"bad_intrinsic_value: {repr(e)}"

    return True, None


def add_example(result, key, value):
    if len(result["examples"][key]) < MAX_EXAMPLES:
        result["examples"][key].append(value)


def inspect_scene(scene_dir, scene_name):
    result = {
        "scene_name": scene_name,
        "scene_dir": scene_dir,
        "keep": False,
        "reason": "",
        "total_frames": 0,
        "valid_frames": 0,
        "missing_rgb": 0,
        "missing_depth": 0,
        "missing_depthconf": 0,
        "missing_pose": 0,
        "bad_frame_info": 0,
        "valid_ratio": 0.0,
        "examples": defaultdict(list),
    }

    if not osp.isdir(scene_dir):
        result["reason"] = "not_dir"
        result["examples"] = dict(result["examples"])
        return result

    meta_path = osp.join(scene_dir, "scene_meta.json")
    if not osp.exists(meta_path):
        result["reason"] = "missing_scene_meta_json"
        result["examples"] = dict(result["examples"])
        return result

    try:
        with open(meta_path, "r") as f:
            cam_info = json.load(f)
    except Exception as e:
        result["reason"] = f"bad_scene_meta_json: {repr(e)}"
        result["examples"] = dict(result["examples"])
        return result

    json_scene_name = safe_get(cam_info, "scene_name")
    if json_scene_name != scene_name:
        result["reason"] = f"scene_name_mismatch: json={json_scene_name}"
        result["examples"] = dict(result["examples"])
        return result

    ok_intr, intr_reason = check_intrinsic(cam_info)
    if not ok_intr:
        result["reason"] = intr_reason
        result["examples"] = dict(result["examples"])
        return result

    if "_applied_transform" not in cam_info:
        add_example(result, "missing_global_meta", "_applied_transform")

    if "_applied_transforms" not in cam_info:
        add_example(result, "missing_global_meta", "_applied_transforms")
    elif "opengl2opencv" not in cam_info["_applied_transforms"]:
        add_example(result, "missing_global_meta", "_applied_transforms.opengl2opencv")

    frames = safe_get(cam_info, "frames", [])
    if not isinstance(frames, list):
        result["reason"] = "frames_not_list"
        result["examples"] = dict(result["examples"])
        return result

    result["total_frames"] = len(frames)

    for i, frame_info in enumerate(frames):
        if not isinstance(frame_info, dict):
            result["bad_frame_info"] += 1
            add_example(result, "bad_frame_info", f"frame_idx={i}, not_dict")
            continue

        frame_valid = True

        rgb_path, rgb_err = build_rgb_path(scene_name, frame_info)
        if rgb_err is not None:
            result["missing_rgb"] += 1
            frame_valid = False
            add_example(result, "missing_rgb", f"frame_idx={i}, {rgb_err}")
        elif not osp.exists(rgb_path):
            result["missing_rgb"] += 1
            frame_valid = False
            add_example(result, "missing_rgb", rgb_path)

        depth_rel = safe_get(frame_info, "mvsanywhere_depth")
        if depth_rel is None:
            result["missing_depth"] += 1
            frame_valid = False
            add_example(result, "missing_depth", f"frame_idx={i}, missing mvsanywhere_depth key")
        else:
            depth_path = osp.join(scene_dir, depth_rel)
            if not osp.exists(depth_path):
                result["missing_depth"] += 1
                frame_valid = False
                add_example(result, "missing_depth", depth_path)

        conf_rel = safe_get(frame_info, "mvsanywhere_depth_confidence")
        if conf_rel is None:
            result["missing_depthconf"] += 1
            frame_valid = False
            add_example(result, "missing_depthconf", f"frame_idx={i}, missing mvsanywhere_depth_confidence key")
        else:
            conf_path = osp.join(scene_dir, conf_rel)
            if not osp.exists(conf_path):
                result["missing_depthconf"] += 1
                frame_valid = False
                add_example(result, "missing_depthconf", conf_path)

        ok_pose, pose_reason = check_transform_matrix(frame_info)
        if not ok_pose:
            result["missing_pose"] += 1
            add_example(result, "missing_pose", f"frame_idx={i}, {pose_reason}")
            # 注意：和你的原 Dataset 对齐，pose 不参与 scan 阶段过滤
            # 所以这里不设置 frame_valid = False

        if frame_valid:
            result["valid_frames"] += 1

    if result["total_frames"] > 0:
        result["valid_ratio"] = result["valid_frames"] / result["total_frames"]

    if result["valid_frames"] >= MIN_NUM_IMAGES:
        result["keep"] = True
        result["reason"] = "kept"
    else:
        result["keep"] = False
        result["reason"] = (
            f"too_few_valid_frames: "
            f"valid={result['valid_frames']}, "
            f"total={result['total_frames']}, "
            f"min_num_images={MIN_NUM_IMAGES}"
        )

    result["examples"] = dict(result["examples"])
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n========== RUN CONFIG ==========")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"RGB_ROOT: {RGB_ROOT}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"MIN_NUM_IMAGES: {MIN_NUM_IMAGES}")
    print(f"MAX_EXAMPLES: {MAX_EXAMPLES}")
    print(f"LIMIT: {LIMIT}")
    print("================================\n")

    scene_names = sorted(os.listdir(DATA_ROOT))
    if LIMIT > 0:
        scene_names = scene_names[:LIMIT]

    all_results = []
    scene_reason_counter = Counter()

    total_frames = 0
    total_valid_frames = 0
    total_missing_rgb = 0
    total_missing_depth = 0
    total_missing_conf = 0
    total_missing_pose = 0

    for scene_name in tqdm(scene_names, desc="Scanning DL3DV scenes"):
        scene_dir = osp.join(DATA_ROOT, scene_name)
        res = inspect_scene(scene_dir, scene_name)
        all_results.append(res)

        if res["keep"]:
            scene_reason_counter["kept"] += 1
        else:
            reason_key = res["reason"].split(":")[0]
            scene_reason_counter[reason_key] += 1

        total_frames += res["total_frames"]
        total_valid_frames += res["valid_frames"]
        total_missing_rgb += res["missing_rgb"]
        total_missing_depth += res["missing_depth"]
        total_missing_conf += res["missing_depthconf"]
        total_missing_pose += res["missing_pose"]

    kept = sum(1 for r in all_results if r["keep"])
    dropped = len(all_results) - kept

    summary = {
        "data_root": DATA_ROOT,
        "rgb_root": RGB_ROOT,
        "min_num_images": MIN_NUM_IMAGES,
        "total_scenes": len(all_results),
        "kept_scenes": kept,
        "dropped_scenes": dropped,
        "total_frames": total_frames,
        "total_valid_frames": total_valid_frames,
        "global_valid_ratio": total_valid_frames / total_frames if total_frames > 0 else 0.0,
        "total_missing_rgb": total_missing_rgb,
        "total_missing_depth": total_missing_depth,
        "total_missing_depthconf": total_missing_conf,
        "total_missing_pose": total_missing_pose,
        "scene_reason_counter": dict(scene_reason_counter),
    }

    json_path = osp.join(OUT_DIR, "dl3dv_scan_full.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "scenes": all_results,
            },
            f,
            indent=2,
        )

    csv_path = osp.join(OUT_DIR, "dl3dv_scan_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_name",
            "keep",
            "reason",
            "total_frames",
            "valid_frames",
            "valid_ratio",
            "missing_rgb",
            "missing_depth",
            "missing_depthconf",
            "missing_pose",
            "bad_frame_info",
        ])

        for r in all_results:
            writer.writerow([
                r["scene_name"],
                r["keep"],
                r["reason"],
                r["total_frames"],
                r["valid_frames"],
                f"{r['valid_ratio']:.6f}",
                r["missing_rgb"],
                r["missing_depth"],
                r["missing_depthconf"],
                r["missing_pose"],
                r["bad_frame_info"],
            ])

    dropped_csv_path = osp.join(OUT_DIR, "dl3dv_dropped_scenes.csv")
    with open(dropped_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_name",
            "reason",
            "total_frames",
            "valid_frames",
            "valid_ratio",
            "missing_rgb",
            "missing_depth",
            "missing_depthconf",
            "missing_pose",
        ])

        for r in all_results:
            if not r["keep"]:
                writer.writerow([
                    r["scene_name"],
                    r["reason"],
                    r["total_frames"],
                    r["valid_frames"],
                    f"{r['valid_ratio']:.6f}",
                    r["missing_rgb"],
                    r["missing_depth"],
                    r["missing_depthconf"],
                    r["missing_pose"],
                ])

    print("\n========== DL3DV DEBUG SUMMARY ==========")
    print(f"total scenes: {len(all_results)}")
    print(f"kept scenes: {kept}")
    print(f"dropped scenes: {dropped}")
    print(f"total frames: {total_frames}")
    print(f"total valid frames: {total_valid_frames}")
    print(f"global valid ratio: {summary['global_valid_ratio']:.4f}")
    print("")
    print(f"total missing rgb: {total_missing_rgb}")
    print(f"total missing depth: {total_missing_depth}")
    print(f"total missing depthconf: {total_missing_conf}")
    print(f"total missing pose: {total_missing_pose}")
    print("")
    print("scene-level reasons:")
    for k, v in scene_reason_counter.most_common():
        print(f"  {k}: {v}")

    print("")
    print(f"saved json: {json_path}")
    print(f"saved csv: {csv_path}")
    print(f"saved dropped csv: {dropped_csv_path}")
    print("========================================\n")

    print("Top 20 dropped scenes:")
    dropped_results = [r for r in all_results if not r["keep"]]
    dropped_results = sorted(
        dropped_results,
        key=lambda x: (x["valid_frames"], -x["total_frames"]),
    )

    for r in dropped_results[:20]:
        print(
            f"{r['scene_name']} | "
            f"{r['reason']} | "
            f"valid={r['valid_frames']}/{r['total_frames']} | "
            f"miss_rgb={r['missing_rgb']} "
            f"miss_depth={r['missing_depth']} "
            f"miss_conf={r['missing_depthconf']} "
            f"miss_pose={r['missing_pose']}"
        )

        for reason, examples in r.get("examples", {}).items():
            if len(examples) > 0:
                print(f"  [{reason}]")
                for ex in examples[:MAX_EXAMPLES]:
                    print(f"    {ex}")


if __name__ == "__main__":
    main()