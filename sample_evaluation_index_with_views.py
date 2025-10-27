import json
import os
from tqdm import tqdm

# ==== CONFIG ====
metadata_txt = "/storage/datasets/nargenziano/scannetpp/splits/hq_lvsm_val.txt"    # TXT with metadata JSON paths
views_json = "/storage/datasets/nargenziano/scannetpp/splits/input_views_hq.json"   # JSON with {scene_name: [filenames_without_ext]}
output_json = "data/scannetpp/evaluation_index_scannetpp_i32_t32.json"
# =================

# Load metadata paths
scene_to_metadata = {}
with open(metadata_txt, "r") as f:
    for line in f:
        path = line.strip()
        if not path:
            continue
        with open(path, "r") as jf:
            data = json.load(jf)
        scene_to_metadata[data["scene_name"]] = data

# Load views JSON
with open(views_json, "r") as f:
    views_per_scene = json.load(f)

result = {}

for scene_name, filenames in tqdm(views_per_scene.items()):
    if scene_name not in scene_to_metadata:
        continue  # skip if no metadata JSON for this scene

    frames = scene_to_metadata[scene_name]["frames"]
    # Build a map: filename without extension -> frame index
    filename_to_index = {
        os.path.splitext(os.path.basename(f["image_path"]))[0]: idx
        # os.path.splitext(os.path.basename(f["image_path"]))[0].split('_')[-1]: idx
        for idx, f in enumerate(frames)
    }

    # Get indices for the requested filenames
    indices = [filename_to_index[fname] for fname in filenames if fname in filename_to_index]
    if len(indices) != 32:
        print(scene_name, len(indices))

    result[scene_name] = {
        "context": indices,
        "target": indices
    }

# Save output
with open(output_json, "w") as out:
    json.dump(result, out, indent=4)

print(f"✅ Saved {len(result)} scenes to {output_json}")
