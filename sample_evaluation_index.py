import json
import random
from tqdm import tqdm


random.seed(42)

# Parameters
V_INPUT = 16
V_TARGET = 3
txt_path = "/storage/datasets/nargenziano/scannetpp/splits/hq_lvsm_val.txt"
output_json = f"data/scannetpp/evaluation_index_scannetpp_i{V_INPUT}_t{V_TARGET}.json"

result = {}

with open(txt_path, "r") as f:
    metadata_paths = [line.strip() for line in f if line.strip()]

for metadata_path in tqdm(metadata_paths):
    with open(metadata_path, "r") as jf:
        data = json.load(jf)
    
    scene_name = data["scene_name"]
    num_frames = len(data["frames"])
    
    if num_frames < V_INPUT + V_TARGET:
        # if too few frames, skip this scene or adjust sampling
        print(f'Skipping scene {scene_name} with {num_frames} frames < {V_INPUT + V_TARGET} selected views')
        continue
    
    # Sample unique frame indices
    all_indices = list(range(num_frames))
    sampled = random.sample(all_indices, V_INPUT + V_TARGET)
    
    # Randomly split into context and target
    context = sorted(sampled[:V_INPUT])
    target = sorted(sampled[V_INPUT:])
    
    result[scene_name] = {
        "context": context,
        "target": target
    }

# Save to JSON
with open(output_json, "w") as out:
    json.dump(result, out, indent=4)

print(f"Saved {len(result)} scenes to {output_json}")
