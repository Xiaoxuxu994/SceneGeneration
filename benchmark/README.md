Reconstruction Benchmark

# Basic Logistics

The basic benchmark logistics are:

- [ ] Sample the evaluation sequence according to evaluation protocals
- [x] Normalize the GT poses and pointmaps to the first-frame's coordinates, and save them as .npy files
- [x] Predict poses and pointmaps using 3R models, e.g. VGGT, and save the outputs as .npy files
- [x] Load the predicted and GT poses and pointmaps, then call evaluation functions
- [ ] repeat for all sequences and aggregate scores

## Usage

```
# There are a example sequence from 7scenes under ./samples/7scenes/
# Let's start from there

# inference
cd models/vggt_official
python benchmark/infer_7scenes.py
# this will output predicted and GT poses and pointmaps and save them to `outputs` (you're now in models/vggt_official)

# evaluate
cd ../.. # back to benchmark root
python eval_7scenes.py --input_dir models/vggt_official/outputs/7scenes/sample_seq/
```

# Data Description

## 7-Scenes

- Data structure:

```text
<root>
    ├── 7scenes
        ├── data
            ├── chess
                ├── seq-01
                    ├── frame-000000.color.png      # RGB, 24-bit
                    ├── frame-000000.depth.png      # don't use this
                    ├── frame-000000.depth.proj.png # depth in millimeters, 16-bit, invalid depth is set to >60000
                    ├── frame-000000.pose.txt       # camera-to-world, 4×4 matrix
```

# Update on 20251231
1. move infer scripts outside model folder, insert path in code
2. put all infer & eval script in './scripts'