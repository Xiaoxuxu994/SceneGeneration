Reconstruction Benchmark

# Basic Logistics

The basic benchmark logistics are:

- [x] Sample the evaluation sequence according to evaluation protocals
- [x] Normalize the GT poses and pointmaps to the first-frame's coordinates, and save them as .npy files
- [x] Predict poses and pointmaps using 3R models, e.g. VGGT, and save the outputs as .npy files
- [x] Load the predicted and GT poses and pointmaps, then call evaluation functions
- [x] repeat for all sequences and aggregate scores

# Usage

We have a simple script for running evaluation with different setups, e.g. in `scripts/run_reliev3r_scannet.sh`. There are some important paths:
```shell
# TODO Change the following settings
MODEL="reliev3r" # ['vggt','pi3','reliev3r']
CKPT="experiments/Reliev3R_bs2v8_260120_warmup/ckpt/5000.ckpt"
CONFIG="experiments/Reliev3R_bs2v8_260120_warmup/config.json"

# prefix of the output folder
# make sure it's consistent with the ckpt path
PREFIX="Reliev3R_bs2v8_260120_warmup_ckpt5k" 
```

Simple usage by running this script:
```shell
./scripts/run_reliev3r_scannet.sh
```

## Detailed usage

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

# Update on 20260108

```
模型推理主脚本：run_infer.py
模型结果评测脚本：run_eval.py
模型结果可视化：run_vis.py
```
```
模型结果保存格式：
    <output_root>
        - 7scenes
            - test_stride{}_max{}
                - frame_ids
                - gt
                - gt_vis
                - vggt
                - vggt_vis
                - pi3
                - pi3_vis
                - reliev3r_{}
                - reliev3r_{}_vis
                - metrics_vggt.xlsx
                - metrics_pi3.xlsx
                - metrics_reliev3r_{}.xlsx
        - nrgbd
        - xxx
```
# Update on 20261024: ScanNet++ held-out split of SceneGeneration + SceneGeneration-trained Pi3 / loop models

Dataset `scannetpp_r3` (`datasets/scannetpp_r3.py`): ScanNet++ DSLR in the reliev3r layout read by the
SceneGeneration training dataloader (`<scene>/images/*.jpg`, `depth/*.png` in mm, `scene_metadata.npz`).
`test` = the held-out scenes of training: scenes sorted by name (or `sequence_list_v2.pkl` when present), kept with
images/ + depth/ and >= 24 frames, last 10% (`split_ratio 0.9`). Frame order = training order. Protocol configs in
`scripts/config/scannetpp_r3/` are those of `scannetpp` (stride 25, 8 views) with `NUM_SEQ=0` (all held-out scenes).
Data root: `./data/scannetpp_r3` or `DATA_ROOT=...` (e.g. `/efs-m3ga/z00533370/reliev3r_data/scannetpp`).

Model `pi3loop` (`run_infer.py`, code in `models/pi3loop` = `pi3/` of SceneGeneration branch
`xiaoxu/Pi3_training_loop_transformer`): `--model_size` spec of `pi3_training.Pi3` (`dinov3_small_w384`,
`dinov3_small_w384x12`, ...), `--loop_layers L --loop_k K` for `Pi3Loop` (0 = no loop), `--ckpt` a DeepSpeed
checkpoint dir (`checkpoint_N` / `best_model`) or a state dict. Outputs and metrics as for `pi3`.
In `scripts/lib/engine.sh` the variables `DATA_ROOT`, `LOOP_LAYERS`, `LOOP_K` are forwarded when set.

```shell
# inside the benchmark container, from benchmark/ (see ../docker/bench_one.sh for the exp1022_00..05 entries)
MODEL=pi3loop MODEL_SIZE=dinov3_small_w384 LOOP_LAYERS=6 LOOP_K=3 CKPT=<run>/ckpts/checkpoint_14 ...
```
