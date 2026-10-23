# ScanNet++ held-out benchmark for the exp1022 runs

The clips are exactly those of the training-time validation: the test split of `data=scannetpp_only`
(last 10% of the sorted scenes), 2000 fixed 8-view clips at 512x336 with seeded view sampling.
Metrics follow the eval3d benchmark (`code/bench/eval3d_bench.py` on the H200) plus the training-validation
pose metrics (RRA / RTA / AUC@5/10/30).

| script | does |
|---|---|
| `build_scannetpp_val8.py` | export the clips: `frames/<clip>/*.png`, `depth/<clip>/*.png` (mm), `gt/<clip>.npz` (c2w, K), `clips.json` |
| `infer.py` | one model -> `<bench>/<name>/<clip>/pred.npz` (c2w, K, depth); official Pi3 or a training checkpoint, any loop K |
| `evaluate.py` | all methods -> `<bench>/eval/{per_clip.csv, summary.json, summary.md}` |
| `run_eval.sh` | everything on the devserver (8 NPUs): build once, official Pi3 + exp1022_00..04, score |

```bash
bash run_sh_s3/run_scannetpp/mox_download.sh            # latest checkpoints -> /efs-m3ga/z00533370/exp/pi3sup/outputs
PI3_WEIGHTS=/path/to/Pi3/model.safetensors bash tools/eval_bench/run_eval.sh
KSWEEP=1 bash tools/eval_bench/run_eval.sh              # add loop models at smaller inference K
```

Notes
- Official Pi3 is built as `pi3_training.Pi3(model_size='dinov2_large_0.95B')` (the repo's `pi3/models/pi3.py`
  defaults to another spec); its confidence head is ignored. Input follows the official preprocessing
  (<= 255k pixels, multiples of 14), so 512x336 frames become 616x406. Pi3 was trained on ScanNet++: treat its
  numbers as a reference, not as a held-out result.
- Our runs see the frames as stored (512x336, patch 16), as during validation.
- Intrinsics of every model: principal point at the centre, focal fitted to the predicted local points.
- Point-cloud metrics align the prediction to GT with a Sim3 on paired points (pure geometry quality).
