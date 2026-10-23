# ScanNet++ in the SceneGeneration (reliev3r) layout, held-out scenes of the training split (last 10%).
# Same protocol as ../scannetpp/518x350.sh (stride 25, 8 views); NUM_SEQ=0 -> every held-out scene.
# Data root: DATA_ROOT (environment / caller), else ./data/scannetpp_r3 (run_infer.py default).
# Resolution for patch-14 models (official Pi3, VGGT).
DATASETS=(
  scannetpp_r3
)

SPLIT="test"
SEQ=""   # e.g. 09c1414f1b ; empty means all
DATA_ROOT="${DATA_ROOT:-}"   # e.g. /efs-m3ga/z00533370/reliev3r_data/scannetpp

MODE="stride"   # stride | random
STRIDE=25       # default stride
SEQ_LEN=8       # use N frames in sequence
MAX_SEQ_LEN=-1  # default max seq len, default -1 to use all
SEED=42             # used when MODE=random
NUM_SEQ=0       # 0 -> all held-out scenes

SKIP_POSE=0 # 1 if skip pose evaluation
SKIP_DEPTHMAP=0 # 1 if skip depthmap evaluation
SKIP_POINTMAP=0 # 1 if skip pointmap evaluation
SKIP_NORMAL=1 # 1 if skip point normal evaluation

RGB_ORIGIN_W=518
RGB_ORIGIN_H=350
CROP_LEFT=0
CROP_RIGHT=0
CROP_TOP=0
CROP_BOTTOM=0
RGB_RESIZE_W=518
RGB_RESIZE_H=350
DEPTH_SCALE=1000.0
DEPTH_MIN=0.1
DEPTH_MAX=10.0
FX=1.0
FY=1.0
CX=1.0
CY=1.0
