DATASETS=(
  ase
)

SPLIT="test"
SEQ=""   # e.g. chess/seq-01 ; empty means all

MODE="random"   # stride | random
STRIDE=5        # default stride
SEQ_LEN=8       # use N frames in sequence
MAX_SEQ_LEN=-1  # default max seq len, default -1 to use all
SEED=42             # used when MODE=random
NUM_SEQ=10

SKIP_POSE=0 # 1 if skip pose evaluation
SKIP_DEPTHMAP=0 # 1 if skip depthmap evaluation
SKIP_POINTMAP=0 # 1 if skip pointmap evaluation
SKIP_NORMAL=1 # 1 if skip point normal evaluation

# DINOv2
RGB_ORIGIN_W=518
RGB_ORIGIN_H=518
CROP_LEFT=28
CROP_RIGHT=28
CROP_TOP=28
CROP_BOTTOM=28
RGB_RESIZE_W=518
RGB_RESIZE_H=518
DEPTH_SCALE=1000.0
DEPTH_MIN=0.1
DEPTH_MAX=10.0
FX=1.0
FY=1.0
CX=1.0
CY=1.0