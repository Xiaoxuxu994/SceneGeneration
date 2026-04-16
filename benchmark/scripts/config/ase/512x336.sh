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

# DINOv3
RGB_ORIGIN_W=518
RGB_ORIGIN_H=518
CROP_LEFT=3
CROP_RIGHT=3
CROP_TOP=91
CROP_BOTTOM=91
RGB_RESIZE_W=512
RGB_RESIZE_H=336
DEPTH_SCALE=1000.0
DEPTH_MIN=0.1
DEPTH_MAX=10.0
FX=1.0
FY=1.0
CX=1.0
CY=1.0