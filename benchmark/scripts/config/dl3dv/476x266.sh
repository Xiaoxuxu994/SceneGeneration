DATASETS=(
  dl3dv
)

SPLIT="test"
SEQ=""   # e.g. chess/seq-01 ; empty means all

MODE="random"   # stride | random
STRIDE=5        # default stride
SEQ_LEN=8       # use N frames in sequence
MAX_SEQ_LEN=-1  # default max seq len, default -1 to use all
SEED=42             # used when MODE=random
NUM_SEQ=100

# DINOv2
RGB_ORIGIN_W=476
RGB_ORIGIN_H=266
CROP_LEFT=0
CROP_RIGHT=0
CROP_TOP=0
CROP_BOTTOM=0
RGB_RESIZE_W=476
RGB_RESIZE_H=266
DEPTH_SCALE=1000.0
DEPTH_MIN=0.1
DEPTH_MAX=10.0
FX=1.0
FY=1.0
CX=1.0
CY=1.0