DATASETS=(
  nrgbd
)

SPLIT="test"
SEQ=""   # e.g. chess/seq-01 ; empty means all

MODE="stride"   # stride | random
STRIDE=100      # default stride
SEQ_LEN=8       # use N frames in sequence
MAX_SEQ_LEN=-1  # default max seq len, default -1 to use all
SEED=42             # used when MODE=random
NUM_SEQ=10

# DINOv3
RGB_ORIGIN_W=518
RGB_ORIGIN_H=392
CROP_LEFT=0
CROP_RIGHT=0
CROP_TOP=0
CROP_BOTTOM=0
RGB_RESIZE_W=512
RGB_RESIZE_H=384
DEPTH_SCALE=1000.0
DEPTH_MIN=0.1
DEPTH_MAX=10.0
FX=554.2562584220408
FY=554.2562584220408
CX=320.0
CY=240.0