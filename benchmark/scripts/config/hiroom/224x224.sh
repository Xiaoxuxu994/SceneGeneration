DATASETS=(
  hiroom
)

SPLIT="test"
SEQ=""   # e.g. chess/seq-01 ; empty means all

MODE="stride"   # stride | random
STRIDE=1        # default stride
SEQ_LEN=8       # use N frames in sequence
MAX_SEQ_LEN=-1  # default max seq len, default -1 to use all
SEED=42             # used when MODE=random
NUM_SEQ=30

SKIP_POSE=0 # 1 if skip pose evaluation
SKIP_DEPTHMAP=0 # 1 if skip depthmap evaluation
SKIP_POINTMAP=0 # 1 if skip pointmap evaluation
SKIP_NORMAL=1 # 1 if skip point normal evaluation

# HiRoom Dataset
RGB_ORIGIN_W=1024
RGB_ORIGIN_H=1024
CROP_LEFT=0
CROP_RIGHT=0
CROP_TOP=0
CROP_BOTTOM=0
RGB_RESIZE_W=224
RGB_RESIZE_H=224
DEPTH_SCALE=655.35
DEPTH_MIN=0.1
DEPTH_MAX=10.0
FX=1
FY=1
CX=1
CY=1