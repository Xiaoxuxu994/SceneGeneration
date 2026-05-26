DATASETS=(
  dl3dv
)

SPLIT="test"
SEQ=""   # e.g. chess/seq-01 ; empty means all

MODE="stride"   # stride | random
STRIDE=4        # default stride
SEQ_LEN=8       # use N frames in sequence
MAX_SEQ_LEN=-1  # default max seq len, default -1 to use all
SEED=42             # used when MODE=random
NUM_SEQ=100

SKIP_POSE=0 # 1 if skip pose evaluation
SKIP_DEPTHMAP=1 # 1 if skip depthmap evaluation
SKIP_POINTMAP=1 # 1 if skip pointmap evaluation
SKIP_NORMAL=1 # 1 if skip point normal evaluation

# DINOv3
RGB_ORIGIN_W=398
RGB_ORIGIN_H=224
CROP_LEFT=87
CROP_RIGHT=87
CROP_TOP=0
CROP_BOTTOM=0
RGB_RESIZE_W=224
RGB_RESIZE_H=224
DEPTH_SCALE=1000.0
DEPTH_MIN=0.1
DEPTH_MAX=10.0
FX=1.0
FY=1.0
CX=1.0
CY=1.0