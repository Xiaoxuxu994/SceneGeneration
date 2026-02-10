<div align="left">

# Ascend-friendly Novel View Synthesis with Minimal 3D Inductive Bias 

## 1. Preparation

### Environment
```
conda create -n 3d-free-nvs python=3.11
conda activate 3d-free-nvs

pip install git+https://github.com/facebookresearch/map-anything.git@46af95e6c976bc29d5de37e433e4240217caee5b#egg=mapanything
pip install -r requirements.txt
```
As we used [xformers](https://github.com/facebookresearch/xformers) `memory_efficient_attention`, the GPU device compute capability needs > 8.0. Otherwise, it would pop up an error. Check your GPU compute capability in [CUDA GPUs Page](https://developer.nvidia.com/cuda-gpus#compute).

### Data
Download the Scannet++v2 dataset from [this link](https://scannetpp.mlsg.cit.tum.de/scannetpp/#:~:text=the%20data%20now.-,Download%20the%20data,-To%20download%20the) and put the data in `<YOUR_RAW_DATAPATH>`.
Run the following command to preprocess the data into our format.
```bash
python process_scannetpp.py --base_path <YOUR_RAW_DATAPATH>
```
The script will save the dataset metadata to `<YOUR_RAW_DATAPATH>/LVSM_metadata` and the absolute paths of train/val splits to `<YOUR_RAW_DATAPATH>/splits`.

**[OPTIONAL]** Download the RealEstate10K dataset from [this link](http://schadenfreude.csail.mit.edu:8000/), which is provided by [pixelSplat](https://github.com/dcharatan/pixelsplat), and `unzip` the zip file and put the data in `<YOUR_RAW_DATAPATH>`.
Run the following command to preprocess the data into our format.
```bash
python process_data.py --base_path <YOUR_RAW_DATAPATH> --output_dir <YOUR_PROCESSED_DATAPATH> --mode ['train' or 'test']
```

### Checkpoints
As discussed in the report, we provide the following checkpoints:
- MapAnything-only Encoder trained on 8 input views and 6 target views (`mapa_encoder_ftDecTok_i8/ckpt_0000000000010000.pt`)
- Dual Encoder trained on 8 input views and 6 target views (`dual_encoder_ft_i8/ckpt_0000000000010000.pt`)
- Dual Encoder trained on 32 input views and 6 target views (`dual_encoder_ftDecTok_i32/ckpt_0000000000010000.pt`)

To train/test a model using the provided checkpoints, unzip `checkpoints.zip` to `<YOUR_CKPT_PATH>` so that the final structure is: `<YOUR_CKPT_PATH>/<EXP_NAME>/ckpt_0000000000010000.pt`.

## 2. Training

### Configs

Each of the three provided experiments has an associated config file:
- MapAnything-only Encoder trained on 8 input views and 6 target views (`confgis/mapa_encoder_ftDecTok_i8.yaml`)
- Dual Encoder trained on 8 input views and 6 target views (`confgis/dual_encoder_ft_i8.yaml`)
- Dual Encoder trained on 32 input views and 6 target views (`configs/dual_encoder_ftDecTok_i32.yaml`)

To launch distributed training across `NUM_GPUS` GPUs and `NUM_NODES`, run:
```bash
torchrun --nproc_per_node NUM_GPUS --nnodes NUM_NODES \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train.py --config configs/<EXP_NAME>.yaml \
    training.checkpoint_dir=<YOUR_CKPT_PATH>/<EXP_NAME> \
    training.dataset_path=<YOUR_RAW_DATAPATH>/splits/hq_lvsm_train.txt \
    training.mlfow_tracking_uri=<YOUR_MLFLOW_DIR>
```
Training progress will be logged through MLflow, you can visualize it with:
```bash
mlflow ui --backend-store-uri <YOUR_MLFLOW_DIR>
```

## 3. Inference

Inference on our pre-defined splits by difficulty levels is performed through the bash script `run_inference.sh`. Set your base paths at the top of the file:

```bash
# Configuration
CKPT_ROOT="<YOUR_CKPT_PATH>"
DATASET_PATH="<YOUR_RAW_DATAPATH>/splits/hq_lvsm_val.txt"
OUT_ROOT="<YOUR_OUT_ROOT>"
RENDER_VIDEO=true
```

You can set `RENDER_VIDEO=false` to avoid rendering a video for each scene and get evaluation metrics more quickly.

```bash
bash run_inference.sh
```

All results will be saved to `<YOUR_OUT_ROOT>` with the following structure:
```
<YOUR_OUT_ROOT>
|--- EXP_NAME
|---|--- i${V_INPUT}_t${V_TARGET}_${DIFFICULTY}
|---|---|--- average_metrics.txt
|---|---|--- summary.csv
|---|---|--- <SCENE_ID>
|---|---|---|--- metrics.json
|---|---|---|--- rendered_video.mp4
```