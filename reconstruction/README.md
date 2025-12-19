# Reliev3R: Relieving Feed-forward Reconstruction from Multi-View Geometric Annotations

This is the codebase of Reliev3R.

## TODO List
- [ ] Update dataloaders to fit multiple datasets. 
- [ ] Integrate benchmark code. 
- [ ] Support for camera pose input. 
- [ ] Support for camera intrinsic estimation. 
- [ ] Change the configuration system from the customized one to `hydra`. 

## Environment Setup

### Docker
* Local (7.242.99.245): Use docker IMAGE ID `87cecfa4a7d6`.
* Online (ModelArts): `euleros_pytorch-2.6.0_cann8.2.rc1-gcc-9.5.0`.

### Dependencies
* In the default conda environment (PyTorch-2.1.0) of the specified docker above, run `pip install -r requirements.txt`.

## Quick Start

### Train at local
Run the following command, 
```
bash train_local.sh
```

### Train online (ModelArts)
Run the following script, 
```
cd ${MA_JOB_DIR}/Reliev3R/

pip install -r requirements.txt

HOSTLIST=(${VC_WORKER_HOSTS//,/ })
export MASTER_ADDR=${HOSTLIST[0]}
export MASTER_PORT=12303

torchrun --nnodes=$MA_NUM_HOSTS --nproc_per_node=$MA_NUM_GPUS --node_rank=$VC_TASK_INDEX --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
train.py --data_url /efs_data/datasets/DL3DV --train_url /efs_data/exp/ssff3d/exp_name --config configs/ours_full/main.yaml
```

## Reliev3R Configuration
Currently we have 6 different experiment setups, each collected under a directory in `configs`. 
To start training with a specific configuration, pass the corresponded `main.yaml` to `train.py`. See `train_local.sh` for example. 
More specifically, we have the following experiment setups now:

* `ours_full`: Train Reliev3R on DL3DV-10K dataset. 
* `ours_overfit`: Train Reliev3R on a sample of DL3DV-10K dataset. Typically for debug. 
* `pi3_full`: Train Pi3 on DL3DV-10K dataset annotated with map-anything annotation pipeline (i.e., moge2 and mvsanywhere). 
* `pi3_overfit`: Train Pi3 on a sample of DL3DV-10K dataset annotated with map-anything annotation pipeline (i.e., moge2 and mvsanywhere). Typically for debug. 
* `pi3_cut3r_full`: Train Pi3 on DL3DV-10K dataset annotated with colmap (i.e., labels released by CUt3R). 
* `pi3_cut3r_overfit`: Train Pi3 on a sample of DL3DV-10K dataset annotated with colmap (i.e., labels released by CUt3R). Typically for debug. 