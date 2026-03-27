# Reliev3R: Relieving Feed-forward Reconstruction from Multi-View Geometric Annotations

This is the codebase of Reliev3R.

## TODO List
- [x] Update dataloaders to fit multiple datasets. 
- [ ] Integrate benchmark code. 
- [ ] Support for camera pose input. 
- [ ] Support for camera intrinsic estimation. 
- [ ] Change the configuration system from the customized one to `hydra`. 

## Environment Setup

### Docker
* Local (7.242.99.245): Use docker IMAGE ID `87cecfa4a7d6`. To start a docker container from the docker image, run:
    ```
    docker run -it --name $CONTAINER_NAME --user root --entrypoint /bin/bash --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc --privileged=true -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/dcmi:/usr/local/dcmi 87cecfa4a7d6
    ```

* Online (ModelArts): `euleros_pytorch-2.6.0_cann8.2.rc1-gcc-9.5.0`.

### Dependencies
* In the default conda environment (PyTorch-2.1.0) of the specified docker above, run `pip install -r requirements.txt`.
* Download DINOv2 & DINOv3 checkpoints into directory `ckpt` (https://github.com/facebookresearch/dinov2), 
    ```
    |- ckpt
        |- dinov2-small
            |- dinov2_vits14_reg4_pretrain.pth
        |- dinov2-base
            |- dinov2_vitb14_reg4_pretrain.pth
        |- dinov2-large
            |- dinov2_vitl14_reg4_pretrain.pth
        |- dinov3_vits16
            |- dinov3_vits16_pretrain_lvd1689m-08c60483.pth
        |- dinov3_vits16plus
            |- dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
        |- dinov3_vitb16
            |- dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
        |- dinov3_vitl16
            |- dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
        |- dinov3_vith16plus
            |- dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
    ```

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
* `reliev3r_full_v8_c32_s1_dinov2`: Train Pi3 based on dinov2-vit, `pi3:model_size` in depth_aligner.yaml should be one of [`dinov2_large_0.95B` `dinov2_large_1.6B` `dinov2_giant_3.57B`]
* `reliev3r_full_v8_c32_s1_dinov3`: Train Pi3 based on dinov3-vit, `pi3:model_size` in depth_aligner.yaml should be one of [`dinov3_huge+_2.81B` `dinov3_huge+_3.2B` `dinov3_huge+_4.24B` `dinov3_huge+_3.51B` `dinov3_huge+_5.17B`]