# Reliev3R: Relieving Feed-forward Reconstruction from Multi-View Geometric Annotations

This is the codebase of Reliev3R.

## Environment Setup

### Docker
* Local (7.242.99.245): Use docker IMAGE ID `87cecfa4a7d6`. To start a docker container from the docker image, run:
    ```
    docker run -it --name $CONTAINER_NAME --user root --entrypoint /bin/bash --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc --privileged=true -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/dcmi:/usr/local/dcmi 87cecfa4a7d6
    ```

* Online (ModelArts): `euleros_pytorch-2.9.0_cann8.5_triton3.2.0`.

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
    > On ROMA, DINOv2 checkpoints will be automatically mounted from `checkpoints` on EFS to `ckpt` via symlink.

### ROMA
* Syncing up with the ROMA platform can be done by easily run `./roma/code_upload.sh` and `./roma/code_download.sh`. Before that, you should set up the account and bucket path accordingly in those two scripts.
* Copy `roma/code_download_template.sh` to `roma/code_download.sh`. Change the parameters according to your own info.
* Copy `roma/code_upload_template.sh` to `roma/code_upload.sh`. Change the parameters according to your own info.

## Quick Start

### Train at local
Run the following command, 
```
bash train_local.sh
```

### Train Online (ModelArts)
Run the following script, 
```
cd ${MA_JOB_DIR}/${your_code_folder_name}/

pip install -r requirements.txt

HOSTLIST=(${VC_WORKER_HOSTS//,/ })
export MASTER_ADDR=${HOSTLIST[0]}
export MASTER_PORT=12303

torchrun --nnodes=$MA_NUM_HOSTS --nproc_per_node=$MA_NUM_GPUS --node_rank=$VC_TASK_INDEX --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
train.py \
--train_url /efs_data/${your_output_path} \
--config configs/${config_name}/main.yaml
```

### Train Online with DeepSpeed (ModelArts)
Run the following script, 
```
source /usr/local/Ascend/cann-8.5.0/set_env.sh

ln -s /efs-m3ga /efs-mi-east4-3

cd ${MA_JOB_DIR}/${your_code_folder_name}/

pip install -r requirements.txt

HOSTLIST=(${VC_WORKER_HOSTS//,/ })
export MASTER_ADDR=${HOSTLIST[0]}
torchrun --nnodes=$MA_NUM_HOSTS --nproc_per_node=$MA_NUM_GPUS --node_rank=$VC_TASK_INDEX --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
train_ds.py \
--config configs/${config_name}/main.yaml \
--train_url /efs_data/${your_output_path} \
--deepspeed \
--deepspeed_config ./ds_config/ds_config_z2_ot.json
```

## Training Options **(Important)**
* Environment variable `pred_z_depth`: If you want to train Pi3, set `pred_z_depth` as **string** `'True'`, otherwise you can leave this empty. 
* Switch between Reliev3R and Pi3: Add `train_pi3=true` in `main.yaml` to train Pi3, otherwise the codebase runs Reliev3R.

## Reliev3R Configuration
Currently we have 6 different experiment setups, each collected under a directory in `configs`. 
To start training with a specific configuration, pass the corresponded `main.yaml` to `train.py`. See `train_local.sh` for example. 
More specifically, we have the following experiment setups now:

* `reliev3r_full_v8_c32_s1_dinov2`: Train Reliev3R based on dinov2-vit, `pi3:model_size` in depth_aligner.yaml should be one of [`dinov2_large_0.95B` `dinov2_large_1.6B` `dinov2_giant_3.57B`]
* `reliev3r_full_v8_c32_s1_dinov3`: Train Reliev3R based on dinov3-vit, `pi3:model_size` in depth_aligner.yaml should be one of [`dinov3_huge+_2.81B` `dinov3_huge+_3.00B` `dinov3_huge+_3.2B` `dinov3_huge+_4.24B` `dinov3_huge+_3.51B` `dinov3_huge+_5.17B`]
* `pi3_dinov3`: Train Pi3 based on dinov3-vit, `pi3:model_size` in depth_aligner.yaml should be one of [`dinov3_huge+_2.81B` `dinov3_huge+_3.00B` `dinov3_huge+_3.2B` `dinov3_huge+_4.24B` `dinov3_huge+_3.51B` `dinov3_huge+_5.17B`]
