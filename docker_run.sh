#!/bin/bash

# 原容器ID或名称
# SOURCE_CONTAINER="gaussian-npu-dev_xiaoxu"
NEW_CONTAINER="ljr_3dgvae"
IMAGE="cb2dce5b2be1"

# 重新创建容器
docker run -dit -u 0 --ipc=host --net=host --privileged --entrypoint="/bin/bash" --name $NEW_CONTAINER \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /home/ljr:/home/ljr \
    -v /efs:/efs \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /mnt/nvme1:/mnt/nvme1 \
    -v /mnt/nvme2:/mnt/nvme2 \
    -v /mnt/nvme3:/mnt/nvme3 \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /home/z00533370:/home/z00533370 \
    -p 50032:22 \
    -p 50038:8080/tcp \
    -p 127.0.0.1:9898:888 \
    -p 8888:8888 \
    $IMAGE \
    /bin/bash