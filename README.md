# 245 训练
```sh
docker exec -it yyx-roma /bin/bash
cd /mnt/nvmeX/YOUR_DIR
git clone https://szv-y.codehub.huawei.com/LargeSpatialModel/SceneGeneration.git
cd SceneGeneration
git checkout yyx/gsrefiner_npu
bash train_gsdecoder.sh
```
在 `train_gsdecoder.sh` 中：
- 使用 `--select_scene` 选择过拟合场景，用 `debug.json` 避免 EFS IO
- 使用 `--opt PARENT.CHILD.LEAF XXX` 覆盖 json 参数
# 云道训练
- 从这里创建新训练：https://csb.roma.huawei.com/csb/#/hec/eiwizard/training/com.huawei.meta.intelligence/trainingjobv2?thirdType=detail&id=6725117c-6a52-484c-80bb-6b40889be65d
- 自定义镜像：yyx-3dgs 0.0.3
    - 0.0.3：1015 版 3DGS（NUMA mbind 问题云道无法解决，暂不影响运行）
    - 0.0.5：1025 版 3DGS（云道上 Vector Core Runtime Error，无法训练）
- 运行用户 ID：0
    - 以 root 运行避免权限问题
```sh
cd /home/ma-user/modelarts/user-job-dir/SceneGeneration; source /home/ma-user/.bashrc; bash train_gsdecoder_roma.sh
```