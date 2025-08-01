#!/usr/bin/env bash
# ASCEND_RT_VISIBLE_DEVICES=2,3 nohup torchrun --nproc_per_node=2 --master-port 29540 train_npu_col.py \
#     --config configs/exp0701_3D_geometry_vae_releaseV1_train.py > train.log 2>&1 &
# ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --nproc_per_node=4 --master-port 29541 train_npu_new_metric.py \
#     --config configs/exp0612_3D_geometry_vae_releaseV1_new_metric.py > train_async.log 2>&1 &

# ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29541 train_npu_new_metric.py \
#     --config configs/exp0612_3D_geometry_vae_releaseV1_new_metric.py
# ASCEND_LAUNCH_BLOCKING=1 ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29503 train_npu_col.py \
#     --config configs/exp0417_3D_geometry_vae_releaseV1_ddp.py

# ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29503 train_npu_col_ctf_new_metric.py \
#     --config configs/train/exp0620_3D_geometry_vae_normv6_ctf_4level_debug.py
# Single Node
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29503 train_npu_col_ctf_new_metric_KL.py --config configs/train/exp0721_3DGeoVAE_normv14_CTF_lr1e-4.py
# Multi Node
echo "-> aigc 3DGeoVAE on D910B"

sleep 2s
echo "-> sleep 2 minutes"
export DEVICE_TYPE=ascend
# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# 配置CANN相关环境变量
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'
sudo cat /etc/Ascend/ascend_cann_install.info
DEFAULT_CANN_INSTALL_PATH="/usr/local/Ascend/"
CANN_INSTALL_PATH="/usr/local/Ascend/"

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ];then
  cat ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
  source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
  cat ${CANN_INSTALL_PATH}/nnae/set_env.sh
  source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi

# 导入依赖库
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/openblas/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib64/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/aarch64_64-linux-gnu

# 配置自定义环境变量
export HCCL_WHITELIST_DISABLE=1

# log
export ASCEND_SLOG_PRINT_TO_STDOUT=0   # 日志打屏, 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       # 日志级别常用 1 INFO级别; 3 ERROR级别
export ASCEND_GLOBAL_EVENT_ENABLE=0    # 默认不使能event日志信息
export ASCEND_LAUNCH_BLOCKING=0 # 控制算子执行时是否启动同步模式, 1 同步; 0 异步

# 系统默认环境变量，不建议修改
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
NGPUS_PER_NODE="$MA_NUM_GPUS"
NUM_PROCESSES=$(($NGPUS_PER_NODE * $NNODES))

echo "------> system config <------"
echo "VC_WORKER_HOSTS: ${VC_WORKER_HOSTS}"
echo "MASTER_HOST: ${MASTER_HOST}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NGPUS_PER_NODE: ${NGPUS_PER_NODE}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "${MA_JOB_DIR}"
echo "------>  <------"
# https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/modeldevpt/ptmigr/ptmigr_0022.html
export HCCL_WHITELIST_DISABLE=1

if [[ $NODE_RANK == 0 ]]; then
    EXT_ARGS="--rdzv_conf=is_host=1"
else
    EXT_ARGS=""
fi

# set npu plog env, https://3ms.huawei.com/hi/group/3225441/wiki_6402466.html
ma_vj_name=`echo ${MA_VJ_NAME} | sed 's:ma-job:modelarts-job:g'`
task_name="worker-${VC_TASK_INDEX}"
task_plog_path=${MA_LOG_DIR}/${ma_vj_name}/${task_name}

mkdir -p ${task_plog_path}
export ASCEND_PROCESS_LOG_PATH=${task_plog_path}

echo "plog path: ${ASCEND_PROCESS_LOG_PATH}"

df -h
npu-smi info

export HCCL_CONNECT_TIMEOUT=7200

export OMP_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4

# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node=4  --nnodes=1  --master-port 29503 --master-addr 172.16.31.243 --node_rank=0 train_npu_col_ctf_new_metric_KL.py --config configs/train/exp0721_3DGeoVAE_normv14_CTF_lr1e-4.py
cd /home/ma-user/modelarts/inputs/data-path_0/
tar -xvzf scannetpp_norm_v14.tar.gz
cd $MA_JOB_DIR/train_code
torchrun --nproc_per_node=$NGPUS_PER_NODE  --nnodes=$NNODES  --master-port 29503 --master-addr $MASTER_ADDR --node_rank=$NODE_RANK train_npu_col_ctf_new_metric_KL.py --config configs/train/exp0721_3DGeoVAE_normv14_CTF_lr1e-4.py