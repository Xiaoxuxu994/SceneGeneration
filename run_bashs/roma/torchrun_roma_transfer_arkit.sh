export DEVICE_TYPE=ascend
export TOKENIZERS_PARALLELISM=false

# 配置CANN相关环境变量
CANN_INSTALL_PATH_CONF=/etc/Ascend/ascend_cann_install.info
sudo cat /etc/Ascend/ascend_cann_install.info
DEFAULT_CANN_INSTALL_PATH=/usr/local/Ascend/
CANN_INSTALL_PATH=/usr/local/Ascend/

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ];then
  cat ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
  source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
  cat ${CANN_INSTALL_PATH}/nnae/set_env.sh
  source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi

# 导入依赖库
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openblas/lib:/usr/local/lib/:/usr/lib64/:/usr/lib/:/usr/lib/aarch64_64-linux-gnu

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
# DEEPSPEED_CONFIG_PATH=${MA_JOB_DIR}/CodeClean/accelerate_config.yaml

# echo "num_machines: ${NNODES}" >> ${DEEPSPEED_CONFIG_PATH}
# echo "num_processes: ${NUM_PROCESSES}" >> ${DEEPSPEED_CONFIG_PATH}

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

export PATH=$CONDA_PREFIX/bin/:$PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/op_api/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch_npu/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
# export ASCEND_CUSTOM_OPP_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/meta_gauss_render/packages/vendors/customize/
# export ASCEND_LAUNCH_BLOCKING=1

mkdir -p /home/ma-user/.cache /home/ma-user/.cache/torch /home/ma-user/.cache/torch/hub /home/ma-user/.cache/torch/hub/checkpoints
cp ckpts/vgg16-397923af.pth /home/ma-user/.cache/torch/hub/checkpoints/

torchrun \
    --nnodes $NNODES \
    --nproc-per-node $NGPUS_PER_NODE \
    --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR \
    trainer_torchrun.py \
    --config_path configs/Refiner/exp_11_21_transfer_arkit_lr$1.json \
    --output_dir /home/ma-user/modelarts/outputs/train_url_0/ \
    > >(awk '{sub(/npu gs_sort kernel donot support, using cpu radix sort instead!$/, ""); if (length) print }') \
    2> >(awk '{sub(/mbind: Operation not permitted$/, ""); if (length) print >"/dev/stderr"}')