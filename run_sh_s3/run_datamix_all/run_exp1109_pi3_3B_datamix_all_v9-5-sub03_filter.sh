#!/usr/bin/env bash
# 实验 exp1109（sub03）：Pi3 3B，数据 datamix_all_v9-5-sub03（24 个数据集：sub02 加回 MegaDepth；MegaDepth、GTA-sfm 读预处理数据；打开过滤开关）。
# 起训前 MegaDepth / GTA-SfM 预处理必须全部完成（run_sh_s3/run_dataset_sh/run_preprocess_md_gta.sh）。
# 128 卡（16 节点 × 8）正式训练；训练超参与 run_exp0604 相同，公共逻辑见 _train_pi3_obs_common.sh。
#
# ModelArts 作业：
#   启动命令  cd <代码目录>; source /home/ma-user/.bashrc; bash run_sh_s3/run_datamix_all/run_exp1109_pi3_3B_datamix_all_v9-5-sub03_filter.sh
#   训练输出  obs://bucket-471/z00533370/exp/pi3sup/outputs/exp1109_pi3_3B_datamix_all_v9-5-sub03_filter
# 日志另存：/efs-mi-pay-gy1/z00533370/exp/logs/exp1109_pi3_3B_datamix_all_v9-5-sub03_filter_node<N>_<时间>.log
export NAME="exp1109_pi3_3B_datamix_all_v9-5-sub03_filter"
export DATA="datamix_all_v9-5-sub03"
exec bash "$(cd "$(dirname "$0")" && pwd)/_train_pi3_obs_common.sh" "$@"
