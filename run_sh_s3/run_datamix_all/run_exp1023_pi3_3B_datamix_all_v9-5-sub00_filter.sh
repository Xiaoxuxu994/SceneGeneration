#!/usr/bin/env bash
# 实验 exp1023：Pi3 3B，数据 datamix_all_v9-5-sub00（23 个数据集，含 MegaDepth；打开过滤开关）。
# 训练超参与 run_exp0604 相同，公共逻辑见 _train_pi3_obs_common.sh。
# 注：MegaDepth 从 OBS 读太慢（单样本等待 100-230s），正式训练改用 run_exp1102_pi3_3B_datamix_all_v9-5-sub01_filter.sh。
#
# ModelArts 作业：
#   启动命令  cd <代码目录>; source /home/ma-user/.bashrc; bash run_sh_s3/run_datamix_all/run_exp1023_pi3_3B_datamix_all_v9-5-sub00_filter.sh
#   训练输出  obs://bucket-471/z00533370/exp/pi3sup/outputs/exp1023_pi3_3B_datamix_all_v9-5-sub00_filter
export NAME="exp1023_pi3_3B_datamix_all_v9-5-sub00_filter"
export DATA="datamix_all_v9-5-sub00"
exec bash "$(cd "$(dirname "$0")" && pwd)/_train_pi3_obs_common.sh" "$@"
