#### 训练代码 ####
1. 设置实验条件, 修改 train_code/exp0701_3D_geometry_vae_releaseV1_train.py 相应参数
    0) 精确复现开关
        accurate_reproduce = False/True
    1) kl 
        model = dict(
            enable_kl = False/True,
            kl_latent_dim = 64/128/256..,
        )
    2) multi-batch
        batch_size = 1,2,3,4,5,6,...
    3) fps 以traindatset举例,valdataset类似
        train_dataset = dict(
            type = 'DatasetForScanNetV2',
            data_root = '/mnt/data1/z00912387/scannetpp_pcd/train',
            data_list_name = 'datacache/scannetpp_trainval/nvs_sem_train.txt',
            query_sample_rate = 0.1,
            max_query_number = 100000,
            Gridsample_mode = 'train',
            grid_size = 0.05,
        )
        train_dataset = dict( # 离线fps, fps/(pcd x epoch)
            type = 'DatasetForScanNetV3',
            data_root = '/mnt/data1/z00912387/scannetpp_pcd/train',
            fps_root = '/mnt/data1/z00912387/scannetpp_pcd/train_fps_v3',
            data_list_name = 'datacache/scannetpp_trainval/nvs_sem_train.txt',
            query_sample_rate = 0.1,
            max_query_number = 100000,
            Gridsample_mode = 'train',
            grid_size = 0.05,
        )
        train_dataset = dict( # 离线fps, fps/pcd
            type = 'DatasetForScanNetV4',
            data_root = '/mnt/data1/z00912387/scannetpp_pcd/train',
            fps_root = '/mnt/data1/z00912387/scannetpp_pcd/train_fps_v4',
            data_list_name = 'datacache/scannetpp_trainval/nvs_sem_train.txt',
            query_sample_rate = 0.1,
            max_query_number = 100000,
            Gridsample_mode = 'train',
            grid_size = 0.05,
        )
        # 
        train_dataset = dict( # 异步fps, 训练脚本对应train_npu_col_asyncfps.py
            type = 'DatasetForScanNetV5',
            data_root = '/mnt/data1/z00912387/scannetpp_pcd/train',
            fps_root = './',
            data_list_name = 'datacache/scannetpp_trainval/nvs_sem_train.txt',
            query_sample_rate = 0.1,
            max_query_number = 100000,
            Gridsample_mode = 'train',
            grid_size = 0.05,
        )
2. 设定卡号,卡数,端口, 修改 train_npu_ddp.sh(train_npu_col.py/train_npu_col_asyncfps.py)
3. sh train_npu_ddp.sh

#### 推理 benchmark 代码 ####
1. 设置实验条件, 修改 train_code/exp0701_3D_geometry_vae_releaseV1_val.py 相应参数
    1) 指定模型权重ckpt所在路径和文件名
        root = './test/expr/000-VecSetAEV4/epoch499-global_step103999'
        ckpt_name = 'ckpt499.pth'
        save_as_pt = True
    2) 指定val输出路径
        outputs = "./test/expr/000-VecSetAEV4-val/"
        prefix = ""
2. 设定卡号, 由于是单卡无需设定卡数和端口, 修改 val_npu.sh
3. sh val_npu.sh


#### 离线fps生成代码 ####
1. DatasetForScanNetV3: 每个pointcloud每个epoch都有不同的采样,将在线采样流程全部转换为离线;整体下采样下标idx_select,在次基础上的fps采样下标fps_select
gen_fps_datasetv3.py -> fps_npu.sh 
由于每个epoch都会生成,硬盘占用高 500epochs:230G
2. DatasetForScanNetV4: 每个pointcloud固定的fps采样;在线流程仍然做下采样,但fps采样点固定,只生成fps采样点坐标fps_coord和网格化坐标点grid_coord
gen_fps_datasetv4.py -> fps_npu.sh
所有epoch使用相同的fps采样点,硬盘占用低 0.46G