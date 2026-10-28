# 数据读取：本地 / OBS 与单独调试（xiaoxu/Pi3_training_v2）

## 1. 统一读写层 `datasets/base/fileio.py`

所有 loader 的文件读取（exists / isdir / listdir / glob / open / PIL / cv2 / numpy / h5py / pandas / imageio）都改为 `fio.*`：
- 本地路径：直接调用原函数，行为不变。
- `obs://` 路径：惰性导入 moxing，整读字节后在内存里解码；每 25 次读取 gc 一次（moxing 内存泄漏的规避办法）。
- `path_remap`：`{efs 前缀: obs 前缀}`，按最长前缀替换。缓存 pkl 里存的是建缓存时的本地绝对路径，用它就能改读 OBS，不必重建缓存。
  目录层级对不上时，键写成 `'re:<正则>'`、值写替换串（`\1` 引用分组），先于前缀规则按顺序尝试。例：Taskonomy 在 OBS 上是 `rgb/taskonomy/<场景>/`，efs 是 `rgb/<场景>/<场景>/`，见 `datamix_all_v9-5_obs.yaml`。

## 2. 新的构造参数（BaseDataset，任何数据集都能写）

| 参数 | 默认 | 说明 |
|---|---|---|
| `cache_dir` | 环境变量 `DATASET_CACHE_DIR`，否则旧的写死目录 | 索引缓存 pkl 的目录，可以是 obs:// |
| `cache_file` | null | 直接指定缓存 pkl 的完整路径（优先） |
| `path_remap` | null | `{src_prefix: dst_prefix}` |

缓存文件名仍是 `{label}_{md5(data_root)[:8]}_cache.pkl`。`data_root` 不变，已有缓存就能直接用。

## 3. 贵阳 OBS 配置

- 路径对照：`configs/data/obs_paths_guiyang.yaml`（来自 Reliev3R_dataset2.xlsx，含每个数据集的 status / note）。
- 缓存目录：`obs://bucket-471/z00533370/dataset/dataset_cache`，已写进每个数据集块的 `cache_dir`。
- 生成的配置：`configs/data/datamix_all_v9-5_obs.yaml`、`configs/data/datamix_all_v9-5_fix_obs.yaml`；
  ARKitScenes、ScanNet、DL3DV 暂不使用，已从 weights 去掉（23 个数据集，权重合计 74,200）。
- 训练：`data=datamix_all_v9-5_obs`。

## 3.1 数据读取与 GT 验证脚本

```bash
bash run_sh_s3/run_dataset_sh/run_vis_datamix_v9-5_obs.sh                 # 全部数据集：读取检查 + 点云可视化
STAGE=check bash run_sh_s3/run_dataset_sh/run_vis_datamix_v9-5_obs.sh     # 只做读取检查
DATASETS="Hypersim CO3DV2" N=3 OBS_OUT=obs://bucket-471/xxx/vis bash run_sh_s3/run_dataset_sh/run_vis_datamix_v9-5_obs.sh
```

输出 `check/check_summary.tsv`（每个数据集是否读通）和 `vis/<数据集>_vis/<样本>/`（current / diag / filtered / cameras.ply、frames.png）。

## 4. 每个数据集单独调试

每个 `datasets/*_dataset.py` 都可以直接运行（在仓库根目录）：

```bash
python -m datasets.hypersim_dataset --config configs/data/datamix_all_v9-5_obs.yaml --n 3
python -m datasets.dl3dv_dataset --config configs/data/datamix_all_v9-5_obs.yaml --idx 0 100 --vis output_debug/dl3dv
python -m datasets.scannet_dataset --cache_dir obs://bucket-471/xxx/dataset_cache --remap /efs-m3ga/z00533370/dataset/scannet_v2_decode=obs://bucket-471/.../scannet
```

它会：从配置里找到该类对应的块 → 构造数据集（打印长度、缓存目录、耗时）→ 对每个样本先直接调用 `_get_views`
（出错打印完整 traceback，不会被训练时的重试掩盖），再走一遍 `__getitem__` → 打印每帧的图像尺寸、深度范围、有效像素比例、
位姿行列式、焦距；`--vis` 时保存点云与图像。全部参数见 `datasets/base/debug.py`。

测试：`python tests/test_fileio.py`（假 moxing，不需要真实数据）。

## 样本内多帧并发读取

`DATA_VIEW_THREADS=N`（环境变量，N>1 生效；训练和 profile 启动脚本默认 8）：`BaseDataset.__getitem__` 里各 loader 对
`_load_one_view` 的逐帧调用改为提交到 N 个线程并发执行，按原顺序取回。读 OBS 主要耗在每次请求的延迟上（约 0.3s/次），
8 帧串行时一个样本 5–15s，并发后接近单帧耗时。每帧的 rng 由主线程从样本 rng 依次派生，同一 seed 结果可复现；
任一帧抛异常时照常进入重试。`debug.py` / `gt_audit.py` 直接调用 `_get_views`，不受影响。测试：`python tests/test_parallel_views.py`。
