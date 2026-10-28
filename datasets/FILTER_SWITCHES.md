# 数据过滤开关（xiaoxu/Pi3_training_v2）

所有开关默认关闭，不打开时训练数据与旧版完全一致。一键打开的配置：`configs/data/datamix_all_v9-5_fix.yaml`（数据集、权重、路径同 v9-5）。

## 公共开关（任意数据集的配置块里都能写）

| 开关 | 默认 | 作用 | 依据 |
|---|---|---|---|
| `pct_valid_only` | false | 98 分位只在 depth > 0 的有效像素上算 | 旧实现把无效像素的 0 也算进分位数，物体类被误删 8%–29% |
| `pct_disable` | false | 跳过分位截断，只保留 `max_depth` / `z_far` | 质检规则评估：除 DL3DV 外删掉的像素都不比保留的脏（Lift < 1.5） |
| `use_dataset_masks` | false | loader 返回的 `extra_valid` 并入 `valid_mask` | 自带 mask / confidence 原先读了没生效 |
| `min_baseline_ratio` | 0 | 所选帧 基线/深度 低于该值时，对同一序列重新选帧 | Matterport3D、MVSSynth、Replica、Spring 等视差不足（建议 0.05 ≈ 3°） |
| `baseline_retries` | 3 | 重新选帧的次数上限，用完仍不满足就照常使用 | |
| `seq_blacklist` | null | txt（每行一个序列名）或列表；与序列 key、`scene_to_path` 或其末级目录名匹配即剔除 | 配合质检的序列级打分 |

训练配置（`train:`）里另有 `worker_reseed: true`：每个 (rank, worker) 用不同种子重置 `np.random` 和各数据集的 `_rng`。
旧版所有 worker、所有卡用的是同一条随机数流，`aug_focal` 等增强在各 worker 间重复。

另外：读取失败每累计 200 次打印一次 `load failures so far: 失败数/尝试数`（旧版失败会被随机样本静默替换）。

## 各数据集的开关与修复

| 数据集 | 开关 | 作用 |
|---|---|---|
| DL3DV | `depth_interp: nearest` | 深度缩放到 RGB 尺寸改用最近邻（旧版 Lanczos 会在深度边缘造出飞点） |
| DL3DV | `pct_twice: false` | 98 分位只在裁剪后截一次（旧版前后各一次） |
| DL3DV | `rgb_root` | 原先写死的 RGB 目录改成参数，默认值不变 |
| DL3DV | `extra_valid` | 返回 `depth_mask`（conf > 0.25），配合 `use_dataset_masks` |
| CO3DV2 | `extra_valid` | 自带 `depth_masks` 随深度一起裁剪缩放后返回，配合 `use_dataset_masks` 。**已在 `_fix` / `_fix_obs` / `sub00` 配置里关闭**：并入后几乎每个 8 帧样本都有一帧有效像素为 0，取样本 10 次重试全失败（`tools/profile_data.py` 实测单样本 841s，还会让训练进程抛异常退出）|
| ARKitScenes | `conf_min: 2` | 读自带的 `confidence/`（0/1/2），只保留 ≥ conf_min 的像素，配合 `use_dataset_masks` |
| Habitat | `pass_z_far: true` | 配置里的 `z_far` 真正传给 BaseDataset（旧版被构造函数截下，不起作用） |

不加开关、直接修复的：
- ScanNet++ `searchsorted` 补上 `sorter` 映射：图像列表有序时结果不变（当前数据就是有序的），无序时旧写法会让位姿与内参错配。
- DL3DV 置信度图与 RGB 尺寸不一致时先对齐到 RGB 尺寸，否则按 RGB 坐标裁剪会错位（只影响返回的 `depth_mask`，训练默认不用它）。

## 对照实验建议

同一 checkpoint、同样步数，每次只开一类开关：
1. `pct_valid_only` / `pct_disable`（问题 1）
2. `use_dataset_masks` + DL3DV `depth_interp: nearest`（问题 3）
3. `min_baseline_ratio: 0.05`（问题 4）
4. `seq_blacklist`（问题 2，等全量序列打分出结果）

## 过滤前后对比（数据侧）

`tools/quality_eval.py dataset` 会读取同样的开关（分位、extra_valid 合并、视差重选帧、黑名单），所以直接换配置跑两遍即可：

```bash
CONFIG=configs/data/datamix_all_v9-5.yaml     OUT=quality_before bash tools/run_quality_eval.sh full --bg
CONFIG=configs/data/datamix_all_v9-5_fix.yaml OUT=quality_after  bash tools/run_quality_eval.sh full --bg
```

每个数据集的 `<数据集>.json` 里有 `switches` 字段，记录这次实际生效的开关。
注意：违例过滤还没有接入训练，过滤后的违例率不会因为它下降；这一项要等离线违例掩码做好再对比。

测试：`python tests/test_dataset_filters.py`（不需要真实数据）。
