# GT 体检：计算流程

脚本：`tools/gt_audit.py`；一键入口：`tools/run_gt_audit.sh`；默认配置：`configs/data/datamix_all_v9-5.yaml`。
全程只读，不改任何 loader。下面按执行顺序列出每一步对应的函数和公式。

## 一键运行

```bash
bash tools/run_gt_audit.sh stats          # 只统计 26 个数据集的规模（分钟级）
bash tools/run_gt_audit.sh full --bg      # 规模 + 几何体检，6 个数据集并行，日志 gt_audit_out.log
python tools/gt_audit.py --summarize --out gt_audit_out   # 随时汇总已跑完的部分
```

## 0. 每个数据集的处理流程（`run_one`）

```
run_one(name)
 ├─ data_root = cfg.train_dataset[name].data_root
 │    不存在 → status=PATH_NOT_FOUND，跳过（不构建 dataset，避免写出空缓存）
 ├─ build_dataset()          按 config 实例化 loader；关闭 aug_crop / aug_focal / shuffle，resolution=[512,336]
 │    抛异常 → status=BUILD_ERROR
 ├─ dataset_stats()          规模统计（第 1 节）
 │    --stats-only 到此结束
 └─ audit_dataset()          抽 N 个样本做几何体检（第 2–6 节）
      全部样本失败 → ALL_SAMPLES_FAILED；部分失败 → PARTIAL_FAIL(k/N)
```

注意：data_root 存在但 loader 没有缓存时，构建会扫描目录并写缓存（loader 自身行为，缓存路径写死在各 loader 里）。

## 1. 规模统计（`dataset_stats`）

都是 `mode=train`、按 config 的 `split_ratio` 切分之后，loader 实际会采样的部分。

| 字段 | 计算 |
|---|---|
| `n_sequences` | `len(ds.sequences)`，即 `__len__`，采样单位（一条轨迹 / 一个物体 / 一个注视点） |
| `n_scenes` | `len({ds.scene_to_path[s] for s in ds.sequences})`；loader 没有 `scene_to_path` 时等于 `n_sequences` |
| `n_images` | `Σ_s len(ds.scene_meta[s][key])`，`key` 取 `frame_ids / frames_id / view_ids / frames_name` 中第一个存在的 |
| `img/seq` | 每条序列帧数的中位数（另存 min / max） |
| `weight` | 该数据集在 config `weights` 中的占比 |

## 2. 取样本（`audit_dataset` → `fetch_views` → `unpack`）

1. 样本下标 `idx = linspace(0, len(ds)-1, N)`，在全集上均匀抽取，不随机。
2. 直接调用 `ds._get_views(idx, [W,H], rng)`，`rng = default_rng(seed·100003 + idx)`。这样做是为了绕过 `__getitem__` 的静默重试（`base_dataset.py:431`），失败会被如实记录。
3. 有效像素按 base 的同一口径计算（`make_valid`）：`valid = depth > 0 & isfinite(depth) & depth < z_far`。
4. 每帧拿到 `depth (H×W)`、`K`、`c2w`；如果 loader 返回了 `depth_mask`（如 DL3DV），也一并取出。

## 3. 多视角重投影（`reproject_stats` / `_accumulate_pair`）

对一个样本（F 帧）的全部有序帧对 (i, j)，i ≠ j，最多 `--max-pairs` 对，源帧像素按 `--stride` 下采样：

```
X_cam = K_i⁻¹ [u, v, 1]ᵀ · d_i(u, v)              # pixel_rays，z 分量为 1 的射线乘深度
X_w   = R_i X_cam + t_i                            # c2w_i
X_j   = R_jᵀ (X_w − t_j)                           # w2c_j
z     = X_j.z ;  (u', v') = round(K_j X_j / z)     # 只保留 z>0 且落在画面内的点
d_j   = depth_j(v', u')                            # 最近邻取值，且要求 valid_j(v', u')
rel   = (z − d_j) / d_j
```

| 指标 | 定义 | 含义 |
|---|---|---|
| `cons` | `mean(|rel| < τ)`，τ 默认 0.03 | 一致率。遮挡会拉低它（物体 360° 环拍尤其明显），只做横向比较 |
| `viol` | `mean(rel < −τ)` | **自由空间违例**：投过去的点落在 j 帧可见表面的前面，遮挡解释不了，是判断真错误的主指标 |
| `cons_center` / `cons_corner` | 按目标像素到主点的距离（以半对角线归一化）分桶，`r<0.3` / `r>0.7` | 仅供参考，会受场景内容影响 |

所有计数跨帧对、跨样本累加之后再相除（`merge` + `ratio`），所以点多的样本权重更大。

## 4. 假设检验

**ray2z（`ray_to_z`）**：把深度当作沿射线的距离，换成 z-depth：`z = d / ‖K⁻¹[u,v,1]‖`，然后用同样的流程再算一次 `cons`。
- `ray2z` 明显高于 `cons`：说明数据其实是射线距离，loader 少做了转换。
- `ray2z` 明显低于 `cons`：说明数据本来就是 z-depth。
- 物体居中、视场角小的数据集两者差别很小，这时该检验不具判别力。

**scale（`best_scale`）**：位姿不动，把所有帧的深度乘以 `s ∈ {10^-3, 10^-2.75, …, 10^3}`，取 `cons` **计数**最多的 s。用计数而不是比例，是因为 s 很小时点全部投出画面，比例没有意义。最后取各样本结果的中位数，偏离 1 说明深度与位姿的量纲不一致。

## 5. 掩码收益（`src_masks`）

在第 3 节的同一批投影里，按源像素属于哪个子集分别统计 `viol`：

| 子集 | 定义 | 对比 |
|---|---|---|
| `edge` | `depth_edge_np`：3×3 邻域内 `(max − min) / depth > 0.03`，与 `pi3.utils.geometry.depth_edge` 同口径 | `edgeV` vs `flatV`：边缘飞点有多严重 |
| `flat` | `valid & ~edge` | |
| `mask_out` / `mask_in` | `valid & ~depth_mask` / `valid & depth_mask`（只有 loader 返回 `depth_mask` 时才计算） | `mOutV` 明显大于 `mask_in`：loader 自带的 mask 该接进 valid_mask 但没接 |

## 6. 98 分位截断的影响（`--raw-compare`，`set_percentile_threshold`）

1. 把各 loader 模块里的 `threshold_depth_map` 临时换成只保留 `max_depth` 绝对截断的版本，用**同一个 seed** 重新取一次样本（位姿一致才比较，`np.allclose`）。
2. `removed = valid_无截断 & ~valid_有截断`。
3. `pctRm = Σ removed / Σ valid_无截断`：被截断删掉的有效像素占比。
4. `pctRmC`：被删像素自身的 `cons`。如果它接近整体 `cons`，说明截断删掉的是好数据；如果明显更低（如 DL3DV），说明删的是坏点。

## 7. 帧级量纲（`frame_*`，各帧中位数）

| 字段 | 计算 |
|---|---|
| `valid` | `mean(valid)` |
| `d_p50`（另存 p5 / p95） | 有效深度的分位数，用于量纲检查（米级是否合理） |
| `det` / `orth_err` | `det(R)`、`max|RRᵀ − I|`：位姿旋转是否为合法旋转；`det < 0` 表示镜像 |
| `baseline_over_depth` | 帧间相机中心距离的中位数 / 深度中位数：基线是否过小（纯旋转）或量纲异常 |

## 8. 告警规则（`flags`，经验阈值，只用于排优先级）

| 告警 | 条件 |
|---|---|
| 射线距离 | `ray2z − cons > 0.05` |
| 尺度不一致 | `scale ∉ [0.75, 1.34]` |
| 自由空间违例 | `viol > 10%` |
| 边缘飞点 | `edgeV > 3·max(flatV, 0.5%)` 且 `edgeV − flatV > 8%` |
| 自带 mask 未接入 | `mOutV > 2·max(mask_in_viol, 0.5%)` |
| 截断删好数据 | `pctRm > 1%` 且 `pctRmC ≥ 0.8·cons` |
| 旋转异常 | `orth_err > 1e-3` 或 `det < 0` |
| 有效像素过少 | `valid < 10%` |
| 加载失败 | 失败率 > 5%（全部失败时状态为 `ALL_SAMPLES_FAILED`） |

在解析渲染的合成场景上做过自测（注入射线距离、深度 ×0.01、位姿错配、边缘飞点）：干净场景不触发告警，每种注入的 bug 都只触发对应的告警。

## 9. 3D 可视化 dump（`tools/gt_vis_dump.py`）

用 MeshLab / CloudCompare / Open3D 打开，肉眼确认上面的指标。和体检用同一 idx、同一 seed，看到的是同一组帧。

```bash
bash tools/run_gt_audit.sh vis                                   # 全部数据集：体检最差 2 个 + 均匀 2 个样本
DATASETS="Matterport3D DL3DV Wildrgbd" N=1 bash tools/run_gt_audit.sh vis
VIS_OUT_ROOT=./gt_vis_local bash tools/run_gt_audit.sh vis       # 不写到数据目录旁
python tools/gt_vis_dump.py --datasets Hypersim --idx 0 61       # 指定样本
```

输出到 `<data_root>_vis/<idx>_<label>_f<帧数>_<W>x<H>/`（没有写权限时退回 `./gt_vis_out/<dataset>_vis/`）：

| 文件 | 内容 | 用来看什么 |
|---|---|---|
| `current.ply` | 现在训练用的点（含 98 分位截断、z_far），RGB + 相机 | 训练实际看到的几何 |
| `diag.ply` | 关掉分位截断后的全部有效点，按类别着色 + 相机 | 问题点在哪里 |
| `filtered.ply` | 去掉违例 / 边缘 / mask 外，保留被截断误删的点，RGB + 相机 | 建议过滤后的效果 |
| `cameras.ply` | 只有相机视锥（第一帧红 → 最后一帧蓝）+ 相机中心连线（黄） | 基线 / 视差：视锥挤在一起、黄线很短 = 视差不足 |
| `frames.png` | 每帧一行：RGB \| 当前深度 \| 诊断类别图 | 2D 上快速定位 |
| `stats.json` / `index.csv` | 基线/深度比、零基线帧对数、各类别像素占比 | 与体检数字对照 |

诊断颜色（互斥，优先级从高到低）：品红 = 自由空间违例（违例次数 ≥ 一致次数）；青 = depth_mask 外；红 = 边缘；黄 = 被 98 分位截掉；其余为原 RGB。
视锥大小默认 0.08 × 深度中位数（`--cam-scale` 可改），文件太大时用 `--point-stride 2`。
