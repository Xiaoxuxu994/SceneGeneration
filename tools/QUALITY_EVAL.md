# 标签质量评估：用法

脚本：`tools/quality_eval.py`（评估）、`tools/calibrate_thresholds.py`（阈值校准）；指标实现：`utils/quality_metrics.py`；
一键入口：`tools/run_quality_eval.sh`。指标定义和公式见报告"质量指标定义"一页。
M1–M7 只用标签自身的深度、内参、位姿，不需要真值；G1（AbsRel_kept、Sel、GTCov）和阈值校准需要真值。

讲解和日常查看用精简口径：**3 个核心指标**（违例率 = M1 FSV、可用率 = M3 VPR、视差 = M5 TRI，决定分档）
\+ **2 项检查**（尺度 = M4，只报 OK/FAIL；边界 = M6，边界违例率是内部的几倍，> 10 倍需处理）
\+ **1 个方法**（规则评估 = M7，删掉的像素比保留的脏几倍）。M2 MVC 只放完整表。终端默认打印精简表，`--full` 打印完整表。
全程只读，不改任何 loader；`dataset` 模式和 `gt_audit.py` 用同一套取样（同一 idx、同一 seed）。

## 一键运行

```bash
bash tools/run_quality_eval.sh test                    # 单元测试，确认环境（只需 numpy）
bash tools/run_quality_eval.sh smoke                   # Hypersim、DL3DV 各 3 个样本，确认数据路径
bash tools/run_quality_eval.sh full --bg               # 全部数据集，6 个并行，日志 quality_out.log
python tools/quality_eval.py summarize --out quality_out   # 随时合并已跑完的数据集

ROOT=/path/to/pseudo bash tools/run_quality_eval.sh pseudo                    # 伪标签
ROOT=/path/to/pseudo CONF_THR=0.5 SAVE_MASKS=1 bash tools/run_quality_eval.sh pseudo
ROOT=/path/to/pseudo_with_gt bash tools/run_quality_eval.sh calib            # 校准 → calib.json
THRESHOLDS=calib.json ROOT=/path/to/pseudo bash tools/run_quality_eval.sh pseudo
```

## 伪标签目录格式

```
<ROOT>/<序列名>/*.npz      每帧一个 npz，按文件名排序
  必需：depth [H,W]（z-depth）、K [3,3]、c2w [4,4]（相机到世界）
  可选：valid / sky / dyn [H,W] bool、conf [H,W] float、gt_depth [H,W]（有真值时算 G1，校准必需）
```

每条序列按 `FRAMES` 帧一个窗口切分（默认不重叠），窗口内算 M1–M6，再按计数合并成序列级指标。
视频帧间隔很小时三角化角不够（`tri_angle_median (usable/reject)`），用 `FRAME_STEP=2` 或更大先抽帧。

## 输出

| 模式 | 文件 | 内容 |
|---|---|---|
| dataset | `<数据集>.json` | 汇总指标、数据集档位、样本档位分布、M7（`depth_mask` / `percentile98` 两条规则） |
| dataset | `<数据集>_samples.jsonl` | 逐样本指标、档位、原因 |
| dataset | `summary.json` | 全部数据集一张表 |
| dataset | `summary_compact.tsv` / `summary_full.tsv` | 精简表（终端默认打印）/ 完整表，可直接贴 Excel |
| pseudo | `sequences.jsonl` | 逐序列指标、档位、原因、G1（有 gt_depth 时）、M7（`CONF_THR` 时） |
| pseudo | `summary.json` | 档位分布、指标中位数、pooled G1 |
| pseudo | `masks/<序列>/<帧>.npz` | `SAVE_MASKS=1` 时：verified / viol / boundary 逐像素掩码 |
| calib | `calib.json` | 选中的 τ、k、分档阈值和校准/留出集结果，可直接传给 `THRESHOLDS=` |

## 分档

每项指标按阈值分为 good / usable / marginal / reject。序列档位：
- **reject**（标签有错）：违例率、drift，或可观测时的尺度，任一项 reject；
- **U**（无法验证）：视差、有效像素比例或可用率不足，且没有触发 reject。说明"验证不了"，不等于标签错；
- **A / B / C**：全部 good / 全部不差于 usable / 其他。

尺度 s\* 在视差不足时不参与判定；s = 1 时已经自洽（违例率 ≤ 5%）而 s\* 偏离 1，也视为不可观测
（远景、小基线下一致点数对尺度不敏感），不判错。`reasons` 列出所有未达 good 的项。
改了分档规则后不必重跑数据：`python tools/quality_eval.py regrade --out quality_out` 按新规则重新分档（原档位存在 `grade_prev`）。
M6 BFR 不参与分档：> 10 时对该数据集开启边界处理（分母设 1% 下限，干净数据约 4–5）。

| 指标 | good | usable | reject |
|---|---|---|---|
| fsv | < 1% | ≤ 5% | > 15% |
| vpr | ≥ 60% | ≥ 30% | < 30% |
| drift | < 0.03 | ≤ 0.10 | > 0.10 |
| tri_angle_median | ≥ 3° | ≥ 1.5° | < 1.5° |
| valid_ratio | ≥ 50% | ≥ 5% | < 5% |
| best_scale | [0.9, 1.1] | [0.75, 1.34] | 之外 |

以上是初始值，伪标签上线前用 `calib` 在带真值的子集上校准后冻结。

## 自测

`tests/test_quality_metrics.py` 在解析渲染的房间场景上注入射线距离、尺度错误、位姿错配、边缘飞点、重复帧、纯旋转，
确认对应指标报警。合成伪标签序列上：干净 → A，局部深度错误 → C（conf 规则 Lift ≈ 177），位姿扰动 → B，尺度 ×2 → reject
（均为 `FRAME_STEP=2`）。
