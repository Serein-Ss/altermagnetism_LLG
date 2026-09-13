# 项目目录与使用约定

2026-09-09 后续更新：文献材料已进一步按 paper_id 分类，新增 conf 与纯约化 R0 入口；详见 [文献目录与运行说明](LITERATURE_REPRODUCTION.md)。下文保留上一轮总体目录说明，不代表新文献基准已通过。
所有运行使用父目录的 `bash run_zrs_mag.sh ...`，即 `zrs-mag` 环境。

## 一级目录

| 目录 | 职责 |
|---|---|
| GUIDE/ | 用户上传的逐步研究和工作计划；保留计划原文，不能视为完成证明 |
| data/ | 文献复现、研究验证、训练/验证/测试数据、生成轨迹和审计数据 |
| assets/ | 全部可视化分析产物，区分 literature 与 research，再按任务保存 |
| scripts/ | 全部活动代码分类保存；archive 仅为冻结的历史代码证据 |
| output/ | 模型 checkpoint、训练记录和定量评估报告；区分 smoke、production、diagnostics |
| slurm/ | 当前 Slurm 提交脚本；archive 保存历史提交脚本原文 |
| logs/ | 服务器任务日志、子进程日志和运行环境记录 |

`model/` 已拆入 scripts；`outputs/` 已改为单数 output 并分类。
`.git/` 是版本库及 LFS 对象，不是实验数据入口；Python/pytest 缓存不是研究产物。

## 数据清单

| 当前目录（相对 data/） | 内容与边界 |
|---|---|
| literature/literature_validation/ | Gomonay 自旋波轨迹与验证指标，以及 Gomonay、Bauer 参考论文 |
| literature/path_literature_validation/ | Bauer 磁链加速 pilot 轨迹、势垒与机制分析，不等于完整寿命认证 |
| literature/noncollinear_validation/crnb3s6/ | CrNb3S6 非共线螺旋态验证数据 |
| research/multimodality_validation/ | 多模态随机路径集合与统计 |
| research/multimodality_validation_dt0p05/ | 较小时间步下的多模态验证 |
| research/multimodality_afm_control/ | Jtilde=0 的 AFM 数值对照，不替代独立材料文献复现 |
| research/screening/ | 驱动强度粗扫、细扫及机制结果 |
| research/size_convergence/ | 尺寸与时间步收敛扫描报告 |
| research/sot_probe_*/ | 0.4、0.6、0.8、1.0、1.2 T 的初步驱动探测 |
| research/scalable_paths/ | 可变尺寸数据流程的软件冒烟样本 |
| research/unbiased_smoke/ | 四条小型随机轨迹，仅供流程检查 |
| research/*.json | 配置/噪声响应与多模态对照的汇总指标 |
| datasets/training_benchmark/ | 小规模统一全空间轨迹基准、manifest 与检查报告 |
| datasets/standard_v2/ | 1 ps V2 数据集、数据卡与阶段性检查报告 |
| datasets/standard_v2_extended_2ps/ | 延长到 2 ps 的 V2 轨迹 |
| datasets/standard_v2_extended_3ps/ | 延长到 3 ps 的 V2 轨迹 |
| datasets/standard_v3/ | registry 与候选协议；目前没有 V3 正式轨迹 |
| audit/software_check_20260909/ | 缩小规模软件预检数据，不能用于正式认证 |
| audit/20260909_3b3b824_guide_thermal/ | 热噪声检查、120 次 Nishino 运行及集合判定；最终门槛未通过 |
| generated/production/ | 正式训练实验对应的模型生成轨迹，按版本、种子、评估尺寸保存 |
| generated/diagnostics/ | 旧 checkpoint 的分尺寸评估与诊断消融轨迹 |

V2 的 `train/` 文件包括内部 train/validation/test 划分，不是所有路径都用于拟合。
`test_large/` 是 L96 尺寸外推测试。原始 V2 有 330 条轨迹，216/27/87 条分别
用于训练/验证/测试；延长版是观测窗口延长，不应当作独立新增样本混入训练。
模型生成轨迹必须与真实 LLG 数据分开，不作为真实测试参考。

## 代码类别与入口

| scripts 子目录 | 内容 |
|---|---|
| core/ | 哈密顿量、LLG、积分器、平衡统计、产物路径工具 |
| literature/ | 自旋波、Bauer、CrNb3S6、Nishino 文献基准入口 |
| generation/ | 随机轨迹生成、V3 分片生成、观测时间延长 |
| datasets/ | 数据集读取、惰性加载、批处理、初态池构建 |
| model/ | 网络骨干、确定性基线、球面几何操作 |
| training/ | 训练目标与训练入口 train.py |
| inference/ | checkpoint 条件采样入口 sample.py |
| analysis/ | 翻转事件定义、指标、模型评估、消融、空间机制与集合分析 |
| validation/ | 噪声、收敛、数据一致性和认证门槛检查 |
| visualization/ | 绘图、动画、结果刷新；model_plots.py 保存模型评估绘图函数 |
| tests/ | 软件及物理回归测试，包括目录与路径回归 |
| workflow/ | 快照与依赖任务提交编排 |
| archive/ | 历史审计代码快照，保留内容和内部结构，不作为当前运行入口 |

代码中的少量审计摘要图仍由对应审计入口调用，写入位置统一由
`core/project_paths.py` 分流到 assets。模型数值计算不因路径重构改变。

从项目父目录运行：

```bash
# CPU 回归测试，不启动正式模拟或训练
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  bash run_zrs_mag.sh -m pytest altermagnetism_LLG/scripts/tests -q

# 查看新入口参数，不执行训练/采样
bash run_zrs_mag.sh -m altermagnetism_LLG.scripts.training.train --help
bash run_zrs_mag.sh -m altermagnetism_LLG.scripts.inference.sample --help
bash run_zrs_mag.sh -m altermagnetism_LLG.scripts.analysis.evaluate --help
```

GPU 训练和模拟仍必须通过 Slurm 申请 GPU，并使用 `--device cuda`。
提交现有普通 sbatch 的工作目录仍为项目父目录；日志目标为项目的 `logs/slurm/`。
仅在确实希望运行时提交，不要把查看脚本当成授权执行正式任务。

## 模型结果、图片和日志

`output/production/production_v2/seed_20260908/`（另有 20260909、20260910）下：

- deterministic/：确定性基线 checkpoint 与训练指标。
- flow/：条件流模型 checkpoint 与训练指标。
- evaluation/L16、L32、L64、L96/：定量评估 JSON。

同一任务通过相同后缀关联：

```text
output/production/production_v2/seed_20260908/evaluation/L96/         指标
data/generated/production/production_v2/seed_20260908/evaluation/L96/ 轨迹
assets/research/models/production/production_v2/seed_20260908/evaluation/L96/ 图与动画
```

`output/smoke/` 是软件/GPU 小测试；`output/diagnostics/` 是模型诊断。
production 只表示实验用途，不表示模型性能或物理认证已通过。

`assets/literature/` 保存文献复现图，`assets/research/` 保存本研究数据图、审计图、
模型评估图。V3 需求文档已从 assets 移到 GUIDE。
`logs/slurm/` 保存原普通任务日志，`logs/audit/<batch>/` 保存审计任务及环境日志。

## 历史记录与迁移验证

`DIRECTORY_MIGRATION.json` 记录 990 个文件的旧位置、新位置、大小和迁移时 inode。
同文件系统 rename 后逐项核对 inode 与大小，未复制/重写科学数据负载。
活动源码和路径配置随后按新布局修改。历史 JSON、HDF5 属性、checkpoint 元数据、
GUIDE 原文及审计快照保留旧路径，避免伪造历史。需要读取这些路径时用
`resolve_recorded_path()`；它依据迁移表解析，而不是修改原始证据。

历史审计 submission.json 中的代码哈希仍对应原始快照内容；快照现位于
`scripts/archive/<batch>/code/`，历史 batch 原文在 `slurm/archive/<batch>/`。
归档脚本不保证可直接重跑，复现应使用新入口并记录新的批次。

LFS 清单及 .gitignore 已更新；`LARGE_DATA.md` 给出分片还原方法。
已有 HDF5 原件、上传分片和 .git/lfs 对象是不同用途的副本，本次不清理。
历史缓存也保留。任何清理、重新训练或远程推送均不属于本次重构。
