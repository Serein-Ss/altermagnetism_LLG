# altermagnetism_LLG

_有限温磁性体系的条件随机路径建模；2026-10-19 正式入口、开发checkpoint和最新诊断整理。_

最新状态以 [2026-10-19进展](GUIDE/CURRENT_PROGRESS_20261019.md) 和 [精简结果网页](assets/research/bauer_path/progress_20261019/index.html) 为准。模型可运行，但条件路径统计仍未通过物理认证；项目计算已按用户要求停止。本次发布不恢复训练或自动续提。

---

## 📍 检查入口

- [修订9：物理调研、模型选择与下一轮受控实验](GUIDE/PATH_MODEL_AND_PHYSICS_PLAN.md)：保留单步转移，对照整段路径；世界模型是功能定位，非已完成能力。

- [暂定冻结的核心目标与生成接口](GUIDE/GENERATIVE_DYNAMICS_SCOPE.md)：尺寸、温度、seed及逐轴边界；方法覆盖1D/2D/3D分别训练与生成。
- [当前项目完整总览：目标、数据、结果、目录和下一步](GUIDE/PROJECT_OVERVIEW.md)
- 当前第一阶段：小体系有限时间联合路径学习，以及相同时间窗口下的尺寸推广；短时动力学准确性尚未证实。
- [全部 Markdown 文档索引](docs/workspace/DOCUMENT_INDEX.md)
- [目录职责和迁移说明](docs/workspace/LAYOUT_20260913.md)
- [研究方案导航](GUIDE/README.md)
- [核心主线与文章证据链](GUIDE/CORE_RESEARCH_MAINLINE.md)
- [当前工作计划和任务快照](GUIDE/WORK_PLAN.md)
- [本次迁移记录](logs/restructure/migration_20260913.json)
- [续跑校验记录](logs/restructure/resume_preflight_20260913.json)

## 📦 目录职责

| 目录 | 保存内容 |
|---|---|
| GUIDE | 可正常提交推送的研究方案、核心主线与执行计划 |
| assets/literature_reproduction | 按文献分类的分析指标、图表与可视化 |
| assets/research | 实际研究的分析指标、诊断与可视化 |
| conf | 文献物理参数、研究模型及实验参数、冻结配置 |
| data/literature_reproduction | 文献模拟数据及模拟断点 |
| data/research | 研究数据集、生成轨迹、参考数据 |
| output | 实际研究模型checkpoint；最新三种子候选在development/bauer_path，未通过物理认证；smoke留tests |
| scripts | 按功能分类的代码；archive 保留原始 Python 内核 |
| slurm | 当前及历史任务提交脚本 |
| logs | 服务器和本地执行日志、迁移与任务记录 |
| docs | 分类 Markdown 文档、历史计划副本及文献附件 |

## 🔧 环境和任务

测试与尚未验证的实验统一放在 `tests/`：`unit/` 单元测试、`experiments/` 调试代码、`conf/` 参数、`slurm/` 提交脚本、`data/` 实验数据、`output/` checkpoint、`results/` 分析、`logs/` 日志。验证并确认后再合入正式目录；冻结历史归档保留不改。

所有 Python 工作使用 `zrs-mag`，从本目录执行 `bash ../run_zrs_mag.sh ...`。
生产计算通过 Slurm 提交，任务名 `zrs-mag`；所有 GPU 任务固定使用 `--partition=rtx4090` 并申请 GPU 分配，不使用 `gpu` 分区。CPU/fat 任务不受此规则影响。
历史目录迁移续跑入口是 `scripts/workflow/resume_after_restructure.py`，不是当前默认任务入口；当前任务以 GUIDE/WORK_PLAN.md 和冻结 submission 记录为准，不重复提交。
恢复时保持物理参数、原始随机数状态及冻结内核一致，只适配文件位置。

## 🔐 GUIDE 和 Git

GUIDE 现在允许和代码一起提交推送。根目录 `data/` 与实验原始数据 `tests/data/` 继续忽略；缓存和临时文件也保持忽略。本地提交钩子只防止原始数据入库，不再阻止 GUIDE。已有 Git LFS pre-push 钩子保持不变。

```bash
# 暂存包含 GUIDE 的非数据变更；执行前先检查 git status
bash scripts/workflow/guide_sync.sh stage
# 检查暂存内容，然后自行 commit / push
git diff --cached --stat
# 可选下载远程 GUIDE：本地 GUIDE 有改动时拒绝覆盖，旧副本备份在 .git/
# bash scripts/workflow/guide_sync.sh download
```

本轮按用户授权整理正式入口、进展与开发checkpoint并发布。原始数据、测试大体积产物和缓存排除；冻结测试档案保留，不作成功结论。`.gitignore` 不清除远程历史，也不自动取消已跟踪文件。不要推送含超限原件的历史备份分支。
