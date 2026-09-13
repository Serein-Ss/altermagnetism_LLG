# altermagnetism_LLG

_有限温磁性体系的条件随机路径建模；2026-09-13 实文件目录重构。_

---

## 📍 检查入口

- [全部 Markdown 文档索引](docs/workspace/DOCUMENT_INDEX.md)
- [目录职责和迁移说明](docs/workspace/LAYOUT_20260913.md)
- [研究方案](GUIDE/)
- [本次迁移记录](logs/restructure/migration_20260913.json)
- [续跑校验记录](logs/restructure/resume_preflight_20260913.json)

## 📦 目录职责

| 目录 | 保存内容 |
|---|---|
| GUIDE | 远程下载的研究方案与执行计划 |
| assets/literature_reproduction | 按文献分类的分析指标、图表与可视化 |
| assets/research | 实际研究的分析指标、诊断与可视化 |
| conf | 文献物理参数、研究模型及实验参数、冻结配置 |
| data/literature_reproduction | 文献模拟数据及模拟断点 |
| data/research | 研究数据集、生成轨迹、参考数据 |
| output | 实际研究的模型 checkpoint；smoke 与 production 分开 |
| scripts | 按功能分类的代码；archive 保留原始 Python 内核 |
| slurm | 当前及历史任务提交脚本 |
| logs | 服务器和本地执行日志、迁移与任务记录 |
| docs | 分类 Markdown 文档、历史计划副本及文献附件 |

## 🔧 环境和任务

所有 Python 工作使用 `zrs-mag`，从本目录执行 `bash ../run_zrs_mag.sh ...`。
生产计算通过 Slurm 提交，任务名 `zrs-mag`；GPU 计算必须申请 GPU 分配。
本次续跑入口是 `scripts/workflow/resume_after_restructure.py`；不要直接重放旧路径的冻结提交命令。
恢复时保持物理参数、原始随机数状态及冻结内核一致，只适配文件位置。

## 🔐 GUIDE 和 Git

GUIDE 保留在本地，未从 Git 索引删除，因此不会通过删除提交误删远程计划。
本地提交钩子阻止提交 GUIDE 的暂存修改；已有 Git LFS 钩子保持不变。

```bash
# 下载远程 GUIDE；旧本地副本备份在 .git/ 下
bash scripts/workflow/guide_sync.sh download
# 暂存其他目录的变化，不执行 commit 或 push
bash scripts/workflow/guide_sync.sh stage
```

本次没有提交或推送远程仓库。大文件原件和上传分片均保留；不要推送含超限原件的备份分支。
