# 分类浏览入口

_2026-09-13：这是原目录的导航视图，不是第二份数据，也不是新的运行工作目录。_

---

## 📍 按任务查看

| 分类 | 内容 | 注意 |
|---|---|---|
| [01_bauer](../../../NAVIGATION/01_bauer) | 当前 Bauer 参考与开发批次 | 各批次分别看报告，不合并认定通过 |
| [02_literature](../../../NAVIGATION/02_literature) | 五篇文献的代码、数据、报告、图 | 按 paper_id 分开 |
| [03_legacy_v2](../../../NAVIGATION/03_legacy_v2) | 旧驱动 V2 数据、checkpoint 和评估 | 不是 Bauer 零场训练集 |
| [04_old_gomonay](../../../NAVIGATION/04_old_gomonay) | 旧 Gomonay 前置检验和探索数据 | 历史证据，不自动重启 |
| [05_frozen_evidence](../../../NAVIGATION/05_frozen_evidence) | 冻结代码、配置和审计 | 不编辑快照 |
| [06_operations](../../../NAVIGATION/06_operations) | 活动代码、Slurm、日志、计划 | GUIDE 只读导航 |

## ⚠️ 使用边界

这些目录中的入口是相对符号链接，指向仓库内唯一的原目录。它们不复制大文件、不释放空间，也不改变任何 manifest 中的路径。

打开链接后修改文件，会修改原文件；不要把导航目录当备份。统计磁盘或扫描数据时不应跟随这些链接重复计数。所有脚本仍从项目根目录运行，不从 NAVIGATION 运行。

## 🔧 核查入口

在项目根目录执行：

```bash
bash ../run_zrs_mag.sh -m scripts.workflow.workspace_catalog
```

若另一台机器没有检出目录符号链接，可直接使用 [原路径清单](../../workspace/catalog.json)；不要把这些别名写入新数据合同。

## 🔗 详细说明

查看 [目录职责与批次说明](../DIRECTORY_STRUCTURE.md) 和 [代码入口表](../../workspace/CODE_MAP.md)。
