# 目录职责与数据批次

_2026-09-13 整理版：原路径保留，新增分类导航；不是物理迁移或清理磁盘。_

---

## 📍 一级目录

| 目录 | 用途 | 使用规则 |
|---|---|---|
| GUIDE | 用户研究计划与配套附录 | 保留原文，本轮不改 |
| conf | 材料参数、运行配置、合同草案 | frozen 内是历史绑定证据，不能原地改 |
| data | 原始参考、训练数据、生成路径与审计 | 按来源、体系、批次和 split 区分 |
| assets | 可视化图与动画 | literature_reproduction 按文献；research 为项目结果 |
| scripts | 按功能分类的代码 | archive 是冻结源码，不是新实验默认入口 |
| output | checkpoint、训练记录、定量报告 | 部分旧批次混存轨迹，见下表；暂不迁移 |
| slurm | 提交脚本 | 不因整理而重新提交 |
| logs | Slurm 和子进程日志 | 在运行任务日志不能移动 |
| NAVIGATION | 相对符号链接组成的分类入口 | 不复制数据，不用作新的运行根目录 |
| docs | 工作索引及历史说明文本 | 不取代 GUIDE 研究计划 |
| .git | Git 历史与 LFS 缓存 | 不手动删除对象目录 |

## 📊 当前有哪些数据

以下容量为整理时 du 的近似占用，会随正在运行的任务变化；不是独立物理样本数。

| 原目录 | 约占用 | 内容 |
|---|---:|---|
| data/literature_reproduction | 43 GiB | 五篇文献分 paper_id 保存的原文、raw 与 derived |
| data/datasets | 19 GiB | 旧 V2 及延长版、training_benchmark、V3 元数据等 |
| data/generated | 45 GiB | 旧模型生成轨迹和诊断；不是真实 LLG 参考 |
| data/research | 6.1 GiB | 驱动探索、旧零场与 Gate-F 前置数据 |
| output/bauer_campaign | 38 GiB | 独立 Bauer 参考/先导数据、报告以及上传分片 |
| output/bauer_first_loop | 153 MiB | 有界模型实验数据、checkpoint、采样及验收报告 |
| output/production | 27 MiB | 历史 V1/V2 模型训练结果 |
| assets | 369 MiB | 文献复现及本项目图像 |
| .git/lfs/objects | 118 GiB | 本地 LFS 缓存，含旧版本；不等同于当前工作文件大小 |

现有 Bauer 大数据放在 output 中是已记录的布局例外，不是建议的新规范。暂时保留是为了维持现有绝对路径、哈希、分片说明和任务引用。完整文件与对应分片约占两份空间，但内容不是两份独立数据。

## 📋 按研究归类

### 当前 Bauer 主线

- `output/bauer_campaign/20260913_classical_independent`：独立参考与先导；配置含不同用途的原条件和 extension，分别验收
- `output/bauer_first_loop/20260913_v1`：已经完成的有界模型实验；decision.json 为 STOP_SCALE_UP
- `output/bauer_revision4/20260913_023043`：三步长原条件补充、CPU/GPU 软件适配和待依赖诊断
- `data/literature_reproduction/bauer_2011/raw`：文献入口的真实长轨迹，含仍在写入的 partial

导航见 [01_bauer](../../NAVIGATION/01_bauer)。不要把这些独立批次简单合并或互换证书。

### 文献复现

五篇文献统一按 `bauer_2011`、`gomonay_2024`、`nishino_miyashita_2015`、`hirst_mn2au_2022`、`laliena_crnb3s6_2020` 查看。
每篇的 data、reports、figures、code 已集中到 [02_literature 导航](../../NAVIGATION/02_literature)，原物理目录不变。

### 历史模型与实验

- [03_legacy_v2](../../NAVIGATION/03_legacy_v2)：1/2/3 ps V2、旧模型和诊断；延长版本不能当作独立新增样本混用
- [04_old_gomonay](../../NAVIGATION/04_old_gomonay)：旧 Gomonay 前置检验；未通过与 inconclusive 原报告保留
- [05_frozen_evidence](../../NAVIGATION/05_frozen_evidence)：冻结源码、配置及审计，保留失败准备目录
- `data/literature`、`assets/literature`：更早布局的遗留入口，优先使用按 paper_id 分类的 literature_reproduction

## ⚠️ 本轮未执行的操作

没有删除原始数据、旧结果、缓存或 LFS 对象；没有移动任何运行输入输出、冻结代码、配置或 GUIDE；没有将目录名作为新的通过证书。

下一次物理迁移必须先等相关任务和依赖结束，列出旧路径→新路径清单，更新脚本、manifest、LFS 路径与重组说明，并验证哈希与引用。不能只移动文件再让脚本临时找路径。

## 🔗 校验与旧说明

[导航清单](../workspace/catalog.json) 记录每个别名的唯一原路径。
从根目录执行 `bash ../run_zrs_mag.sh -m scripts.workflow.workspace_catalog` 可核查全部导航链接。

旧目录说明完整保存在 [历史文本](../history/DIRECTORY_STRUCTURE_before_20260913.md.txt)；根目录其他旧状态文档的用途见 [文档索引](../workspace/DOCUMENT_INDEX.md)。
