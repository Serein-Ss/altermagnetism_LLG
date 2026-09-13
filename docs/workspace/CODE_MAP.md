# 代码入口与批次归属

_2026-09-13：只整理入口，不改求解器或科学结论。_

---

## 📍 Bauer 的三个不同入口

| 入口 | 对应产物 | 用途 |
|---|---|---|
| [bauer_campaign.py](../../scripts/workflow/bauer_campaign.py) | output/bauer_campaign | 独立参考批次；不等同于已有慢速补充任务 |
| [bauer_first_loop.py](../../scripts/workflow/bauer_first_loop.py) | output/bauer_first_loop | 有界训练与分布比较；以 decision.json 为准 |
| [bauer_revision4.py](../../scripts/workflow/bauer_revision4.py) | output/bauer_revision4 | 软件适配及原条件三步长补充验证 |

这三套记录不应只因同为 Bauer 就合并为一份证书。批次的参数、实现、路径数、保存频率、split 和哈希分别核对。

## 📚 通用代码分类

| scripts 子目录 | 职责 |
|---|---|
| core | Hamiltonian、LLG、积分器、流式存储和参数换算 |
| literature | 按文献保存的模型与复现入口 |
| generation | 原始参考轨迹生成与延长 |
| datasets | 惰性读取、划分和数据集构建 |
| model | 网络、基线、球面几何与图适配 |
| training / inference | 训练及 checkpoint 采样 |
| validation / tests | 科学验收工具与软件测试；两者不是同一层证据 |
| analysis / visualization | 数值分析、绘图和动画 |
| workflow | 任务编排、批次快照、目录导航 |
| archive | 运行时冻结证据；不作为新实验默认源码 |

## ⚠️ 历史入口

`gate_f_prerequisites.py`、`zero_field_campaign.py` 属于旧 Gomonay 前置/文献计划。
`train_gate_f.py` 等旧入口的实际适用范围必须检查合同，不能因为名称通用就用于任意体系。
旧 V2 的 `train.py` 和 `sample.py` 与 Bauer 新主线分开使用。

## 🔧 环境与运行约定

从项目根目录使用 `bash ../run_zrs_mag.sh ...`。GPU 计算通过 Slurm 分配并显式使用 CUDA；大量 CPU 数据任务使用 fat。目录整理不构成提交任务的命令。
