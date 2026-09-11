# 严格文献复现与约化 LLG 重构实施方案

> 版本：2026-09-09
>
> 适用仓库：`altermagnetism_LLG`
>
> 本文规定五篇文献的原条件、特殊方程及逐项复现门槛；零场路径数据与生成模型按 [研究与验收协议](PUBLICATION_GRADE_RESEARCH_PROTOCOL_20260911.md) 执行。两者分工互补，新协议不豁免本文失败项；发现冲突应记录并暂停依赖它的生产步骤，不自行采用较宽松标准。

## 1. 目标、成功标准与不能夸大的结论

本阶段的目标不是生成模型训练数据，而是用同一套 PyTorch 自旋动力学代码复现五篇文献中相互独立的解析结果、平衡统计、热激活路径、反铁磁动力学、交错磁色散和非共线纹理，从而建立后续数据生成器的证据链。

完成全部复现后可以声称：在已经覆盖的哈密顿量、边界条件、温度范围、时间步和观测量上，代码与解析结果或文献结果一致，并通过了预先规定的数值收敛测试。不能声称“程序在所有体系和所有条件下绝对没有任何问题”；有限个基准无法证明不存在未覆盖的缺陷。

正式放行必须同时满足：

1. 文献公式和邻接逐项审计通过；
2. 带单位原值到约化参数的换算可复算、可测试；
3. 有效场等于约化哈密顿量对自旋的负梯度；
4. 确定性和随机积分器分别通过时间步收敛；
5. 目标文献图或原文数值在预注册指标下通过；
6. 原始数据、派生数据、图、报告和配置具有同一 `run_id`、Git commit 和 SHA-256；
7. 一个体系失败时不得用另一个体系的成功结果替代，也不得进入正式数据集生成。

## 2. 参数与代码必须分层

### 2.1 四层参数

每个 YAML 配置必须把参数分成四层：

- `source_parameters`：文献中的原值和原单位，只用于溯源与换算；
- `reduction`：参考能量、参考磁矩、参考长度和参考时间的定义；
- `reduced`：求解器唯一允许读取的无量纲物理参数；
- `numerics`：网格、时间步、总步数、保存间隔、随机种子和收敛扫描等数值选择。

文献未给出的 `dt`、随机种子、重复数和误差阈值必须标为 `numerical_choice` 或 `validation_extension`，不得标成文献参数。

### 2.2 通用约化 Gilbert LLG

对单位自旋 `s_i`，定义

```text
h = H / E0
b_i = -∂h/∂s_i
tau = t / t0
t0 = mu0 / (gamma E0)
theta = k_B T / E0
```

通用无量纲随机 Gilbert 方程为

```text
ds_i/dtau = -[s_i × (b_i + eta_i)
              + alpha s_i × (s_i × (b_i + eta_i))] / (1 + alpha^2)
```

并采用 Stratonovich 解释：

```text
<eta_i,mu(tau)> = 0
<eta_i,mu(tau) eta_j,nu(tau')>
    = 2 alpha theta delta_ij delta_mu,nu delta(tau-tau')
```

离散到步长 `delta_tau` 后，每个独立分量的标准差为

```text
sqrt(2 alpha theta / delta_tau)
```

求解器不得读取 tesla、joule、meV、mRy、kelvin、second、meter 等量。实际单位只存在于 YAML 的 `source_parameters` 与 `reduction` 中；需要回到物理单位绘图时，由后处理使用同一换算记录恢复。

### 2.3 文献特殊方程不能被通用公式抹平

Bauer 2011 使用文中 Landau–Lifshitz 随机方程约定：噪声进入进动项，确定性阻尼使用 `lambda`，且 `epsilon^2 = 2 lambda theta`。该复现必须选择 `equation_convention: bauer_ll`，不能静默改成 Gilbert 约定。

Nishino–Miyashita 2015 必须同时复现：

- case A：固定 `alpha=0.05`，由涨落耗散关系计算每个温度的噪声强度；
- case B：固定 `D=1.0`，由涨落耗散关系计算每个温度的 `alpha`。

两种约定具有相同平衡分布，但动力学时间尺度不同，不能只做 case A 后声称完整复现图 1。

## 3. 配置、脚本、数据和结果目录

每篇文献使用固定 `paper_id`：

```text
nishino_miyashita_2015
bauer_2011
hirst_mn2au_2022
gomonay_2024
laliena_crnb3s6_2020
```

目录必须为：

```text
conf/
  reduced_llg.yaml
  literature/<paper_id>.yaml

scripts/
  core/literature_config.py
  config/validate_literature_configs.py
  literature/<paper_id>/
    model.py
    run.py
    analyze.py
    plot.py

data/literature_reproduction/<paper_id>/
  reference/       # 数字化文献曲线、图注和来源映射；不放模拟结果
  raw/<run_id>/    # 不可修改的原始轨迹/统计量
  derived/<run_id>/# 由 raw 计算的频谱、寿命、拟合参数等

assets/literature_reproduction/<paper_id>/<run_id>/
  figures/         # 只保存 PNG
  animations/      # GIF/MP4

output/literature_reproduction/<paper_id>/<run_id>/
  logs/
  report.json
  report.md
  manifest.json
```

禁止五篇文献共用一个 `validation.json`、一个 `trajectory.npz` 或一个未标识来源的图片目录。`reference/` 中提取的论文图必须记录论文 DOI、图号、页码、提取方式和文件 SHA-256。

## 4. 配置冻结和运行流程

每次正式运行按以下顺序执行：

1. `python scripts/config/validate_literature_configs.py`；
2. 运行有效场自动微分检查、边界条件检查和零温定态检查；
3. 运行最小 smoke test，只检查程序与输出结构，不参与论文结论；
4. 运行 `dt, dt/2, dt/4` 耦合噪声或确定性收敛测试；
5. 选择通过收敛的最大步长，冻结 YAML，并记录配置 SHA-256；
6. 创建新的 `run_id=YYYYMMDD_<git7>_<config8>`；
7. 只向 `raw/<run_id>.partial` 写入，完成检查后原子改名；
8. 分析脚本只读 `raw`，图脚本只读 `derived` 与 `reference`；
9. 生成逐项验收报告；任何硬指标失败则 `status: fail`。

正式运行中不允许：自动调整物理参数、筛掉失败轨迹、根据结果修改阈值、用加速条件替换论文条件、覆盖旧 `run_id`。

## 5. 共用验收门

### Gate 0：配置与溯源

- DOI、论文版本、勘误和目标图号正确；
- YAML 的原值、单位、约化公式、约化值可由独立测试复算；
- HDF5/NPZ/JSON 中保存 config hash、code commit、seed、dtype、设备和完整数值协议。

### Gate 1：哈密顿量和邻接

- 每种键类型用手工小晶格检查键数、符号和是否重复计数；
- 周期边界、开放边界和混合边界分别测试；
- 随机双精度自旋上比较解析有效场与自动微分 `-∂h/∂s`，最大相对误差 `<1e-10`，近零分量使用绝对误差 `<1e-10`；
- 全局自旋旋转只在哈密顿量物理上允许时检查不变性，不能对含各向异性的模型错误要求完整 SO(3) 不变。

### Gate 2：积分器和随机热浴

- 确定性 `alpha=0`：自旋模长误差 `<1e-10`，相对能量漂移随减半步长按预期下降；
- 阻尼 `alpha>0, T=0`：无外驱时能量不得系统上升；
- 随机项：至少 `10^6` 个独立分量检查均值、方差、交叉分量相关和时间自相关；
- Stratonovich Heun/中点使用同一步 Wiener 增量完成 predictor/corrector；
- 对依赖人工归一化的积分器，必须同时保存归一化前模长误差，并与几何中点结果比较平衡观测量。归一化后模长为 1 不能单独证明积分器正确。

### Gate 3：数值收敛

- 文献报告 `dt` 时先使用原值，并额外做 `dt/2`；
- 文献未报告 `dt` 时，`dt, dt/2, dt/4` 都是本项目数值选择，以主要观测量差异小于预注册容差为准；
- 保存间隔只影响输出，不得改变积分步长；频谱必须满足 Nyquist 和分辨率要求；
- 有限体系结果必须做至少三个尺寸或按论文原尺寸运行，不能把小尺寸加速结果叫严格复现。

## 6. Nishino–Miyashita 2015

### 6.1 严格目标

来源：Phys. Rev. B 91, 134411 (2015)，并检查 Phys. Rev. B 97, 019904 (2018) 勘误。

第一目标是原文图 1：`N=10^3` 个非相互作用磁矩，`h=2`、`M=1`、`gamma=k_B=1`、`delta_tau=0.005`，共 80,000 步，其中前 40,000 步平衡、后 40,000 步测量。主积分器使用附录 B 的中点方法；同时复现 case A 和 case B，并与 Langevin 函数

```text
m(theta) = coth(2/theta) - theta/2
```

比较。

### 6.2 当前问题和必须修改的脚本

当前 `scripts/literature/validate_nishino_full.py` 只生产 case A，且把 Heun、中点和三个时间步混在同一正式结果中。它还把每个温度拆成 10 个额外重复，这可以用于本项目置信区间，但不是论文原条件。

重构为：

```text
scripts/literature/nishino_miyashita_2015/run.py
scripts/literature/nishino_miyashita_2015/analyze.py
scripts/literature/nishino_miyashita_2015/plot.py
```

`run.py` 从 YAML 读取 case、温度索引和重复索引；论文的中点方法是主结果，Heun 与 `dt/2, dt/4` 只进入数值附录。

### 6.3 旧数据清理与重新生成

在服务器确认新脚本、配置测试和 smoke test 全部通过后，删除以下旧结果及其对应图片：

```text
data/audit/20260909_3b3b824_guide_thermal/nishino/
data/audit/software_check_20260909/nishino/
assets/literature/audit/20260909_3b3b824_guide_thermal/nishino_decision.png
```

必须先生成删除清单并核对解析后的绝对路径位于仓库内，再删除；删除后立即以新目录结构重跑，不保留混合新旧结果。本地当前不提前删除这些数值文件，因为本机不执行正式重跑，避免仓库出现“旧结果已删、新结果为空”的中间状态。

### 6.4 修正后的判断指标

原先“差值既小于容差、又小于 CI 半宽”的判定不是等价性检验。新验收采用：

```text
abs(mean_sim - mean_exact) + CI95_halfwidth <= tolerance_abs
```

其中 `tolerance_abs=0.01` 预先冻结。该式要求整个 95% 置信区间都落在解析值的等价带内。另对 `cos(theta)` 分布做两样本/解析 CDF 检查并报告效应量；多温度检验使用 Holm 校正。每个 case 必须分别通过，不能平均后掩盖失败温度。

## 7. Bauer et al. 2011

### 7.1 严格目标

来源：J. Phys.: Condens. Matter 23, 394204 (2011)。使用论文开放链随机 Landau–Lifshitz 方程：`L=100`、`K/J=0.1`、`theta=k_BT/J=0.11`、`lambda=0.1`、总时长 `750000 hbar/J`，无外场、无 SOT、无重要性筛选。

首先复现论文条件下的典型反转轨迹：边缘成核、畴壁形成和随机游走传播；随后按论文给出的长度和温度点复现寿命标度及 Arrhenius 能垒等于畴壁能量的结论。长度/温度序列必须从论文图或正文逐点录入并保存在 `reference/`，禁止自行补点后称为论文复现。

### 7.2 禁止加速替代

删除 `accelerated_pilot` 作为默认或正式入口。`L=40`、`theta=0.20`、`duration=500` 的旧数据只能标记为历史软件测试，不参与任何文献误差、机制统计或模型认证。

论文没有报告积分时间步和随机种子，因此当前 `dt=0.02` 不是文献参数。必须以 `0.02, 0.01, 0.005` 做收敛扫描并使用共同 Wiener 路径；只有观测量收敛后才能冻结生产步长。不得把较大时间步称为“论文原始设置”。

### 7.3 样本量与验收

- 静态检查：离散畴壁能与连续近似仅作为哈密顿量单元测试，不是动力学复现；
- 轨迹检查：所有轨迹等权保存，反转前后路径完整，不只保存成功事件；
- 正式寿命估计：每个条件至少 500 个完成反转事件；未在观察窗内反转的样本作为右删失进入 Kaplan–Meier/生存模型；
- 边缘成核比例、首达时间分布、寿命对链长和温度的拟合均给 bootstrap 95% CI；
- Arrhenius 拟合能垒与论文/畴壁能的相对差异目标 `<10%`，斜率和截距的不确定度必须报告。

## 8. Hirst et al. 2022：Mn2Au

### 8.1 参数版本修正

严格复现使用 Phys. Rev. B 106, 094402 (2022) 正式发表版，不使用 arXiv:2206.08625 v1 的旧参数表。正式版 Table I 为：

| 参数 | 原值 | 约化参考 `E0=30.8040 mRy` 后 |
|---|---:|---:|
| `J1` | `-5.3422 mRy` | `-0.1734255292` |
| `J2` | `0.6484 mRy` | `0.0210492144` |
| `J3` | `-0.6341 mRy` | `-0.0205849890` |
| `J4` | `-6.8986 mRy` | `-0.2239514349` |
| `J0,same` | `2.5934 mRy` | `0.0841903649` |
| `J0,inter` | `-30.8040 mRy` | `-1` |
| `d_z` | `-0.0663 mRy` | `-0.0021523179` |
| `d_x*` | `0.0026 mRy` | `0.0000844046` |

晶格常数为 `a=3.330 Å`、`c=8.537 Å`，约化后只向求解器提供 `c/a=2.5636636637`；磁矩 `3.8663 mu_B` 只用于定义时间、场和温度换算。`J1/J2/J3` 每个原子各有四个相互作用，`J4` 每个原子一个，必须由 Fig. 1 的实际位移模板实现，不能只用配位数乘平均场。

### 8.2 复现层次

1. **Fig. 2 平衡性质**：30×30×30 周期体系、随机 s-LLG、Heun、平衡计算中的 `lambda=1`；复现子晶格磁化、纵向磁化率和 `T_N≈1335 K`。温度在求解器中以 `theta=k_BT/E0` 输入，`T_N=1335 K` 对应 `theta≈0.27449`。
2. **Fig. 3 AFMR**：两个子晶格从易平面转出 20°，分别在 300、1000、1200 K 复现 `m_z(t)`；对应约化温度约为 0.061683、0.205610、0.246732。阻尼按原文对应条件分别录入，不得用高阻尼加速。
3. **Fig. 11 畴壁**：复现约 31.2 nm 的零温/低温平衡畴壁宽度，并检查 ASD 与论文宏观模型的对应观测量。

### 8.3 实现前置条件

当前仓库没有 Mn2Au 的三维晶格与 `J1–J4` 精确邻接，因此现有任何二维双层 AFM 都不能冒充 Hirst 复现。必须新增专用 `model.py`，并通过小晶胞键表、能量/自动微分场、平移周期性、AFM 基态和交换和式五项检查后，才能运行文献图。

## 9. Gomonay et al. 2024

### 9.1 严格目标和约化参数

来源：npj Spintronics 2, 35 (2024)，离散双层哈密顿量使用补充材料 Eq. (S.8)，动力学使用 Eq. (S.9)，色散使用 Eqs. (S.14)–(S.16)。以 `E0=J1=11.1 meV`、`a0` 为长度尺度：

```text
J1/E0       = 1
J2/E0       = 0.1693693694
J_tilde/E0  = 0.0720720721
K_SW/E0     = 0
K_DW/E0     = 0.0042342342
```

### 9.2 必须复现的结果

1. **Fig. 2 / Fig. S2 色散**：论文使用 500×500 磁矩体系、时空 sinc 激励和时空 Fourier 变换。必须覆盖第一布里渊区的多条高对称路径与角度，而不是当前单个 `(k,k)` 模式。验收包括两支频率、分裂 `Delta omega`、节点方向 `[100]/[010]`、最大分裂方向 `[110]/[1-10]` 及旋转 90° 后符号改变。
2. **`J_tilde=0` 控制**：分裂在数值精度内消失；它是哈密顿量消融，不是独立材料文献复现。
3. **Fig. 3 / Fig. S4 静态畴壁**：论文的 10000×30 体系沿短边周期、长边开放，先用 `alpha_G=0.75` 按原文弛豫；复现 `n_z`、`m_z` 和随畴壁晶向改变的磁化梯度。
4. **Fig. 5 / Fig. S5 动力学**：在论文驱动定义下复现速度、相位进动和 Walker breakdown。只有前两项完成时可先发布“色散与静态纹理复现”，不能写成整篇动力学已复现。

当前 `validate_spinwave.py` 使用 16×16、单一模式和 SI 内核，只证明某个频率点与同一代码内的解析式相符；它不是论文 Fig. 2/S2 的严格复现。现有模型两个方向均周期，也不能直接用于论文长条畴壁边界。

## 10. Laliena et al. 2020/2022：CrNb3S6

### 10.1 必须先修正的内容

来源为 Scientific Reports 10, 20430 (2020)，DOI `10.1038/s41598-020-76903-8`；必须同时应用 2022 Author Correction，DOI `10.1038/s41598-022-06147-1`。勘误修正了式 (13)、式 (14)、图 2b、图 3a/3b，并将 BVP 临界值修正为 `Gamma_c=1.2405`。

当前 `validate_crnb3s6_helix.py` 只检查一个人工选成 48 nm 的周期盒中一周螺旋是否近似定态；它没有实现修正后的 BVP，也没有复现论文的受电流孤立手性孤子结果。因此旧结果只能保留为单元测试，不能作为 Laliena 严格复现。

### 10.2 原参数与约化参数

文献参数为 `A=1.42 pJ/m`、`D=369 μJ/m²`、`K=-124 kJ/m³`、`M_s=129 kA/m`、`alpha=0.01`、`beta=0.02`、`P=1`、`B_y=300 mT`、晶格常数 `a=0.6 nm`。定义

```text
q0 = D/(2A)
x' = q0 x
B0 = D^2/(2 A M_s)
kappa = 4 A K / D^2
h_y = B_y/B0
```

得到约化值：

```text
q0*a   = 0.0779577465
kappa  = -5.1726999655
h_y    = 0.8071914865
alpha  = 0.01
beta   = 0.02
```

### 10.3 复现顺序与验收

1. 无场从非约束初态弛豫，不预先固定 winding，测得 `q/q0` 和螺旋周期；盒长扫描消除 48 nm 人工整周期偏置；
2. 实现勘误后的 BVP 方程，复现修正图 2b 的 `Gamma_c(h_y/h_yc)`；
3. 在 `kappa=-5.17, h_y=0.807, Gamma=0.89` 复现修正图 3a 的稳态孤子剖面；
4. 扫描电流复现图 3b 的速度、中心倾角和宽度，并检查 `j_c=1.372 TA/m²` 对应 `Gamma=1.224` 与 BVP `Gamma_c=1.2405` 一致；
5. 曲线用数字化文献点或解析 BVP 独立比较，禁止把同一实现产生的解析曲线当外部真值。

## 11. 每篇文献的报告必须回答什么

每个 `report.md` 采用完全相同的结构：

1. 论文、版本、勘误、目标公式和目标图；
2. 文献原参数表；
3. 约化定义和逐项换算；
4. 哈密顿量、邻接、边界和积分器实现映射；
5. 数值参数中哪些是论文值、哪些是本项目选择；
6. 时间步、尺寸、时长和统计收敛；
7. 文献图与复现图并排图；
8. 每个指标的定义、数值、置信区间、阈值和 pass/fail；
9. 已证明范围、未证明范围和剩余偏差；
10. 原始文件、配置、代码和图片 SHA-256。

若论文没有公开逐点原始数据，图像数字化误差必须单列，复现容差不得小于数字化不确定度。

## 12. 推荐执行顺序与停止规则

```text
R0  完成约化求解器、YAML 加载、转换测试和通用 Gate 0–3
R1  Nishino Fig. 1 case A/B
R2  Bauer 静态单元测试 + 原条件长时随机路径
R3  Hirst Mn2Au Fig. 2/3/11
R4  Gomonay Fig. 2/S2，再做静态/动态畴壁
R5  Laliena 修正 BVP 与修正图 2/3
R6  汇总跨体系认证矩阵；全部通过后才冻结正式数据生成器
```

停止规则：R0 或 Nishino 失败，停止所有有限温生产；某一模型的邻接/场检查失败，只停止该模型；长时统计不足时状态为 `inconclusive`，不能按 pass 处理。

服务器运行应先用 10,000 步计时，再按实测吞吐量申请 walltime。计算资源与物理协议独立；使用 CPU/GPU 不改变方程，但必须记录设备、dtype 和随机数实现。不得为缩短运行时间更改温度、尺寸、阻尼或总物理时长。

## 13. 当前仓库状态与下一步实际编码任务

本次已经完成：

- 清理旧 GUIDE，只保留本实施方案；
- 建立通用约化 LLG 规范和五篇文献的 YAML 配置候选；
- 建立只向运行脚本暴露 `reduced` 与 `numerics` 的配置加载器；
- 将正式出版版/勘误版与旧预印本参数差异写入配置；
- 增加配置可加载和约化换算回归测试。

仍未完成、不得误报为已完成：

- 现有 `unified_llg.py` 仍是旧 SI/混合单位实现，尚未替换为纯约化求解内核；
- 五篇文献脚本尚未全部迁移到按 `paper_id` 分目录并加载 YAML；
- Mn2Au 三维精确邻接尚未实现；
- Gomonay 全布里渊区和混合边界长条体系尚未实现；
- Laliena 勘误后的 BVP 与电流驱动尚未实现；
- Bauer 原时长统计和 Nishino case A/B 新数据尚未重新运行；
- 本次没有删除旧 Nishino 数值结果，因为服务器端新实现尚未就绪。

下一次编码应只执行 R0：先增加纯约化哈密顿量接口与求解器，迁移 Nishino，完成测试后再清理并重跑旧 Nishino 数据。不要同时开始五个大规模任务。

## 14. 一手文献

- [Nishino and Miyashita, Phys. Rev. B 91, 134411 (2015)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.91.134411)
- [Nishino and Miyashita, Erratum, Phys. Rev. B 97, 019904 (2018)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.019904)
- [Bauer et al., J. Phys.: Condens. Matter 23, 394204 (2011)](https://arxiv.org/abs/1010.4730)
- [Hirst et al., Phys. Rev. B 106, 094402 (2022)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.094402)
- [Gomonay et al., npj Spintronics 2, 35 (2024)](https://www.nature.com/articles/s44306-024-00042-3)
- [Laliena et al., Scientific Reports 10, 20430 (2020)](https://www.nature.com/articles/s41598-020-76903-8)
- [Laliena et al., Author Correction, Scientific Reports 12, 2432 (2022)](https://www.nature.com/articles/s41598-022-06147-1)
