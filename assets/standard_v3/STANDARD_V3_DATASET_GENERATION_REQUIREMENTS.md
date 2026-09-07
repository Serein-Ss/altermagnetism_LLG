# Standard V3 有限温 LLG 路径数据集生成需求

> 状态：服务器生产前规范（protocol specification）
> 日期：2026-09-08
> 适用目录：`altermagnetism_LLG/data/standard_v3/`
> 核心决定：正式训练数据采用统一固定物理时长；先用变长先导模拟确定该时长。不得覆盖或改写 `data/standard_v2/`。

## 1. 数据集目标与允许的科学主张

Standard V3 的目标是为同一种物理约束生成方法提供多个独立的文献体系数据集，验证其能否学习

\[
p[\mathbf S(0:T)\mid \mathbf S_0,\,c,\,\mathcal H]
\]

而不是只拟合平均轨迹。最终论文的主体是 Gomonay *et al.* 的 d-wave 交错磁模型；普通铁磁、常规反铁磁和非共线磁体是方法普适性基准。

第一阶段的“通用性”定义为：对每个体系采用相同的路径生成思想、流匹配目标、训练流程和评估指标，但允许使用与该体系维度和邻接关系相匹配的空间后端，并分别训练 checkpoint。一个 checkpoint 零样本覆盖 1D/2D/3D、开放/周期边界及不同子晶格数属于更强的后续目标，不作为 V3 的硬性要求。

V3 数据集只有通过本文第 12 节的全部门槛后，才能标记为 `production_certified`。仅生成成功、文件可读或自旋归一化正确，均不足以获得该状态。

## 2. 固定时长方案

### 2.1 正式轨迹

同一“体系—尺寸—温度—驱动协议”组内的所有正式轨迹必须具有完全相同的物理时间网格：

\[
t=0,\Delta t_{save},\ldots,T_{fixed}.
\]

所有轨迹无论何时进入稳定磁盆，都继续积分到 `T_fixed`。快速到稳路径因此包含更长的稳态尾段，慢速路径包含更短尾段；模型仍可从完整轨迹中学习不同的到稳时间。

### 2.2 如何确定 `T_fixed`

正式生成前，每个体系至少在最慢、近阈值和最大训练尺寸条件下运行 128–200 条变长先导轨迹。对每条先导轨迹检测 `t_steady`，并取

\[
T_{fixed}\ge Q_{99\%\text{--}99.5\%}(t_{steady})+T_{tail}.
\]

`T_tail` 必须不短于稳定磁盆内能量、主序参量和空间结构量的最大积分自相关时间的 5 倍；推荐使用 10 倍。`T_fixed` 应向上取整为便于 HDF5 分块和训练的保存间隔整数倍。

不能预先认定 1、2、4 或 8 ps 一定足够。交错磁候选扫描使用 `1/2/4/8 ps`；其他体系使用各自文献时间单位中的几何递增序列。最终值写入冻结后的 `protocol.yaml`，在生产任务开始后不得修改。

### 2.3 未决轨迹

有限温随机过程不能在有限时间内保证 100% 到稳。若轨迹在 `T_fixed` 时仍未满足稳态条件，必须保留并记录：

```text
event_observed = 0
outcome = unresolved_censored
censor_time = T_fixed
sample_weight = 1
```

不得删除、替换随机种子或重复模拟直到其成功。`unresolved_censored` 比例原则上应低于 1–5%；超过 5% 时应重新延长 `T_fixed`，而不是修改标签。

## 3. 有限温稳态的统一判据

“稳态”指指定观测时间尺度上的统计稳态或亚稳磁盆，不指单条能量曲线变成常数。热噪声在稳态中继续存在。

### 3.1 平衡磁盆校准

每个体系、温度和尺寸首先运行无驱动平衡轨迹，得到正/负磁盆或其他稳定相中主序参量的分布。磁盆阈值必须由该分布确定，不能统一硬编码为 `|m_z|=0.8` 或 `|n_z|=0.8`。

建议用平衡磁盆分布的 1%–5% 尾部分位数确定保守边界，并将阈值及其置信区间写入认证文件。

### 3.2 单条轨迹到稳检测

轨迹只有同时满足以下条件，才记录 `event_observed=1` 和 `steady_time`：

1. 主序参量进入某个预先校准的稳定磁盆；
2. 后续连续窗口内不再跨出该磁盆；
3. 能量、主序参量以及结构因子峰或畴壁密度的分块均值无显著漂移；
4. 上述量的时间斜率在考虑自相关后的 95% 置信区间内包含零；
5. 至少连续三个窗口满足条件；
6. 在额外的 `T_tail` 内没有重新越过磁盆边界。

窗口长度不得短于相应量的 5 倍积分自相关时间。统计误差必须用分块均值或自相关修正后的有效样本数计算，不能把相邻保存帧当作独立样本。

### 3.3 统一结果标签

```text
no_crossing
crossed_returned
committed_switch
other_committed_state
unresolved_censored
```

`negative_endpoint` 只能作为几何描述符，不能再等同于稳定翻转。每条轨迹另存 `first_crossing_time`、`steady_time`、`steady_state_id` 和 `event_observed`。

## 4. 文献体系和参数状态

参数分为三种状态：

- **文献冻结值**：直接来自论文公式或参数表，不得为改善结果而调整；
- **数值候选值**：论文未给出或当前代码自行选择，必须做收敛扫描；
- **生产冻结值**：先导扫描通过后写入 `protocol.yaml` 的最终值。

### 4.1 Nishino–Miyashita 普通磁矩热平衡基准

来源：[Phys. Rev. B 91, 134411 (2015)](https://doi.org/10.1103/PhysRevB.91.134411)，勘误：[Phys. Rev. B 97, 019904 (2018)](https://doi.org/10.1103/PhysRevB.97.019904)。

| 参数 | 数值 | 状态 |
|---|---:|---|
| 外场 `H_z` | 2.0 | 文献/当前实现的无量纲 Fig. 1 基准 |
| 磁矩 `M` | 1.0 | 文献冻结值 |
| `gamma` | 1.0 | 文献无量纲约定 |
| `k_B` | 1.0 | 文献无量纲约定 |
| 温度 | 1.0、2.0、5.0 | 当前已验证条件 |
| 子晶格数 | 1 | 冻结 |
| 相互作用 | 无；独立磁矩 | 冻结 |
| 解析目标 | `M_z=M[coth(HM/kBT)-kBT/(HM)]` | 必须复现 |

当前代码数值起点为 `alpha=0.05`、`dt=0.005`、准备 10000 步、保存轨迹 2000 步、每 40 步保存。这些属于当前数值选择，不是全部文献冻结参数。V3 必须补做 `dt`、准备长度和保存间隔收敛；该数据主要用于热浴认证，不负责证明空间路径机制。

### 4.2 Bauer 2011 开放铁磁链

来源：[J. Phys.: Condens. Matter 23, 394204 (2011)](https://doi.org/10.1088/0953-8984/23/39/394204)。

| 参数 | 数值 | 状态 |
|---|---:|---|
| 最近邻交换 `J` | 1.0 | 文献无量纲能量单位 |
| 单轴各向异性 `K/J` | 0.1 | Fig. 2 主要条件；另验证 0.01 |
| 链长 `L` | 100 | Fig. 2 文献条件 |
| `k_B T/J` | 0.11 | Fig. 2 文献条件 |
| 阻尼 `lambda` | 0.1 | 文献条件 |
| 总时间 | `750000 hbar/J` | Fig. 2 文献时间窗 |
| 边界 | 开放边界 | 冻结，不得改成周期边界冒充复现 |
| 子晶格数 | 1 | 冻结 |
| 当前数值 `dt` | 0.02 | 论文未报告，属于候选值 |
| 当前保存间隔 | `250*dt=5` | 候选值 |

必须先复现连续模型畴壁能垒、边缘核化、成功/失败翻转尝试和寿命随链长的趋势。论文没有报告随机种子和积分时间步，因此 `dt` 必须至少与减半值比较。加速条件 `L=40`、`kBT/J=0.20`、总时间 500 只能作为软件/机制先导，不能标记为 Fig. 2 定量复现。

### 4.3 CrNb3S6 非共线单轴手性磁体

来源：[Scientific Reports 10, 20430 (2020)](https://doi.org/10.1038/s41598-020-76903-8)。

> 重要元数据修正：当前 `scripts/core/unified_llg.py` 写成了 `10.1038/s41598-020-76989-2`，V3 参数注册表必须改为 `10.1038/s41598-020-76903-8`；修改前不能发布 V3 清单。

| 参数 | 数值 | 状态 |
|---|---:|---|
| 交换刚度 `A` | `1.42e-12 J/m` | 文献冻结值 |
| 单轴 DMI `D` | `369e-6 J/m^2` | 文献冻结值 |
| 各向异性 `K` | `-124e3 J/m^3` | 文献冻结值 |
| 饱和磁化 `M_s` | `129e3 A/m` | 文献冻结值 |
| 晶格/离散长度起点 | `0.6 nm` | 文献材料尺度；仍需离散收敛 |
| `gamma` | `1.76085963023e11 rad/(s*T)` | 当前统一求解器常数 |
| Gilbert 阻尼 `alpha` | 0.01 | 论文动力学参数 |
| 非绝热系数 `beta` | 0.02 | 论文电流驱动条件；仅在实现相应 STT 时使用 |
| 极化率 `P` | 1 | 论文电流驱动条件 |
| 零场螺旋周期 | 48 nm | 必须复现 |
| 横向临界场 | 约 230 mT | 必须复现或给出离散误差 |
| 轴向临界场 | 约 2.3 T | 文献检查量 |
| 边界 | 手性轴周期边界 | 主训练集冻结 |

当前零温验证脚本使用 `length=80`、`cell=0.6 nm`、一周 winding、`dt=1 fs` 和 `alpha=0.05`。其中 `alpha=0.05` 是加速弛豫的数值选择，不能替代生产有限温数据中的论文值 0.01。V3 必须分别保存螺旋波矢、手性、winding、结构因子峰和稳态磁化。

### 4.4 Mn2Au 常规反铁磁基准

来源：[Phys. Rev. B 106, 094402 (2022)](https://doi.org/10.1103/PhysRevB.106.094402)，预印本：[arXiv:2206.08625](https://arxiv.org/abs/2206.08625)。

该体系用于替代 `J_tilde=0` 的数值消融作为“文献常规 AFM”基准。正式实现前必须从论文 Table I、Hamiltonian 和补充材料逐项录入并双人或双脚本核对：

```text
J1-J4 及其精确邻接壳层
两种各向异性常数及轴向约定
Mn 原子磁矩
晶格常数与子晶格位置
Gilbert 阻尼和热噪声约定
论文各验证任务的温度、尺寸、脉冲和边界条件
```

这些值当前没有进入本仓库的 `unified_llg.py`，因此本规范不填写未经全文表格核验的数字。服务器端只有在生成 `mn2au_parameter_audit.json`，其中每个值都带论文页码/表号、单位转换和邻接计数测试后，才允许生成 Mn2Au 正式轨迹。

最低文献复现量为：有限温平衡磁化/磁化率、阻尼振荡、热脉冲响应和热梯度畴壁运动中至少两个定量结果。若实现维度或边界与论文不同，只能标记为派生模型，不能标记为原文复现。

### 4.5 Gomonay 2024 d-wave 交错磁主数据集

来源：[npj Spintronics 2, 35 (2024)](https://doi.org/10.1038/s44306-024-00042-3)，离散哈密顿量对应补充材料 Eq. (S.1)、Fig. S1 和 Table I 的 domain-wall 参数集。

| 参数 | 数值 | 状态 |
|---|---:|---|
| 层间 AFM 交换 `J1` | 11.1 meV | 文献冻结值 |
| 层内 FM 交换 `J2` | 1.88 meV | 文献冻结值 |
| 交替对角交换 `J_tilde` | 0.8 meV | 文献冻结值 |
| 单轴各向异性 `K` | 0.047 meV | 文献冻结值 |
| 原子磁矩 `mu_s` | `1 mu_B` | 文献冻结值 |
| 晶格常数 `a0` | 0.448 nm | 文献冻结值 |
| `gamma` | `1.76085963023e11 rad/(s*T)` | 当前统一求解器常数 |
| 子晶格数 | 2 | 冻结 |
| 边界 | x/y 周期边界 | 主数据集冻结 |
| 温度锚点 | 5 K | V2/V3 首个生产切片；不是论文材料预测 |
| `alpha` | 0.01 | 当前协议选择，不是上述表格的材料定值 |
| 脉冲宽度 | 0.5 ps | 当前协议选择 |
| SOT field-like 分量 | 0 T | 当前协议选择 |
| SOT 极化起点 | `[1/sqrt(2),1/sqrt(2),0]` | 当前 `[110]` 协议 |

V2 的 0.70、0.78、0.80 T 是数值阈值扫描，不是论文给出的器件电流或材料驱动参数。V3 必须先用它们及两侧包围点做先导扫描，再按观察到的 `P(committed_switch)` 选择约对应 0.05、0.25、0.50、0.75、0.95 的五个冻结驱动点。筛选完成后不得根据正式样本结果再次移动驱动值。

晶向扩展至少包括 `[100]`、`[010]`、`[110]` 和 `[1-10]`。相关旋转和子晶格交换后的路径分布必须作为对称性测试；晶向不能只存成字符串，必须保存归一化 SOT 向量和 `crystal_x/y/z`。

训练尺寸候选为 L32、L48、L64；L16 仅作相干动力学辅助。L96 和 L128 全部作为严格尺寸外测试。只有当 L48/L64 中确实出现并收敛了空间非均匀机制，才允许研究“小体系训练到大体系泛化”。

## 5. 初态准备标准

每个体系的驱动轨迹必须从对应温度下的平稳初态池抽样，而不是每条都从完全相同的理想共线态开始。

1. 从已知零温稳定态开始，关闭驱动，在目标温度运行多条独立准备链；
2. 同时监测总能量、能量分量、主序参量、结构因子和畴壁/缺陷密度；
3. 计算最大积分自相关时间 `tau_int`；
4. burn-in 至少 `20 tau_int`，推荐 `50 tau_int`；
5. 初态抽样间隔至少 `2 tau_int`，推荐 `5 tau_int`，或使用完全独立准备链；
6. 多链 `R-hat < 1.05`，并记录有效样本数；
7. 对预先制备的正磁盆条件，可从正磁盆平衡分布抽样；这是初态条件，不是对驱动结果的筛选；
8. 同一 `initial_state_id` 派生的不同驱动轨迹必须位于同一 train/validation/test 分区。

## 6. 数值收敛扫描

### 6.1 Gomonay 交错磁

| 项目 | 候选扫描 | 当前起点 | 通过规则 |
|---|---|---|---|
| 积分步长 | 0.10、0.05、0.025 fs | 0.05 fs | 0.05 与 0.025 fs 的统计量在 95% CI 内且预设误差达标 |
| 完整自旋保存间隔 | 2.5、5、10 fs | 10 fs | 机制标签不变，首达时间误差不超过一个粗时间间隔 |
| 总时间 | 1、2、4、8 ps | 1 ps | 延长一倍后 committed 比例变化 <=3%，机制比例变化 <=5%，未决比例 <=5% |
| 尺寸 | L32、L48、L64、L96、L128 | L16/32/64/96 | 局域机制与每面积统计量达到晚尺寸稳定；L96/L128保留测试 |

完整自旋可每 10 fs 保存；能量、Néel序参量和畴壁密度建议每 1–2 fs 保存，以较小存储代价完成稳态诊断。必须用同一细时间步轨迹的下采样评估保存间隔，避免把不同噪声实现误判为采样误差。

### 6.2 其他体系

采用相同的“当前时间步、减半时间步、再次减半时间步”原则，但在各自论文单位中设置。保存间隔至少比较当前值和减半值，总时间至少比较 `T` 与 `2T`。Bauer 的 `dt=0.02`、CrNb3S6 的 `dt=1 fs` 均只是当前数值起点。

主要统计量至少包括：终态磁盆概率、未决比例、首达/到稳时间分布、能量和主序参量均值曲线、空间机制比例、结构因子以及文献特有可观测量。

## 7. 正式条件和样本量

### 7.1 条件设计

每个体系必须同时包含：

- 论文原条件 `literature_reference`；
- 无驱动平衡条件；
- 不翻转、近阈值和高成功率条件；
- 至少一个完整留出的驱动或温度；
- 至少一个严格尺寸外测试；
- 若体系具有晶向/手性，对应的对称相关条件和物理不等价条件。

不要在第一批中对温度、驱动、尺寸和晶向做完整笛卡尔积。先固定一个文献锚点，分别建立驱动切片、温度切片、晶向切片和尺寸切片，待模型可行后再补交互条件。

### 7.2 数量

- 收敛先导：每个关键条件 128–200 条；
- 普通 FM、AFM、非共线正式基准：每个冻结条件至少 500 条；
- 交错磁近阈值正式条件：每个冻结条件推荐 1000 条；
- 低于 1% 的稀有机制：不能依赖 500–1000 条普通样本做精确估计，应增加到万级无偏样本，或建立单独且带统计权重的稀有事件数据集。

正式无偏数据中所有轨迹 `sample_weight=1`，禁止按结果平衡。最终数量应根据最稀有目标类别的二项置信区间决定，而不是只根据模型 loss 决定。

## 8. 8:1:1 划分与外推测试

普通训练条件按完整初态组进行 8:1:1 划分。例如每条件 1000 条为 800/100/100。不得跨分区重复：

```text
initial_state_id
preparation chain segment
thermal RNG stream
trajectory seed
同一轨迹的时间窗口
周期平移或旋转增强版本
```

以下数据不执行随机 8:1:1，而全部标为测试：

- L96/L128 等尺寸外集合；
- 完整留出的温度、驱动和晶向；
- 未参与训练的新 Hamiltonian；
- 比 `T_fixed` 更长的长期验证轨迹。

## 9. HDF5 Schema V3

不同体系和不同空间形状分别保存文件或分片，不把不兼容的稠密张量强行合并。每个分片建议包含 50–100 条轨迹，写入 `.partial` 后原子重命名。

### 9.1 必需轨迹数组

```text
time                       [frame]
spins                      [path,frame,sublattice,x,y,(z),xyz]
initial_spins              [path,sublattice,x,y,(z),xyz]
energy_total               [path,frame]
energy_components          [path,frame,component]
magnetization              [path,frame,3]
neel                       [path,frame,3]        # 适用时
structure_factor_summary   [path,frame,...]
wall_or_defect_density     [path,frame]
drive_waveform             [path,frame,3]
```

### 9.2 必需轨迹元数据

```text
system_id
hamiltonian_id
condition_id
trajectory_id
initial_state_id
preparation_seed
dynamics_seed
split
sample_weight
first_crossing_time
steady_time
steady_state_id
event_observed
censor_time
outcome_label
```

### 9.3 必需物理条件

```text
temperature
alpha
gamma
moment_or_Ms
integration_dt
saved_dt
fixed_duration
pulse_duration
field_like_amplitude
damping_like_amplitude
polarization_vector
crystal_frame
lattice_shape
lattice_constant_or_cell_size
boundary_condition
number_of_sublattices
```

### 9.4 Hamiltonian/几何描述

```text
neighbor_template / edge_index
bond_type
bond_vector
coupling_by_bond_type
sublattice_id
site_mask
periodic_axes
parameter_names
parameter_values
parameter_units
```

不存在的 DMI、SOT 或 bond 类型应通过显式掩码表示，不得伪造非零向量。驱动必须保存逐帧波形，不能只保存“振幅+脉冲宽度”。

正式物理主文件使用 `float32` 保存自旋，能量建议 `float64`。可以额外制作 float16/int16 训练缓存，但必须保留 float32 规范源文件并记录量化误差。HDF5 使用空间/时间 chunk、shuffle 和 gzip/lzf，训练时惰性读取。

## 10. 随机场与随机数

所有有限温生产轨迹使用各向同性、零均值、格点/时间/笛卡尔分量独立的热场，并与方程中的磁矩和旋磁比保持同一单位约定：

\[
\langle B_{\mu i}(t)B_{\nu j}(t')\rangle=
\frac{2\alpha k_BT}{\gamma\mu_s}
\delta_{\mu\nu}\delta_{ij}\delta(t-t').
\]

离散步长 `dt` 下每个分量标准差为

\[
\sigma_B=\sqrt{\frac{2\alpha k_BT}{\gamma\mu_s\,dt}}.
\]

必须采用随机 Heun/Stratonovich 约定。三维热场始终同时启用；单方向随机脉冲只能作为诊断，不能作为生产热浴。

每条轨迹保存独立、可重放的准备种子和动力学种子。种子命名空间由 `base_seed + system + size + condition + trajectory_id` 的加密哈希派生，并在整个 V3 套件上检查唯一性。

## 11. 模型接口要求

正式轨迹固定时长，因此模型仍可使用稠密张量，无需把未来 `steady_time` 作为输入。模型输入包含初态、物理/无量纲条件、逐帧时间和驱动波形；输出为完整固定时长自旋路径。

对生成路径使用与真实 LLG 完全相同的离线稳态检测器，即可得到生成的 `steady_time`、最终磁盆和未决比例。必须比较完整分布，而不是只比较平均轨迹。

不同体系具有不同自然时间和能量尺度。原始文件保留真实单位，模型同时读取：

```text
dimensionless_time
kBT/J_ref
drive/J_ref_or_exchange_field
K/J_ref
D/J_ref_or_dimensionless_DMI
alpha
```

不得把每条轨迹按自己的到稳时间重缩放到 `[0,1]`，否则模型无法学习真实到稳时间。

## 12. Production certification 门槛

每个体系依次获得以下状态：

```text
literature_parameters_audited
literature_observable_reproduced
equilibrium_certified
dt_certified
save_cadence_certified
duration_certified
size_certified
production_certified
```

最终门槛：

- 解析有效场与能量有限差分一致；
- 文献特征量在预注册容差内复现；
- 零温无驱动阻尼动力学不增能；
- 热浴平衡分布通过 Nishino/Langevin 基准；
- 初态池的多链 `R-hat < 1.05`，自相关和有效样本数已记录；
- 时间步、保存间隔、总时长和尺寸收敛通过；
- 自旋长度最大误差 `<1e-6`，无 NaN/Inf/损坏帧；
- 稳态尾段统计平稳，未决轨迹透明保留；
- 训练、验证和各种外推测试无数据泄漏；
- 每个文件具有 SHA-256、字节数、代码 commit、环境版本和参数来源；
- 每个路径类别至少人工检查若干完整空间动画；
- 生成器和认证器分别实现，生成器不得自行宣布收敛。

## 13. 服务器执行顺序

```text
00_parameter_audit
01_hamiltonian_field_tests
02_literature_reproduction
03_equilibrium_initial_pool
04_dt_convergence
05_save_cadence_convergence
06_variable_duration_pilot
07_freeze_T_fixed_and_protocol
08_size_convergence
09_production_shards
10_merge_hash_and_split_audit
11_stationarity_and_mechanism_certification
12_visualization_and_manual_QA
```

生产任务开始前应生成不可变的 `registry.yaml` 和各体系 `protocol.yaml`。生产任务按 50–100 条轨迹分片；失败时只重跑未完成分片。协议冻结后若任何物理或数值参数改变，必须提升数据版本或建立新条件 ID，不能静默覆盖。

## 14. 建议目录

```text
data/standard_v3/
├── registry.yaml
├── validation_thermal/
│   └── nishino2015/
├── fm_bauer2011/
│   ├── literature_reference/
│   ├── equilibrium_pool/
│   ├── convergence/
│   ├── train/
│   └── test_long/
├── afm_mn2au2022/
├── noncollinear_crnb3s6_2020/
└── altermagnet_gomonay2024/
    ├── literature_reference/
    ├── equilibrium_pool/
    ├── convergence/
    ├── train_small/
    ├── validation/
    ├── test_iid/
    ├── test_condition/
    ├── test_size_L96/
    ├── test_size_L128/
    └── test_long/
```

## 15. V2 与 V3 的边界

`standard_v2` 的 330 条 1 ps、5 K、0.70/0.78/0.80 T 轨迹继续保留，用于代码回归和第一阶段模型可行性，不重新命名为 V3。其状态仍是 `ready_for_phase1_model_training`，不是 `production_certified`。

V3 只有在先导模拟确定 `T_fixed`、初态平衡和全部收敛门槛通过后才正式生成。最优先的服务器工作不是立即扩大轨迹数，而是完成参数审计、初态池、变长到稳先导以及固定时长冻结。
