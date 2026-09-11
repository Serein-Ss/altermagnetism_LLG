# 面向 Nature 系列与 PRL 的随机 LLG 路径生成研究协议

> 版本：2026-09-11。对象：服务器 AI 执行。本文档定义可审计的研究与投稿证据链，不把计划写成已完成结果。

## 1. 研究命题

研究对象是有限温、无外场磁性体系的条件路径分布
$
p_\theta[\mathbf S_{0:T}|\mathbf S_0,\mathcal H,T,\alpha,G],
$
其中 $\mathbf S_i\in\mathbb S^2$，$\mathcal H$ 是交换、各向异性和邻接，$G$ 是尺寸、边界、子晶格和晶向。模型必须表达进动、阻尼和热噪声造成的路径分支，而不是只拟合平均轨迹或末态。

随机初态和独立平衡初态是两个实验层。非平衡实验从解析、随机、纹理或畴壁初态出发；平衡实验从同温长链抽取多个初态，再对每个精确复制的初态使用独立 Wiener 噪声。训练和测试按初态 ID、噪声复制 ID 和条件块分组，禁止随机切帧造成泄漏。

## 2. 物理模型

统一哈密顿量为
$
H=-\sum_{(i,j)\in E}J_{ij}\mathbf S_i\cdot\mathbf S_j-K\sum_i(S_i^z)^2-\sum_i\mathbf B_{ext}\cdot\mathbf S_i.
$
新研究数据强制 $\mathbf B_{ext}=0$、SOT=0、STT=0、电流=0。文献目录可以保留原论文需要的驱动，但必须与 `zero_field_paths` 分目录和 manifest。有效场 $\mathbf b_i=\sum_jJ_{ij}\mathbf S_j+2KS_i^z\hat z+\mathbf B_{ext}$。每条键只出现一次，并用能量有限差分验证场。

采用 Stratonovich Gilbert LLG
$
d\mathbf S_i=-\frac{\mathbf S_i\times(\mathbf b_i dt+\sqrt{2\alpha\vartheta}d\mathbf W_i)+\alpha\mathbf S_i\times[\mathbf S_i\times(\mathbf b_i dt+\sqrt{2\alpha\vartheta}d\mathbf W_i)]}{1+\alpha^2},
$
其中 $dW_{i\mu}\sim N(0,dt)$。配置必须写出 $\gamma$、磁矩、单位制、温度和时间换算。使用 paper midpoint、Heun、几何 midpoint 做收敛；保存 predictor/raw 未归一化误差，再投影。强收敛比较复用 Brownian bridge，弱收敛才可独立随机。

Gomonay 参数注册为 $J_1=11.1$、$J_2=1.88$、$\tilde J=0.8$ meV，分别使用 $K_{SW}=0$、$K_{DW}=0.047$ meV。100 晶向为二基点，110 为行列式二的四基点，几何取 `scripts/literature/gomonay_2024/model.py`。色散验收使用
$
a=\cos(k_x/2)\cos(k_y/2),\ b=1+K/(2J_1)+(J_2/J_1)(\sin^2(k_x/2)+\sin^2(k_y/2)),
$
$
c=(\tilde J/J_1)\sin k_x\sin k_y,\quad \omega_\pm=4J_1(\sqrt{b^2-a^2}\pm c).
$
必须补齐连续高对称路径、两晶向、$\tilde J=0$ 对照、线性和非线性畴壁。现有小尺寸频点及单一自由壁速度不足以称完整复现。

## 3. 数据设计

每条样本是完整事件：`system_id, model_version, geometry_id, temperature_id, alpha_id, initial_id, noise_replica_id, seed, t_grid, spins, energy, field, metadata`。建议目录为 `data/zero_field_paths/v1/{system}/{geometry}/{temperature}/`，数组与 manifest 分离并保存 SHA-256。

每条件设置三类初态：解析基态/反平行态；随机球面和纹理态；长链间隔抽取的平衡态。随机初态不得按结果筛选。保存初始能量、磁化、结构因子、拓扑和畴标签。平衡池须报告 burn-in、间隔、ACF、分块 ESS；未通过标记为 candidate_equilibrium。

每个初态至少 32 个独立噪声复制，稀有事件至少 128；至少 8 个不同初态。相同初态逐元素哈希相同，噪声 seed 不同。Gomonay 矩阵为 100 晶向 L=16,32,64,128，110 为 L=16,32,64；T/J1={.05,.10,.20,.40,.60}，alpha={.01,.05,.10}。训练只取预注册块，测试整块留出尺寸、温度和材料。

事件定义为序参量越过预先校准阈值并持续驻留时间；零事件只能使用 Kaplan–Meier、RMST 和删失比例报告，不能把观测终点称平均寿命。非平衡窗口和寿命窗口分开定义。

## 4. 认证指标

每条件检查有限值、模长、能量/场有限差分、噪声方差、时间步、独立复制、初态覆盖、能量分布、结构因子、子晶格序参量和边界一致性。均值之外报告 block bootstrap 95% CI、ESS、跨初态和跨噪声方差。模型与参考使用相同初态但独立噪声，指标覆盖 geodesic 逐点误差、短时 MMD/energy distance、ACF/谱/结构因子、首达时间和机制分类。

基线必须包括逐步 LLG、平均场、欧氏 flow matching、无 Hamiltonian features、无 FiLM、无 SO(3) 约束模型。必须报告参数量、函数评估次数、GPU 小时和 wall-clock speedup。

## 5. 生成模型

状态空间与基本几何运算为：

$$
\mathcal M=(\mathbb S^2)^{N_s},\qquad \Pi_s(v)=v-(s\cdot v)s.
$$

$$
\operatorname{Exp}_s(v)=\cos\|v\|s+\sin\|v\|\frac{v}{\|v\|}.
$$

反平行点的 `sphere_log` 要有显式分支并统计频率，不能用 clamp 隐藏奇异性。

测地流匹配使用：

$$
x_\tau=\operatorname{Exp}_{x_0}\!\left(\tau\log_{x_0}x_1\right).
$$

训练损失为：

$$
\mathcal L=\mathbb E\left[\left\|v_\phi(x_\tau,\tau\mid x_0,\mathcal H,T,\alpha,G)-\partial_\tau x_\tau\right\|^2\right].
$$

其中 $\tau$ 是生成运输时间，不是物理时间。当前参考采样把切向量固定为 sigma 半径，可能造成低维支持集，必须加入高斯半径、真实 LLG 增量和端点参考三种消融。

当前 PeriodicConv3d 只支持周期二维双子晶格。若声称尺寸/材料泛化，须实现边界 mask、可变子晶格和显式 bond message passing 或 adapter。输出切向速度，做随机 SO(3)、平移和反演/时间反演测试；各向异性存在时只声称 proper SO(3) 协变。条件至少包括 T、alpha、lag、耦合比、K、边界、晶向和尺寸。

## 6. 实验包与论文证据

* A 物理包：补齐五篇文献的连续曲线、晶向、长时事件、充分弛豫和空间收敛；每篇独立 certificate。
* B 零场包：Gomonay 随机初态、平衡池、同初态多噪声、多初态、多尺寸、多温度、长时和删失统计。
* C 方法包：四层指标、RFM/扩散/神经 SDE/自回归/LLG 基线，几何、条件、参考噪声、长时消融。
* D 泛化包：留出尺寸、温度、初态类型和至少一种材料；多体系需显式材料参数，分别报告插值和外推。
* E 复现包：环境锁定、配置、哈希、Slurm 命令、失败日志、许可、最小公开数据和单脚本主图。

候选创新只有在反事实消融、盲测和预注册指标同时通过后才可使用：路径级物理生成；磁自旋流形协变；同初态复制与条件块留出的数据协议；长时误差和失效边界的量化。Riemannian flow matching、Timewarp 和 TITO 已有流匹配、可迁移和长时分布先例，因此禁止未经检索使用“首次、通用、无偏”。

## 7. 期刊审稿门槛

| 期刊 | 当前欠缺 | 必须补齐 |
|---|---|---|
| Nature | 尚无改变领域认识的发现，复现和跨学科证据未闭合 | 可独立验证的新物理结论、跨体系外推、完整失效边界，最好有实验或独立模拟 |
| Nature Machine Intelligence | 几何 flow matching 和可迁移 MD 已有近邻，架构仍限周期二维 | 方法层新理论或显著效率优势，盲测、强基线、消融和可迁移原则 |
| Nature Computational Science | 长时稳定、尺寸外推、算力收益和开放复核不足 | 复杂计算问题的误差-成本曲线、跨几何/材料 transfer、开放数据 |
| PRL | 目前像方法论文，未形成一个决定性的物理发现 | 聚焦交错磁手性与热路径统计等单一发现，以解析理论、扫描和对照闭合 |

Nature 审稿标准要求技术可靠、强证据、新颖、领域重要和一般科学读者兴趣；PRL 要求显著推进物理并满足至少一项影响标准。NMI 重点是 ML/AI 及其科学影响，NCS 重点是推进科学的计算技术和数学模型。

## 8. 执行链与停止规则

```text
P0 inventory/source audit -> P1 Hamiltonian/noise/integrator certificates
-> P2 literature completion -> P3 zero-field initial/equilibrium audit
-> P4 long-path pilot -> P5 factorial data -> P6 baselines/held-out tests
-> P7 figures/release package
```

证书失败即停止该体系正式训练，保留失败版本并新建 model/data version。阈值在测试集前登记，不能按结果修改。每次运行输出 inventory、manifest、certificate、命令、commit、样本数、重复设计和失败项。

当前 V2 是 330 条、5 K、固定阻尼、带 SOT 脉冲短路径；3 ps 是同一轨迹延续，不是独立样本。R2–R5 报告仅支持部分 Gomonay 频谱/自由壁证据；Bauer 无越零事件，Hirst 壁仍弛豫，Laliena 临界分支差约 2.54%，均未完整认证。图在 `output/literature_reproduction/r2_r5_analysis_20260910/figures/`。

## 9. 服务器 AI 的逐项执行清单

### P0 盘点

读取最新 commit，列出代码、配置、原始/派生文件、环境、GPU、磁盘和 Slurm 队列。为每个 paper_id 生成 `inventory.json`，记录命令、退出码、SHA-256 和 `pass/fail/inconclusive/not_run`。禁止把作业 COMPLETED 当科学通过。

### P1 内核证书

对随机种子运行单自旋、双自旋和小环测试；比较能量有限差分与 `field`；验证噪声方差为 $dt$；同时运行三种积分器并保存未投影误差。强收敛使用共同 Brownian bridge，弱收敛使用独立噪声。若 raw 误差缺失，证书必须为 inconclusive。

### P2 文献补全

Gomonay 先生成 100/110 晶向连续色散与 $\tilde J=0$ 对照，再扫描壁速度和晶向。Bauer 按原参数延长到足够事件数，采用右删失；Hirst 先延长零温壁弛豫并解决 AFMR 可辨识性；Laliena 同时控制空间步长和域长并解释 2.54% 临界差。Nishino 保留历史 fail 和用户接受记录，新增独立平稳性检验。

### P3–P5 零场数据

先冻结初态池和条件矩阵，再生成复制。每个 shard 写入 seed、初态哈希和噪声哈希；生成后随机抽样逐元素重放。平衡池用 ACF、ESS、能量直方图和跨链 KS/JSD 检查。长时数据不得通过升温、降低能垒或外场加速事件。

### P6–P7 训练、测试、发布

在测试块冻结后训练五个模型 seed。测试报告必须分解初态内、初态间、条件间误差，并与所有基线同算力比较。发布包包括环境锁定、配置、代码 commit、数据许可证、最小可下载 shard、图脚本、失败日志和 `CITATION.cff`。任何论文数字必须可由一条 manifest 命令重建。

## 10. 主文图表规划

| 图 | 内容 | 必需证据 |
|---|---|---|
| Fig.1 | 问题定义、球面路径、条件和数据层 | 物理量与生成 $\tau$/时间区分 |
| Fig.2 | Gomonay 文献色散、晶向、零 $\tilde J$ 对照 | 连续曲线、误差、解析式 |
| Fig.3 | 同初态多噪声路径云和初态间分布 | 分层 CI、无伪重复 |
| Fig.4 | 尺寸/温度盲测及误差-成本曲线 | 完整矩阵和基线 |
| Fig.5 | 长时能量、序参量、事件率和删失 | KM/RMST、失败边界 |
| Fig.6 | 几何/条件/参考噪声消融 | 反事实方法证据 |
| Extended | 五文献复现、所有 seed、失败实例和敏感性 | 全量可审计 |

## 11. 不可接受的结论替换

“同一体系测试所以意义不明确”通过按条件块和初态块留出解决；但若只在训练体系内测试，只能称内分布验证。随机初态不能自动代表平衡，平衡初态也不能覆盖非平衡纹理；两者必须分别标注。没有同初态独立噪声，无法估计条件随机性。没有长时事件，不能声称寿命或稀有事件。没有外部独立曲线，不能称文献完全复现。归一化后的模长为一，不能替代积分器误差证书。

## 12. 进度门槛

* Gate 1：所有内核测试通过，噪声和单位可追溯。
* Gate 2：至少一个体系的文献连续曲线和时间/空间收敛通过。
* Gate 3：零场数据满足复制、初态、尺寸、温度和长时要求。
* Gate 4：盲测优于或解释性地匹配基线，几何与条件消融闭合。
* Gate 5：所有主图可重建，审查报告中的 Blocking concerns 均有 resolution test。

Gate 3 之前不得进行正式创新结论训练；Gate 4 之前不得写“验证创新点”；Gate 5 之前不得投稿。

## 13. DMI 与阻挫体系扩展包

DMI 和阻挫不直接混入第一阶段 Gomonay 基线。它们作为第二阶段挑战集，用来检验模型对手性、多稳态、长相关时间和稀有跃迁的处理能力。每新增一种相互作用都必须创建新的 `model_version`，重新做能量、有效场、积分器和文献基准认证。

### 13.1 Dzyaloshinskii–Moriya 相互作用

键型 DMI 写为

$$
H_{DMI}=\sum_{(i,j)\in E}\mathbf D_{ij}\cdot(\mathbf S_i\times\mathbf S_j).
$$

有效场必须由完整哈密顿量求导得到：

$$
\mathbf b_i^{DMI}=-\frac{\partial H_{DMI}}{\partial\mathbf S_i}.
$$

每条有向键必须记录 \(\mathbf D_{ij}\)，并注明界面型或体型 DMI、晶体对称性、键方向和边界条件。不得只增加一个标量 D 而省略 DMI 向量。验收包括螺旋波数、左/右手性比例、非互易色散

$$
\omega(\mathbf k)\neq\omega(-\mathbf k),
$$

以及畴壁手性、边界倾斜和 D→0 连续性。至少使用一维链和二维薄膜两个几何，周期边界与开放边界分别测试。

### 13.2 阻挫交换

阻挫来自不能同时满足全部键的竞争交换，例如

$$
H_{J_1J_2}=-J_1\sum_{\langle i,j\rangle}\mathbf S_i\cdot\mathbf S_j
-J_2\sum_{\langle\langle i,j\rangle\rangle}\mathbf S_i\cdot\mathbf S_j.
$$

建议从三角晶格反铁磁、Kagome 晶格和方格 \(J_1-J_2\) 模型开始。必须扫描耦合比并记录基态能量、结构因子、简并度、相关长度和自相关时间。不能只用最终能量最低的一条轨迹作为“平衡态”；阻挫体系需要多起点、多链和独立初态池，并报告未混合链的比例。

### 13.3 DMI 与阻挫联合挑战

四类数据集按相同协议生成：

| 数据集 | 目的 | 最低验收 |
|---|---|---|
| 无 DMI、无阻挫 | 基础正确性 | 能量、色散、稳态和积分器证书 |
| 只有 DMI | 手性和非互易动力学 | 螺旋波数、\(\omega(k)-\omega(-k)\)、畴壁手性 |
| 只有阻挫 | 多稳态和慢弛豫 | 结构因子、相图、ACF/ESS、多初态覆盖 |
| DMI+阻挫 | 高难度迁移 | 多稳态转移矩阵、稀有事件和长时稳定性 |

这四类必须使用相同的训练、验证和测试规则。测试集应留出未见的 D/J 比、晶格尺寸、边界和初态类型；DMI 符号反转与交换符号反转应作为物理对照，而不是数据增强后混入同一事件。

### 13.4 生成模型接口扩展

条件输入从 \((T,\alpha,\Delta t,K,G)\) 扩展为键级张量

$$
\{J_{ij},\mathbf D_{ij},K_i,\mathrm{bond\ type}_{ij},\mathrm{lattice\ graph}\}.
$$

PeriodicConv3d 不能表达开放边界、Kagome 或可变配位数。正式挑战集必须加入显式 bond message passing 或图/网格混合编码器，输出仍需满足每个自旋的切向约束。模型测试包括：旋转所有自旋与 \(\mathbf D\) 后的协变性；同时反转自旋和 DMI 后的时间反演关系；删去 DMI 或设 \(J_2=0\) 时退化到已认证基线。

### 13.5 研究问题和发表价值

DMI 集检验模型能否学习手性和方向不对称；阻挫集检验模型能否表达多稳态、长记忆和初态依赖；联合集检验模型在复杂能量景观中的路径分布和稀有跃迁能力。只有当模型在这些挑战集上相对欧氏 flow、无图结构模型和逐步 LLG 基线显示可重复优势时，才可将“复杂相互作用下的物理生成”列为核心创新。DMI 或阻挫本身不是创新，创新必须来自可证伪的模型能力或新的物理结论。

执行顺序固定为：先完成无 DMI/无阻挫基线，再单独加入 DMI，再单独加入阻挫，最后运行联合挑战。任何阶段失败都保留失败数据，不得通过改变耦合、温度或外场来制造事件。

三份互盲预审报告位于 `D:/WORKSPACE/CodePlace/publication_review_20260911/R1.md`、`R2.md`、`R3.md`。它们共同指出文献闭环、参考半径支持集、真实复制/盲测、近邻基线、长时统计和算力证据是当前阻塞项。实际训练入口为 `scripts/training/train.py`、`scripts/inference/sample.py`。

## 13. V2 温度标签、剧烈轨迹与 RuO2 对照

V2 数据卡中的 `temperature_K=5.0` 是物理温度 5 K，不是约化温度 5。生成器同时使用 0.70、0.78、0.80 T 的阻尼型 SOT 脉冲，持续 0.5 ps；因此剧烈翻转和路径分叉主要由强外部驱动与脉冲后的进动/阻尼造成，不能解释为 5 K 热噪声本身。V2 是来自 Gomonay 论文参数化的二维周期双子晶格 d-wave 交错磁模型，不能直接称为完整 RuO2 原子模型。

RuO2 是类比体系而非当前代码已严格等同的材料。已发表工作通常报道其反铁磁有序高于室温，常见实验/综述范围约 300 K 以上；近期 RuO2 薄膜热成像工作报告约 370 K。不同薄膜制备、氧空位和测量方法会造成差异，也存在对部分样品磁序的争议。因此文档和论文必须写明来源、样品和不确定区间，不能把单一 370 K 当作普适常数。

后续 RuO2 研究应先注册材料参数、晶格常数、磁矩、交换表和各向异性，再通过本模型自己的热平衡扫描测定有限尺寸 $T_N$。温度用 $T/T_N$ 分层，至少覆盖 0.1、0.3、0.5、0.7、0.9、1.0、1.1 和 1.3；$T_N$ 附近增加路径长度、初态数和噪声复制。接近 $T_N$ 时局部涨落通常增强，但同时可能出现临界减速，不能简单假设“变化越剧烈就越快达到稳态”。

V2 的带驱动结果只能用于早期模型调试和驱动路径基准。正式零场训练集必须把外场、SOT、STT 和电流字段置零，并单独报告温度、约化温度、驱动和 $T/T_N$，以免把驱动效应误判为温度效应。

### 参考

- https://www.nature.com/nature/for-referees/policies-and-processes
- https://www.nature.com/natmachintell/submission-guidelines/about/aims
- https://www.nature.com/natcomputsci/natcomputsci/natcomputsci/about/aims
- https://journals.aps.org/prl/about
- https://arxiv.org/abs/2302.03660
- https://arxiv.org/abs/2302.01170
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13060594/
- https://www.nature.com/articles/s44306-024-00042-3

