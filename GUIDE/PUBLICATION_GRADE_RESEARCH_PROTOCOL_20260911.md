# 物理约束随机 LLG 路径生成：研究与验收协议

**版本**：2026-09-12，修订2；保留原文件名以维持链接。
**对象**：服务器研究与开发执行者。
**状态**：研究设计，不代表实验完成，不解锁生产，不自动提交计算任务。
**修订依据**：远端 `b8efdb64a3fd97a1dd0943bd216a01adba5eff9f` 的协议、模型和训练实现审查；本次没有重跑物理实验。

## 一、范围、优先级与研究目标

本协议负责零场路径数据、生成模型与统计验收。`STRICT_LITERATURE_REPRODUCTION_PLAN.md` 负责五篇文献的参数、特殊方程、原条件及逐项复现；本协议不豁免其失败项。旧计划所称“唯一执行依据”限定于文献复现。本文件替代自身2026-09-11版本的公式、阶段、样本数和通用验收描述。发现冲突时记录并暂停依赖它的生产步骤，不自行选择较宽松标准。

第一阶段研究问题是：给定完整初态、一个已认证的Gomonay型Hamiltonian、温度、阻尼和几何，能否学习固定物理窗口内的随机LLG条件路径分布，并在明确误差容差下取得计算收益？

\[
p_\phi[\mathbf S(t),0\le t\le t_{end}\mid\mathbf S_{init},\mathcal H,T_{bath},\alpha,G].
\]

时间终点、浴温和模型参数使用不同符号。第一阶段必做未见初态测试，再预先选择尺寸或温度留出之一；未见噪声种子本身不是物理条件泛化。多材料、DMI、阻挫、多lag和长期事件率是后续独立扩展，不作为首次生产同时必须覆盖的矩阵。

球面约束、协变性和可审计数据是质量保障，不能单独替代方法创新或新物理发现。价值必须由分布误差、成本及失效范围证明；未完成最邻近方法比较前，不使用“首次、通用、无偏”。五体系整体结论仍要求五套证书；未使用体系可独立继续复现，不能用其成功替代当前模型证据。

## 二、物理方程与参数合同

### 2.1 单位与有符号键

YAML保留 `source_parameters / reduction / reduced / numerics` 四层，实际单位只在输入换算和后处理使用。对本阶段相同磁矩和旋磁比的单位自旋：

\[
h=H/E_0,\quad \tilde t=t/t_0,\quad t_0=\mu_{ref}/(\gamma E_0),\quad
\vartheta=k_BT_{bath}/E_0,\quad \mathbf b_i=-\partial h/\partial\mathbf S_i.
\]

`mu_ref`是磁矩，不是真空磁导率。SI外场的Zeeman项为 `-mu_ref*B_ext dot S`，约化场为 `beta=mu_ref*B_ext/E0`。不同位点磁矩/旋磁比需另行推导，不能直接复用该形式。

无向交换键只计数一次：

\[
h=-\sum_{(i,j)\in E}j_{ij}\mathbf S_i\cdot\mathbf S_j
-\sum_i\kappa_i(\mathbf S_i\cdot\mathbf e_i)^2-\sum_i\boldsymbol\beta_i\cdot\mathbf S_i,
\]
\[
\mathbf b_i=\sum_jj_{ij}\mathbf S_j+2\kappa_i(\mathbf S_i\cdot\mathbf e_i)\mathbf e_i+\boldsymbol\beta_i.
\]

正键权对应FM，负键权对应AFM。文献幅值不等于有符号键权。必须登记基点、位移、端点、计数规则和符号，检查配位、基态、边界及能量负梯度。

Gomonay幅值为 `J1=11.1,J2=1.88,J_tilde=0.8 meV`，取 `E0=11.1 meV`。当前100晶胞的跨子晶格四条键为 `-J1/E0`，同子晶格轴向键为 `+J2/E0`；子晶格符号sigma为+1/-1，对角 `(1,-1)` 权重 `-sigma*J_tilde/E0`，`(1,1)` 为 `+sigma*J_tilde/E0`。110通过精确超晶胞映射产生四基点，不能复制两基点索引冒充同一几何。独立审计以原文补充材料为准，代码内部一致性不是外部认证。

### 2.2 随机方程、积分器分工与材料例外

**确定性LLG保留RK4，有限温随机LLG另行选择经过验证的随机积分方案；本协议不要求将现有RK4统一替换为Heun或midpoint。** 截至本次核对的 `bfb3fcc`，Gomonay配置中的自旋波、零交错交换控制、静态畴壁和自由运动畴壁仍使用 `method: rk4`；这些是无热噪声的确定性协议。本节更新仅说明方法分工，不修改求解器代码或运行配置。

| 方法 | 适用任务 | 在本项目中的角色与限制 |
|---|---|---|
| 普通RK4 | 无热噪声的确定性LLG，可包含阻尼或确定性外场 | 保留现有零温色散和畴壁协议；检查时间步、投影前误差及对应守恒/耗散行为 |
| 随机Heun | Stratonovich约定的有限温随机LLG | 作为通用零场热路径的候选主方法；同一步预测与校正复用噪声，投影方式显式登记；须通过统计和步长认证后才能冻结用于生产 |
| 几何midpoint | 随机LLG的几何与数值交叉验证 | 在隐式方程充分收敛时保持模长至求解容差；记录残差与迭代失败，计算成本通常更高，不默认用于全部生产轨迹 |
| 文献指定方法 | 特定文献的复现 | Nishino保留原文显式midpoint，Bauer保留已实现的弱随机RK及其特殊LL约定；显式midpoint与几何隐式midpoint不能混称 |

普通RK4的确定性四阶精度不能直接外推到白噪声随机方程：Wiener增量按sqrt(dt)缩放，子步噪声的构造和复用影响随机积分约定与收敛性质。不能简单向普通RK4加入随机场便宣称随机四阶精度；专门的随机RK须按其强/弱收敛定义验证，不能与普通RK4混用。[随机自旋积分方法参考](https://arxiv.org/abs/1002.1801)

几何midpoint优先用于代表性参数和可负担尺寸的对照验证，不要求每条生产路径用两种方法重复计算。小尺寸通过不能替代目标尺寸的步长与观测量收敛；模长保持也不等于平衡分布、谱或反转率正确。正式主方法与步长应依据目标观测量误差、稳定性及单位有效样本成本冻结；没有相关证据前不预定Heun或midpoint必然优于RK类方法。这里的物理LLG积分器与第三节在运输时间tau上的生成模型ODE积分器分别配置、分别验证。

零场生产强制外场、SOT、STT、电流均为零；有场/有流文献复现独立归档。约化Gilbert方程采用Stratonovich解释：

\[
d\mathbf S_i=-\frac{\mathbf S_i\times[\mathbf b_i d\tilde t+\sqrt{2\alpha\vartheta}\circ d\mathbf W_i]
+\alpha\mathbf S_i\times\{\mathbf S_i\times[\mathbf b_i d\tilde t+\sqrt{2\alpha\vartheta}\circ d\mathbf W_i]\}}{1+\alpha^2},
\quad E[dW_{i\mu}dW_{j\nu}]=\delta_{ij}\delta_{\mu\nu}d\tilde t.
\]

预测—校正复用同一Wiener增量；强误差用嵌套增量，粗增量等于细增量之和，Brownian bridge可用于条件细分。弱统计可以独立抽样，也可使用适当耦合降低方差，不能限定为“弱收敛只能独立随机”。弱RK离散随机变量不能冒充Wiener强耦合。

Nishino原文显式midpoint、Heun、几何midpoint分别登记；保存预测/最终投影前误差、残差、失败与投影选择。Bauer保留 `bauer_ll`：噪声只进入进动项，确定性阻尼为lambda，`epsilon^2=2*lambda*theta`，不能用统一Gilbert式替代。Hirst LLB有纵向自由度，不属于单位球面LLG模长证书或本阶段生成器。

### 2.3 零场热模型与温度标尺

`K_SW=0`仅用于规定的零温色散基准。第一阶段有限温零场路径明确采用 `K=K_DW=0.047 meV` 的易轴模型；配置显式记录 `thermal_model_id,K/E0`，不依赖wall参数默认值。K=0研究作为不同模型另行登记。

二维、有限层数、短程且自旋旋转连续对称的K=0模型不预设非零热力学T_N；空间交换各向异性不自动解除该限制[MerminWagner1966]。先导温度用 `theta=kBT/E0`，不输入未经确定的T/T_N。有限尺寸交叉温度记为T_star(L)，与热力学T_N分开。

相对温度需独立校准：多尺寸平衡、Binder/磁化率和尺寸标度，冻结估计与不确定性。模型T_N不等于块体RuO2实验温度。不得用最终测试动力学结果选择温度标尺。改变K、边界或模型后重新检查。接近/高于临界区仅对所定义的经典固定模长模型作结论，不自动外推为真实材料定量预测。

### 2.4 色散及扩展相互作用

以下j1/j2/jt/kappa均除以E0，kx/ky以a0约化且沿原始晶轴：

\[
a=\cos(k_x/2)\cos(k_y/2),\quad b=1+\kappa/(2j_1)+(j_2/j_1)[\sin^2(k_x/2)+\sin^2(k_y/2)],
\quad c=(j_t/j_1)\sin k_x\sin k_y,
\]
\[
\tilde\omega_\pm=4j_1(\sqrt{b^2-a^2}\pm c),\quad\omega_\pm=\tilde\omega_\pm/t_0,\quad f_\pm=\omega_\pm/(2\pi).
\]

这是对应无阻尼线性基准；110倒空间和带折叠映射到同一物理波矢。磁矩/旋磁比未审定前只报约化频率。

DMI扩展采用定向键 `h_DMI=sum d_ij dot (S_i cross S_j)`，登记d_ji=-d_ij、计数、方向和负梯度检查。J1–J2也登记有符号权重；多个交换常数不证明阻挫，需具体竞争几何、基态和结构因子证据。独立DMI、独立阻挫之后再做联合挑战，不把这些项添加成Gomonay原模型的隐含成分。

## 三、完整路径RFM技术规格

### 3.1 张量、随机源与锚定

目标路径Y为 `[B,F,A,Nx,Ny,3]`，物理初帧S_init=Y[:,0]。自由路径流形为 `(S²)^((F-1)*A*Nx*Ny)`。每次独立采样完整参考路径Z，强制Z[:,0]=S_init；其余帧随机，记录分布、时间相关、尺度和seed。参考噪声是潜变量，不是LLG热噪声，不要求与某条真路径Wiener流逐点对应。参考半径消融不改变真实热浴。

\[
\Pi_s(v)=v-(s\cdot v)s,\quad
\operatorname{Exp}_s(v)=\cos\|v\|s+\frac{\sin\|v\|}{\|v\|}v,
\]
\[
U=\operatorname{Log}_Z(Y),\quad X_\tau=\operatorname{Exp}_Z(\tau U),\quad V_\tau=\partial_\tau X_\tau.
\]

tau是运输时间，不是物理时间。第一帧恒为S_init，速度为零且不计入损失。禁止用 `Exp_S_init(tau*Log_S_init(Y))` 取代随机源插值。小角度用稳定极限；反足附近log/cut-locus处理登记并测试，报告发生率，不隐蔽筛除困难真路径。

### 3.2 网络、条件与对称性

输入 `X_tau,tau,S_init,c`；条件包括theta、alpha、全套约化键/各向异性、物理帧时间、窗口时长、保存间隔、基点和边界，T/T_N仅在冻结校准后使用。晶轴、键向量、DMI、场等条件按实际存在的模型提供。

标量不变量网络和逐块FiLM产生系数，组合自旋、邻域差分、Hamiltonian场及必要叉乘等协变基，再投影到X_tau切平面，输出与路径同形。必须能够表达进动叉乘结构。归一化逐位置作用于通道，避免隐含尺寸统计。

当前PeriodicConv3d只代表规则二维周期实现。多基点、开放边界、任意图需真实接口和独立测试，文档写了图条件不等于网络已经消费它。裁剪须带足够halo或真实全局邻接，不能把内部crop两端变成伪周期邻居。

联合proper SO(3)、周期平移、C4与子晶格/位移联合变换、重编号/图置换分别验证。SO(3)测试需同时变换自旋、场、晶轴和相关张量，不证明晶格对称或材料泛化。反射涉及自旋/DMI变换须另行推导，当前不声称E(3)。100/110比较匹配物理盒长、边界和观测方向；改变晶胞表示本身不应被包装为晶向物理效应。

### 3.3 训练与推理

\[
L_{FM}=E\left[\frac{1}{(F-1)AN_xN_y}\sum_{f=1}^{F-1}\|v_\phi(X_\tau,\tau|S_{init},c)-V_\tau\|^2\right].
\]

硬锚定和切向投影优先于重复软惩罚。可选Hamiltonian辅助项必须给出数学定义、适用系综、权重和消融；禁止有限温能量守恒损失或对未解析热增量施加确定性LLG残差。推理从新的Z出发，用经步数收敛检查的几何积分从tau=0到1；固定第一帧。重复采样形成集合。训练seed、潜变量seed、真实噪声seed分开；编号相同不代表物理耦合。

## 四、初态、数据与划分

数据域 `literature_driven / legacy_v2_driven / zero_field_temperature` 分开。历史V2包含1ps和后续扩展数据，逐文件按manifest登记，不统一说成只有1ps；带SOT数据不能证明零场统计或T_N。

三类初态为基态扰动、随机球面、独立平衡池。前两类用于非平衡热化，第三类用于稳态；分别报告，不任意混成物理系综。平衡池需多链、burn-in、ACF/ESS、链间诊断及磁盆混合；盆内平衡不能冒充全局平衡。

保存 `spins,time,energy,neel,initial_id,noise_id,seed,condition,hash,parent_trajectory_id,source_chain_id,split` 及实际初态。field若重算，记录模型/config/hash并抽样核对；若保存，登记精度和频率。原始数据不覆盖，中断标complete=false并拒绝正式分析。

先按条件和来源族划分，再截窗口。同一initial_id全部噪声、同一父轨迹全部窗口只能属于一个split；相邻相关平衡构型或共同母初态视为同一来源族。优先按独立平衡来源链划分，换seed/window_id不消除依赖。

标准化、核带宽、特征、磁盆阈值和调参只用训练/开发集。Gate-F可迭代但不称最终盲测；正式测试另行独立生成，在模型冻结后一次揭盲。揭盲后修改模型，旧测试转开发，新确认性结论需要新测试批次。

分别登记dt、保存间隔、F、总窗口 `(F-1)*save_dt`、准备时长和单位。20进动周期/50相关时间只可作候选，不代替收敛和事件功效。细保存用于验证下采样是否丢失频谱、越盆及驻留事件。低温反转等待可远大于盆内ACF。多峰反转不是所有条件必需；临界区无稳定磁盆时报告无序化/连续涨落，不强制反转标签。

## 五、指标与统计合同

| 条件 | 能量验收 |
|---|---|
| 零温、无阻尼、无驱动 | 守恒与步长收敛 |
| 零温、有阻尼、无驱动 | 耗散方向与步长收敛 |
| 有限温、非平衡初态 | 与真实LLG热化曲线/分布一致，允许弛豫 |
| 有限温、平衡初态 | 集合分布平稳，单条路径能量允许涨落 |

使用每自旋约化能量或固定物理尺度；初能量接近零时不除以它。“零场”不等于能量守恒。

| 指标 | 实施约束 |
|---|---|
| 逐点球面距离 | 仅用于确定性极限、明确耦合或描述；独立真/生成路径任意配对不作为主排名 |
| MMD/energy distance | 固定初态比较路径集合；登记特征/带宽/归一化/时间权重，加入等样本量LLG–LLG底线 |
| 能量/M/Néel | 按条件、初态类型、时刻/窗口比较均值、分布与尾部 |
| 结构因子 | 用真实基点、子晶格符号与归一化；跨尺寸映射共同物理波矢，不比较不同长度展平向量 |
| ACF/ESS | 使用平稳段，冻结估计器/积分窗口并做窗口敏感性；不机械截在首过零忽略慢尾；多盆/临界用多观测量及多链 |
| 谱 | 登记采样率、时窗、窗函数、分辨率、激发支持和单位，不解释未激发模式任意峰 |
| 事件 | 独立真实数据校准磁盆；区分越零/负末态/完成反转/返回/未决；无磁盆不分类 |
| 首达/生存 | 预定义驻留和事件记进入还是确认时刻；末端驻留窗不足按冻结规则删失；报告KM、RMST、风险人数及删失率 |
| 对称性 | SO(3)、平移、晶格/子晶格、置换分别测，处理零输出分母并报绝对误差 |
| 成本 | 函数评估、batch、硬件、精度、I/O、GPU小时、显存及有效样本吞吐 |

固定初态以噪声路径为重复；跨初态采用初态—噪声分层bootstrap；相关平衡来源再按来源链/时间块处理。重复事件和窗口按父轨迹聚类，训练seed波动单报。bootstrap至少2000次并记seed；少数初态的区间可能不稳，更多噪声不能代替更多初态。

真实参考至少拆成两套独立集合测有限样本底线。CI重叠/不显著不等于等价；预注册主指标、效应量容差、CI和多重比较方案。宽CI记inconclusive。原始高维MMD一个指标不证明全路径正确。

事件精度按功效与CI确定，32/128复制不保证稀有概率准确。独立n条路径零事件仅给单侧95%上界 `1-0.05^(1/n)`。全删失RMST等于观察窗，不是无限时域平均寿命；不外推。停机/补样规则先冻结，不能增加样本直到显著。

## 六、统一执行顺序与Gate-F

| 阶段 | 工作与放行 |
|---|---|
| P0 | 清点来源/版本/资源，冻结问题及适用模型 |
| P1 | 键、field、噪声、积分器及所用文献基准认证，仅对覆盖条件有效 |
| P2 | 明确K的零场热化/平衡先导、初态池、theta和时间标尺 |
| P3 | Gate-F固定窗口开发；允许迭代，不称最终测试 |
| P4 | 冻结生产子矩阵、模型、主指标、测试和资源；不自动解锁 |
| P5 | 正式数据、匹配基线、独立确认性测试 |
| P6 | 分别扩展尺寸/温度、多lag与长时，每个扩展单独认证 |
| P7 | 材料、DMI、阻挫及联合挑战，先补接口/物理证书 |
| P8 | 图、失败范围、复现包及论文 |

### 6.1 Gate-F规模和来源

前置P1/P2覆盖所用热模型。固定K=0.047meV、100晶向、周期L16/32、alpha=0.05；三个theta由独立先导选择并冻结，覆盖可解析的不同涨落强度，不要求跨未知T_N或都出现反转。

每尺寸—温度条件有三类初态，每类4初态，每初态16噪声，共 `2*3*3*4*16=1152` 条。每类2初态训练、2初态开发，即576/576；没有剩余初态再划最终测试。加入同规模110为2304条，先核验四基点接口和物理盒长匹配。最终确认另用独立新初态。F=101或201由P2保存收敛选择并冻结。

开发初态16条真路径固定拆8+8用于真实底线；生成集合以同等数量比较。ED先在每个固定初态内计算，再对组内初态等权汇总，禁止先混合初态掩盖条件学习失败。如此小样本只用于技术可行性，不能包装为高精度泛化证书。

### 6.2 技术GO/HOLD/STOP合同

下面阈值是**项目技术可行性数值选择**，不是文献常数或投稿充分条件。P3前写入并冻结 `gate_f_contract.yaml`：代码/数据版本、初态清单、theta、窗口、特征、bootstrap、随机种子及资源上限。缺项不训练；P1物理容差不在此降低。

| 必须项 | 技术GO条件 |
|---|---|
| 完整性 | 数值有限、哈希/来源/划分通过，无未声明丢样 |
| 模长及初态 | float32最大绝对误差各<=1e-5；float64各<=1e-10 |
| 网络SO(3)/平移 | 至少20个冻结变换，float32相对误差<=1e-4，float64<=1e-8；分母 `max(norm(output),1e-8*sqrt(numel))`，另报绝对误差；源采样器和生成分布另测 |
| 主分布 | 主特征为每自旋约化能量、M/Néel各分量在11固定时点值；训练尺度标准化，距离除sqrt(特征维数)。各尺寸—温度—初态类型组的生成/真ED减真/真ED之95%同时CI上界<=0.10 |
| 条件均值 | 主特征生成—真均值差95%同时CI位于[-0.10,0.10]，不用非显著代替等价 |
| 物理系综 | 分别通过热化或稳态窗口检查；无平衡证据不能报稳态GO |
| 基线 | 同数据/预算比较确定性或平均场、欧氏flow、简单随机自回归；相对最佳学习基线ED差同时CI上界<=0.02为非劣，声称优势另需上界<0 |
| seed与步数 | 三训练seed分别通过；共同潜变量作采样步数加倍检查，主特征均值差CI在[-0.05,0.05]；不能平均后才通过 |
| 事件 | 只在磁盆存在且功效足够时启用；不足记inconclusive，不放行事件率主张，但不阻断固定窗口涨落结论 |

主特征零或极小训练方差的标准化下限须在P3前按物理尺度明确冻结；不能用测试统计选尺度。ED固定使用经验V统计量（包含组内对角项）并对真/生成保持相同样本数；CI按相关结构重采样，主比较用同时bootstrap区间；检验族（条件、主特征、seed、基线）事前列全。所有结构因子、ACF、完整路径和尾部诊断也保存，但上述ED合同仅证明有限技术目标。P5再冻结有足够功效的物理主指标。

GO只表示固定窗口技术可行；HOLD表示CI宽或证据不足。最多一次预注册补充：每开发初态真路径16增至32，生成样本同步匹配，其余不变；按两次查看分配总错误预算0.05，每次使用97.5%同时区间（比表中单次95%更严格），全部检验族同时控制；首次GO可停，HOLD才补样。仍不足则终止本版本为inconclusive。STOP用于内核错误、泄漏、非有限值或预算超限。修改后新contract_id，旧数据保留开发属性，不反复测试到通过。

## 七、生产规模和资源

候选全矩阵：100的L16/32/64/128和110的L16/32/64，共7几何，8温度、3阻尼、3初态类型、每类8初态、每初态32噪声，总129024。8个T/T_N只有独立标尺完成后可用，否则登记8个theta；不是默认提交指令。

按100/110分别2/4自旋、1001帧、三分量float32，未压缩spins约14.40TB（十进制）。逐帧三分量field另需近似同量，尚有备份、统计和生成结果。压缩比实测；盒长为匹配物理尺寸而变化时重新计算。

先测代表条件的每步吞吐、所需积分步数、相关时间、batch/显存、I/O及训练采样耗时；分别估算先导、1152条Gate-F、一次补充和三seed基线。冻结 `resource_budget.json`，含设备、精度、GPU/CPU小时、walltime、磁盘/余量、重试上限。超额停机，不改温度、缩窗或丢失败轨迹；未测前不承诺数小时至两天。P4按功效选择20000–50000或更小子矩阵，条数不等于证书。

## 八、长时扩展和科学加速

固定窗口先闭合，再分别测试多lag、末帧锚定chunk和混合算法。每段存初态哈希与独立潜变量；完整状态的马尔可夫假设须成立，有记忆热浴/隐变量则扩展状态。

CK比较直接2lag与两次lag组合并对照真实LLG，以冻结条件状态/特征测试；不能假定低维序参量本身严格马尔可夫。段界、能量分布、ACF、谱、转移和生存都检查；采样步数、chunk长度、保存频率分别收敛。固定F扩大lag会丢高频与短事件，不能同时承诺保留它们。

`T_valid(condition,observable,tolerance)` 是预注册检查时刻中连续通过范围的最大终点，报告时间分辨率和未测区间，不给无条件全局保证。100–10000个滚动窗口只是候选，数量不证明稳定。

少量高精度LLG从当前生成分布继续演化，不保证消除先前偏差；球面投影只保证几何。混合算法单独认证，不替纯生成器背书。平衡MCMC接受率校正不能直接赋予链步真实物理时间。

同等误差和目标有效样本量、同硬件及优化batch下比较。分别报告推理加速和含数据生产/训练/调参的总成本、摊销盈亏点；未闭合精度只报吞吐。下采样损失带来的速度收益与算法收益分开。

## 九、文献、创新及复现包

| 文献 | 目标及边界 |
|---|---|
| Gomonay2024 | Fig.2色散、Fig.3局部磁化、Fig.5畴壁及SI；Fig.4尖端力需另加非均匀场探针，不是零场生成器默认必做 |
| Bauer2011 | 铁磁单原子开放链热激活反转、长度/温度/阻尼寿命；改变条件的pilot不替原条件长时统计 |
| Nishino2015 | Fig.1 case A/B平衡磁化及分布；自由磁矩不是畴壁反转基准，保留勘误/偏差记录 |
| Hirst2022 | Mn2Au磁化/磁化率、ASD/AFMR、LLB及热梯度壁；LLB单独证书 |
| Laliena2020/2022 | 修正BVP、临界分支、螺旋/孤子及电流；残差小不等于参考值闭合 |

各包保存来源、参数/图号版本、数字化误差、manifest、比较图和报告；仅解析参考时明确标注，不伪造数字化曲线。原图按实际许可再分发，否则留链接/元数据和可分享派生图。并排原图不是认证。

Timewarp用normalizing flow作为MCMC提议改善平衡采样，不是RFM，也不能将其采样链当真实时间路径。TITO为更直接的转移动力学近邻，需比较状态、lag、信息、动力学指标和成本。“这些工作共同缺少……”须逐篇证据表支持，验证清单完整性不自动构成创新。

RuO2模型不等于真实样品磁序确认。保留下列质疑来源；支持/质疑完整证据表仍待核对，不声称争议已解释。移除缺失条目RuO2_review2024。原文D盘R1/R2/R3路径未核验可访问性，不作为互盲审查已完成证据；将来须归档报告及日期/hash。

## 十、参考文献登记

1. **[Gomonay2024]** Gomonay et al., *Structure, control, and dynamics of altermagnetic textures*, npj Spintronics 2,35 (2024). https://www.nature.com/articles/s44306-024-00042-3
2. **[Bauer2011]** Bauer et al., *Thermally activated magnetization reversal in monoatomic magnetic chains on surfaces studied by classical atomistic spin-dynamics simulations*, JPCM 23,394204 (2011). https://doi.org/10.1088/0953-8984/23/39/394204 ; https://arxiv.org/abs/1010.4730
3. **[Nishino2015]** Nishino and Miyashita, *Realization of the thermal equilibrium in inhomogeneous magnetic systems by the Landau-Lifshitz-Gilbert equation with stochastic noise, and its dynamical aspects*, PRB 91,134411 (2015). https://arxiv.org/abs/1507.03075 ; erratum https://doi.org/10.1103/PhysRevB.97.019904
4. **[Hirst2022]** Hirst et al., *Temperature-dependent micromagnetic model of the antiferromagnet Mn2Au: A multiscale approach*, PRB 106,094402 (2022). https://doi.org/10.1103/PhysRevB.106.094402
5. **[Laliena2020]** Laliena et al., *Current-driven dynamics of chiral magnetic solitons in CrNb3S6*, Scientific Reports 10,20430 (2020). https://doi.org/10.1038/s41598-020-76903-8 ; correction https://doi.org/10.1038/s41598-022-06147-1
6. **[Rozsa2019]** Rózsa et al., *Reduced thermal stability of antiferromagnetic nanostructures*, PRB 100,064422 (2019). https://doi.org/10.1103/PhysRevB.100.064422 ; https://arxiv.org/abs/1808.07665
7. **[ChenLipman2023]** Chen and Lipman, *Flow Matching on General Geometries*, ICLR2024. https://arxiv.org/abs/2302.03660
8. **[Lipman2023]** Lipman et al., *Flow Matching for Generative Modeling*, ICLR2023. https://arxiv.org/abs/2210.02747
9. **[Timewarp2023]** Klein et al., *Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics*, NeurIPS2023. https://arxiv.org/abs/2302.01170
10. **[TITO2026]** Viguera Diez et al., *Transferable generative models bridge femtosecond to nanosecond time-step molecular dynamics*, Science Advances 12,eaed2333 (2026). https://doi.org/10.1126/sciadv.aed2333 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC13060594/
11. **[ML_micromagnetics2021]** *Machine learning methods for the prediction of micromagnetic magnetization dynamics*. https://arxiv.org/abs/2103.09079
12. **[ML_magnetoelastic2021]** *Data-driven magneto-elastic predictions with scalable classical spin-lattice dynamics*, npj Computational Materials (2021). https://doi.org/10.1038/s41524-021-00617-2
13. **[ML_exchange2026]** Gao, Bokdam and Kelly, *Smooth overlap of spin orientations: Machine learning exchange fields for ab initio spin dynamics*, PRB 113,144413 (2026). https://doi.org/10.1103/kknv-7ypx （Crossref出版元数据已核验；逐方法比较仍待完成。）
14. **[RuO2_challenge2024]** Plouff et al., *Revisiting altermagnetism in RuO2: a study of laser-pulse induced charge dynamics by time-domain terahertz spectroscopy*. https://arxiv.org/abs/2412.11240
15. **[MerminWagner1966]** Mermin and Wagner, *Absence of Ferromagnetism or Antiferromagnetism in One- or Two-Dimensional Isotropic Heisenberg Models*, PRL17,1133 (1966). https://doi.org/10.1103/PhysRevLett.17.1133

这是来源登记，不声称本轮重读所有论文/SI。重点核对Bauer/Rózsa错配、RFM/Timewarp/TITO定位及二维温度前提；投稿前补全逐论断证据及元数据。

## 十一、进展与输出

截至本轮核对b8efdb6的历史记录：V2已训练/评估但不代表零场闭合；Nishino primary为24/24，前序207/216中有接受的平稳性偏差，原fail与接受记录同时保留；R2–R5有65项分析，Bauer短窗无保存帧越零，Hirst壁仍弛豫，Gomonay有内部色散/壁运动证据，Laliena临界值约低2.54%。这是历史证据，不是本轮重算或服务器实时状态。

新状态账本记录日期、commit、config/data hash、命令、原退出码、样本/事件数、作用范围及pass/fail/inconclusive/accepted_with_deviation。每阶段输出inventory/manifest/certificate，Gate-F另存contract/resource_budget/feasibility_report。证书不得只有无范围的pass。

论文主图围绕方法、物理基准、固定初态分布、独立泛化、误差—成本及失败范围组织；长时和稀有事件图仅在对应认证后使用。附录保留全seed、分割审计、收敛和失败例。期刊选择在结果与创新成立后进行，期刊名称不构成放行门槛或发表承诺。

## 十二、修订记录

修订2覆盖：随机源RFM与时间维；约化单位/符号/Bauer例外；非零K热模型与温度前提；能量/磁盆判据；条件分布和真实底线；来源隔离及层级统计；1152/2304样本与14.40TB预算；统一阶段和可执行开发门槛；混合校正/CK/T_valid边界；文献错配、外部审查证据和研究范围。所有新增技术阈值均是设计选择，不是实验认证结果。
