# 物理约束随机 LLG 路径生成：论文级研究协议

**版本**：2026-09-11　**对象**：服务器 AI　**状态**：方案与证据记录，不代表实验全部完成。

## 一、研究问题、意义、方法与创新

### 1. 研究问题

给定完整初态、Hamiltonian、温度、阻尼、晶格几何和边界，能否学习有限温随机 LLG 的条件路径分布

$$
p_\theta[\mathbf S_{0:T}\mid \mathbf S_0,\mathcal H,T,\alpha,G]
$$

并在未见初态、噪声实现、尺寸、温度、晶向和材料参数上保持短时转移、稳态统计与动力学事件的正确性？研究对象是路径分布，不是平均轨迹、末态分类或单条确定性预测。

### 2. 研究意义

随机 LLG 的细时间步限制了长时间、稀有跃迁和大尺寸统计。若模型能直接近似条件转移路径，就可能减少逐步积分成本，同时保留自旋几何、热涨落和动力学机制。价值必须由误差、速度和失效边界共同证明。

### 3. 方法和可证伪创新

先用显式键 Hamiltonian 和 Stratonovich LLG 生成数值认证数据，再在 $(\mathbb S^2)^{N_s}$ 上训练条件 Riemannian flow matching。候选创新及证明分别为：路径级而非末态生成（逐点/转移/长时指标）；自旋流形和 Hamiltonian 协变（SO(3)、平移和反事实消融）；同初态独立噪声和条件块留出的可审计协议（分层方差/ESS）；DMI/阻挫复杂相互作用泛化（留出耦合比、几何和材料）；长时误差边界（能量漂移、删失、算力）。RFM、Timewarp 和 TITO 已有流匹配与可迁移动力学先例，未经系统比较不得写“首次、通用、无偏”。

## 二、文献与 Introduction

随机 LLG 将交换、各向异性、阻尼和热涨落连接到自旋波、畴壁、弛豫和热激活跃迁。经典 LLG 的热平衡条件和涨落耗散关系由 Nishino–Miyashita 系统讨论 [Nishino2015]；Bauer 研究了反铁磁畴壁的随机反转、长度标度和 Arrhenius 统计 [Bauer2011]；Rózsa 等进一步展示了热激活磁结构寿命的统计困难 [Rozsa2019]。Gomonay 等给出 RuO2 类 d-wave 交错磁的晶向相关色散、亚晶格各向异性和畴壁动力学 [Gomonay2024]；Hirst 等建立 Mn2Au 从第一性原理到 ASD/LLB 的多尺度链条 [Hirst2022]；Laliena 等研究 CrNb3S6 手性螺旋和电流动力学 [Laliena2020]。

磁学中还存在 DMI、偶极相互作用、交换竞争、阻挫、拓扑缺陷和自旋晶格耦合。DMI 可产生手性畴壁、螺旋和非互易自旋波；三角/Kagome/$J_1$–$J_2$ 晶格的竞争交换会造成简并、长相关时间和多稳态。它们是本项目第二阶段挑战集，而不是 Gomonay 原文 Hamiltonian 的隐含项。RuO2 的磁序和 altermagnetism 仍有样品依赖和争议，必须同时记录支持和质疑证据，不能把单一材料结论外推到所有薄膜 [RuO2_review2024] [RuO2_challenge2024]。

机器学习与磁学结合已有三条路线。第一类用机器学习预测交换场或磁性机器学习势，再嵌入 ASD/自旋晶格动力学，例如磁性 Gaussian approximation potential 和数据驱动 magneto-elastic 势 [ML_exchange2026] [ML_magnetoelastic2021]。第二类用神经网络直接学习微磁学磁化动力学或做代理模型，目标是减少重复求解成本 [ML_micromagnetics2021]。第三类是物理信息神经网络、神经 ODE/SDE、等变网络和生成模型，用约束或概率转移表示动力学；Riemannian Flow Matching 处理流形 [ChenLipman2023]，Timewarp 和 TITO 学习跨时间尺度转移 [Timewarp2023] [TITO2026]。

这些工作的共同限制是：磁性路径的同初态独立噪声、多初态分层、晶格键级条件、零场长时稳态和稀有事件通常没有同时作为验收对象。因此本项目的重要性不应只表述为“把 flow matching 用在磁性上”，而应表述为：建立从物理 Hamiltonian、随机 LLG、可审计路径数据到跨条件生成和长时统计的完整验证链，并测试该链在交错磁、DMI 和阻挫体系中的边界。

## 三、Methods

### 3.1 Hamiltonian 和 LLG

$$
H=-\sum_{(i,j)\in E}J_{ij}\mathbf S_i\cdot\mathbf S_j-K\sum_i(S_i^z)^2-\sum_i\mathbf B_{ext}\cdot\mathbf S_i.
$$

$$
\mathbf b_i=-\frac{\partial H}{\partial\mathbf S_i}=\sum_jJ_{ij}\mathbf S_j+2KS_i^z\hat z+\mathbf B_{ext}.
$$

零场数据强制 $\mathbf B_{ext}=0$、SOT=0、STT=0、电流=0。随机 Gilbert 方程为

$$
d\mathbf S_i=-\frac{\mathbf S_i\times(\mathbf b_i dt+\sqrt{2\alpha\vartheta}\,d\mathbf W_i)+\alpha\mathbf S_i\times[\mathbf S_i\times(\mathbf b_i dt+\sqrt{2\alpha\vartheta}\,d\mathbf W_i)]}{1+\alpha^2}.
$$

进动、阻尼和噪声分别来自叉乘项、双叉乘项和 Wiener 增量。采用 paper midpoint、Heun、几何 midpoint；保存投影前误差。强收敛复用 Brownian bridge，弱收敛才独立随机。

### 3.2 Gomonay 基准

注册 $J_1=11.1$、$J_2=1.88$、$\tilde J=0.8$ meV，$K_{SW}=0$、$K_{DW}=0.047$ meV。100 晶向为二基点，110 为四基点，几何取 `scripts/literature/gomonay_2024/model.py`。

$$
a=\cos(k_x/2)\cos(k_y/2),\quad b=1+\frac K{2J_1}+\frac{J_2}{J_1}[\sin^2(k_x/2)+\sin^2(k_y/2)],
$$

$$
c=\frac{\tilde J}{J_1}\sin k_x\sin k_y,\quad \omega_\pm=4J_1(\sqrt{b^2-a^2}\pm c).
$$

### 3.3 DMI 和阻挫

$$
H_{DMI}=\sum_{(i,j)\in E}\mathbf D_{ij}\cdot(\mathbf S_i\times\mathbf S_j),
$$

$$
H_{J_1J_2}=-J_1\sum_{\langle i,j\rangle}\mathbf S_i\cdot\mathbf S_j-J_2\sum_{\langle\langle i,j\rangle\rangle}\mathbf S_i\cdot\mathbf S_j.
$$

先做无 DMI/无阻挫，再做单独 DMI、单独阻挫，最后联合挑战。Gomonay 原始实现没有独立 DMI 项；多个交换常数本身也不等于阻挫相，须由闭环、基态简并和结构因子验证。

### 3.4 生成模型

$$
\mathcal M=(\mathbb S^2)^{N_s},\qquad \Pi_s(v)=v-(s\cdot v)s,
$$

$$
\operatorname{Exp}_s(v)=\cos\|v\|s+\sin\|v\|\frac v{\|v\|}.
$$

$$
x_\tau=\operatorname{Exp}_{x_0}(\tau\log_{x_0}x_1),\qquad
\mathcal L=\mathbb E\|v_\phi(x_\tau,\tau\mid c)-\partial_\tau x_\tau\|^2.
$$

$\tau$ 是运输时间，不是物理时间。输入为参考自旋路径、$\tau$、$T$、$T/T_N$、$\alpha$、lag、耦合比、尺寸、边界、晶向、键图和 DMI 向量；输出为切向速度场，经 ODE 积分得到随机路径。PeriodicConv3d 目前只适合周期二维网格；跨边界、可变基点和阻挫图需 mask 与 bond message passing。固定半径参考噪声必须与高斯半径和真实 LLG 增量消融。

### 3.5 评估

逐点 geodesic 误差；MMD/energy distance；能量、磁化、Néel 序参量、结构因子、谱、ACF；首达时间、Kaplan–Meier、RMST、删失率；模长/能量漂移、等变误差、函数评估次数、GPU 小时和加速比。均值配 block bootstrap CI 和 ESS。基线包括逐步 LLG、平均场、欧氏 flow、扩散、神经 SDE、自回归和去掉物理约束的消融。

## 四、数据设计与用途

数据域分为 `literature_driven`、`legacy_v2_driven` 和 `zero_field_temperature`。V2 是 5 K、强 SOT、1 ps、101 帧历史数据，只用于调试，不用于零场温度或 $T_N$ 判断。

正式条件由体系、晶向、尺寸、温度、阻尼、边界、初态类型和相互作用共同定义。Gomonay 建议 100 晶向 $L=16,32,64,128$，110 晶向 $L=16,32,64$，$T/T_N=0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3$，$\alpha=0.01,0.05,0.10$。这些是统计设计，不是文献唯一规定值。

每条件含基态扰动、随机球面态、平衡池三类初态；每类至少 8 个初态，每个初态至少 32 个独立噪声复制，稀有事件提高至 128 或按功效分析确定。平衡池需 burn-in、ACF、ESS、能量和多链一致性。每条样本保存 `spins,time,energy,field,neel,initial_id,noise_id,seed,condition,hash`。同一条件内使用相同窗口；短时至少覆盖 $20\tau_{prec}$，长时至少 $50\tau_{corr}$，1001 帧是保存工程默认值，不是物理要求。

训练使用零场轨迹的固定长度连续窗口；验证使用不同初态和 seed；测试整块留出温度、尺寸、晶向或材料。输出是与请求窗口相同帧数的路径，重复采样得到路径集合。变长能力分三阶段：固定窗口；把 lag 作为条件的多时间尺度；固定 chunk 滚动生成长路径。滚动必须检查段间能量、序参量、ACF 和转移核漂移。

### 4.6 长时间路径模拟研究包

长时间模拟是后续独立研究目标。当前模型一次生成固定窗口，不能直接等同于长期稳定性。实施三条路线：

1. **多 lag 直接生成**：把物理 lag $\Delta t$ 作为条件，训练短、中、长窗口，并在未见 lag 上比较转移分布。
2. **chunk 滚动生成**：每次生成 $F_{chunk}$ 帧，把末帧作为下一段初态，累积 $10^2$ 至 $10^4$ 个窗口；记录每段初态哈希、随机种子和条件。
3. **混合校正**：生成若干 chunk 后插入少量高精度 LLG 或约束校正；分别报告生成器本身和混合算法结果。

长时数据同时包含短期高分辨率和长期低保存频率轨迹。验收包括能量/Néel 序参量漂移、模长、chunk 边界跳变、ACF、功率谱、稳态能量分布、Chapman–Kolmogorov 一致性、首达时间、Kaplan–Meier/RMST 和 rollout 置信区间。若误差随 chunk 累积、稳态偏离或事件率超出预注册区间，则记录最大可信时长 $T_{valid}$，不得使用更长结果作科学结论。接近 $T_N$、强阻挫和 DMI+阻挫体系需增加参考时长和复制数。成功标准是同时保持短时转移、稳态和事件统计，并给出速度与 $T_{valid}$；否则只称固定窗口生成器。

## 五、执行计划与实时进展

```text
P0 inventory/source audit -> P1 Hamiltonian/noise/integrator certificates
-> P2 literature completion -> P3 zero-field equilibrium pools
-> P4 long-path pilot -> P5 factorial data -> P6 baselines/held-out tests
-> P7 figures/release/manuscript
```

当前：V2 330 条带驱动短轨迹；Gomonay 有小尺寸频谱、零 $\tilde J$ 对照和单一自由壁速度；Nishino primary 条件已运行但平稳性需独立统计；Bauer 无越零事件；Hirst 畴壁仍弛豫；Laliena 临界分支与目标差约 2.54%。每次更新本节记录日期、commit、命令、哈希、输出、样本数、判定、失败项和下一步。证书失败即停止该体系正式训练，保留失败版本。

## 六、实验包与论文证据

**A 物理包**：五篇文献逐曲线、参数和收敛。**B 零场包**：多初态、多噪声、多尺寸、多温度、长时。**C 方法包**：基线和消融。**D 泛化包**：条件、材料和 DMI/阻挫留出。**E 复现包**：环境、配置、哈希、失败日志、许可和图脚本。每包生成 `inventory.json`、`manifest.json`、`certificate.json`。

## 七、期刊门槛与当前差距

| 期刊 | 门槛 | 当前差距 |
|---|---|---|
| Nature | 卓越重要性、跨学科结论、强证据 | 尚无改变领域认识的新发现，物理闭环不足 |
| Nature Machine Intelligence | ML 原创性和科学影响 | RFM/Timewarp/TITO 有近邻，当前架构和基线不足 |
| Nature Computational Science | 计算方法推动复杂科学问题 | 长时稳定、误差-成本和开放复核未完成 |
| PRL | 集中的决定性物理突破 | 尚未形成单一新物理结论 |

## 八、正文与附录图表

正文六图：问题与方法、Gomonay 连续色散、同初态多噪声、尺寸/温度盲测、长时统计/事件率、基线/消融。附录：五文献全曲线、所有 seed、初态和复制审计、数值收敛、等变性、失败实例、DMI/阻挫挑战、资源统计。所有图由 manifest 脚本重建。

## 九、参考文献

1. **[Gomonay2024] Gomonay et al., “Structure, control, and dynamics of altermagnetic textures”**，npj Spintronics 2, 35 (2024)。文献参数、晶向、自旋波与畴壁基准：[文章链接](https://www.nature.com/articles/s44306-024-00042-3)。
2. **[Bauer2011] Bauer et al., “Thermally activated switching in antiferromagnetic nanostructures”**，J. Phys.: Condens. Matter 23, 394204 (2011)，[arXiv 预印本](https://arxiv.org/abs/1010.4730)。
3. **[Nishino2015] Nishino and Miyashita, “Realization of the thermal equilibrium in inhomogeneous magnetic systems by the Landau-Lifshitz-Gilbert equation with stochastic noise, and its dynamical aspects”**，arXiv:1507.03075，[预印本](https://arxiv.org/abs/1507.03075)。
4. **[Hirst2022] Hirst et al., “Temperature-dependent micromagnetic model of the antiferromagnet Mn2Au: A multiscale approach”**，Physical Review B 106, 094402 (2022)，[文章链接](https://doi.org/10.1103/PhysRevB.106.094402)，[arXiv 预印本](https://arxiv.org/abs/2206.08625)。
5. **[Laliena2020] Laliena et al., “Current-driven dynamics of chiral magnetic solitons in CrNb3S6”**，Scientific Reports 10, 20430 (2020)，[文章链接](https://doi.org/10.1038/s41598-020-76903-8)，勘误：[2022 correction](https://doi.org/10.1038/s41598-022-06147-1)。
6. **[Rozsa2019] Rózsa et al., “Lifetime of antiferromagnetic skyrmions”**，Physical Review B 100, 064422 (2019)，[arXiv 预印本](https://arxiv.org/abs/1808.07665)。
7. **[ChenLipman2023] Chen and Lipman, “Flow Matching on General Geometries”**，ICLR 2024， [arXiv 预印本](https://arxiv.org/abs/2302.03660)。
8. **[Lipman2023] Lipman et al., “Flow Matching for Generative Modeling”**，ICLR 2023， [arXiv 预印本](https://arxiv.org/abs/2210.02747)。
9. **[Timewarp2023] Klein et al., “Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics”**，NeurIPS 2023， [arXiv 预印本](https://arxiv.org/abs/2302.01170)。
10. **[TITO2026] Viguera Diez et al., “Transferable generative models bridge femtosecond to nanosecond time-step molecular dynamics”**，Science Advances 12, eaed2333 (2026)，[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13060594/)。
11. **Nature editorial criteria**，技术可靠性、强证据、新颖性和广泛兴趣：[审稿标准](https://www.nature.com/nature/for-referees/policies-and-processes)。
12. **Nature Machine Intelligence aims**，[期刊范围](https://www.nature.com/natmachintell/submission-guidelines/about/aims)。
13. **Nature Computational Science aims**，[期刊范围](https://www.nature.com/natcomputsci/natcomputsci/natcomputsci/about/aims)。
14. **Physical Review Letters acceptance criteria**，[期刊标准](https://journals.aps.org/prl/about)。
15. **[ML_micromagnetics2021] “Machine learning methods for the prediction of micromagnetic magnetization dynamics”**，arXiv:2103.09079，[预印本](https://arxiv.org/abs/2103.09079)。
16. **[ML_magnetoelastic2021] “Data-driven magneto-elastic predictions with scalable classical spin-lattice dynamics”**，npj Computational Materials (2021)，[文章](https://doi.org/10.1038/s41524-021-00617-2)。
17. **[ML_exchange2026] “Smooth overlap of spin orientations: Machine learning exchange fields for ab initio spin dynamics”**，Physical Review B (2026)，[文章](https://journals.aps.org/prb/abstract/10.1103/kknv-7ypx)。

互盲审查：`D:/WORKSPACE/CodePlace/publication_review_20260911/R1.md`、`R2.md`、`R3.md`。训练入口：`scripts/training/train.py`、`scripts/inference/sample.py`。

## 十、生成模型完整技术规格

### 10.1 张量和条件流

原始路径张量记为 $S\in\mathbb R^{B\times F\times A\times N_x\times N_y\times3}$，其中 $B$ 为 batch，$F$ 为帧数，$A$ 为子晶格数。数据加载器先读取 $S_0$ 和目标窗口 $S_1$，按训练集统计量标准化标量条件 $c_s$，并将键表/边界编码成图条件 $c_g$。参考采样器产生 $z$，但强制 $z[:,0]=S_0$。

对每个格点执行切向投影

$$
\Pi_{S_0}(z)=z-(S_0\cdot z)S_0,
$$

然后计算球面 `log`，得到 $u=\log_{S_0}(S_1)$。训练状态和目标速度为

$$
x_\tau=\operatorname{Exp}_{S_0}(\tau u),\qquad
u_\tau=\partial_\tau x_\tau.
$$

### 10.2 编码器、残差块和输出头

每个时间帧构造标量不变量通道：$S\cdot S$、相邻点积、$S\cdot\mathrm{roll}(S)$、$S\cdot b_H$、条件标量和 $τ$。向量基包括 $S$、邻域差分、空间 Laplacian、有效场、晶轴和 DMI 向量。所有系数只由标量通道产生，向量输出由这些协变基线性组合。

张量首先经过 3D 卷积：

$$
h^{(0)}=\operatorname{Conv3D}(\operatorname{concat}[q_{scalar},q_{vector\ invariant}]).
$$

第 $l$ 个残差块为

$$
r^{(l)}=h^{(l)}+\operatorname{Conv3D}_2\!\left(\operatorname{SiLU}\left(\operatorname{Norm}(\operatorname{Conv3D}_1(h^{(l)}))\right)\right).
$$

条件 FiLM 在每个块产生 $(\gamma_l,\beta_l)$：

$$
\operatorname{FiLM}(h,c)=(1+\gamma_l(c))h+\beta_l(c).
$$

最后输出每个基的系数 $a_k$，再组合为

$$
\tilde v=\sum_k a_k(q)\,e_k(S,c_g),\qquad
v=\Pi_x(\tilde v).
$$

输出张量形状仍为 `[B,F,A,Nx,Ny,3]`。ODE solver 在运输时间 $τ=0\rightarrow1$ 上积分，得到整段生成路径，而非单独分类标签。

### 10.3 损失、采样和变长

基础损失为

$$
L_{FM}=\frac1{BFAN_xN_y}\sum\|v-u_\tau\|^2.
$$

总损失可包含初态锚定 $L_{anchor}=\|x_0-S_0\|^2$、切向惩罚 $L_{tan}=\|x\cdot v\|^2$ 和 Hamiltonian 一致性 $L_H$，但每个权重必须在实验前冻结。模型先生成固定 $F$ 帧窗口；多 lag 模型把物理 $Δt$ 输入条件；长时模型以固定 chunk 滚动并独立检查误差累积。

## 十一、评估指标计算定义

| 指标 | 计算方法 | 解释 |
|---|---|---|
| 球面误差 | $d(s,\hat s)=\arccos(\mathrm{clip}(s\cdot\hat s,-1,1))$，报告均值和 95% CI | 局部方向误差 |
| 终态分布 MMD | 用 RBF 核 $k(x,y)=e^{-\|x-y\|^2/(2\sigma^2)}$ 计算两样本 MMD，$σ$ 只由训练集定 | 分布差异 |
| Energy distance | $2E\|X-Y\|-E\|X-X'\|-E\|Y-Y'\|$ | 路径/终态分布距离 |
| 能量漂移 | $(E(t)-E(0))/|E(0)|$，零场稳态另报告均值和斜率 | 长时稳定性 |
| ACF | $C(\ell)=E[(q_t-\bar q)(q_{t+\ell}-\bar q)]/C(0)$，积分至首个过零 | 相关时间和 ESS |
| 功率谱 | 对 $q(t)$ 去均值后 FFT，报告峰频、带宽和谱距离 | 进动/自旋波 |
| 结构因子 | $S(k)=N^{-1}|\sum_j q_j e^{-ik\cdot r_j}|^2$ | 空间相关和相 |
| 转移概率 | 在固定初态和 lag 下统计 basin-to-basin 频率，使用 Wilson/Bootstrap CI | 随机动力学 |
| 首达时间 | 首次进入目标 basin 并持续 $τ_{res}$ 的时间 | 跃迁动力学 |
| 生存分析 | Kaplan–Meier $\hat S(t)$、RMST $\int_0^\tau\hat S(t)dt$ 和删失率 | 零事件也可正确报告 |
| 等变误差 | 比较 $f(RS,Rc)$ 与 $Rf(S,c)$ 的相对范数 | 旋转协变 |
| 加速比 | 参考 LLG wall-clock / 生成 wall-clock，固定硬件和目标有效样本数 | 计算收益 |

所有指标按初态和噪声两级 bootstrap，不能把帧数当样本数。测试集只在模型冻结后计算。

## 十二、详细实施计划表

| 阶段 | 具体实现 | 原因 | 输出/验收 | 当前状态 |
|---|---|---|---|---|
| P0 | inventory、环境、commit、文件哈希和文献参数登记 | 防止版本和参数混淆 | inventory.json | 部分完成 |
| P1 | BondHamiltonian、DMI field、噪声方差、能量有限差分、三积分器 | 先证明物理内核正确 | kernel certificate | 部分完成 |
| P2 | Gomonay 100/110 连续色散、零 $\tilde J$、速度和尺寸扫描 | 对应原文理论基准 | reference comparison | 频点证据已有 |
| P3 | 无场热平衡链、burn-in、ACF、ESS、$T_N(L)$ | 得到温度标尺和可信初态 | equilibrium certificate | 未完成 |
| P4 | 同初态 32/128 噪声、多初态和三类初态 | 估计条件随机性，避免伪重复 | replicate audit | 未完成 |
| P5 | 固定窗口零场数据矩阵生成 | 建立不受驱动污染的训练集 | HDF5 + manifest | 未完成 |
| P6 | flow matching、FiLM、图条件、切向输出 | 学习物理条件路径 | checkpoint + seed report | 旧 V2 调试完成 |
| P7 | LLG/RFM/扩散/SDE/自回归基线和物理消融 | 证明创新来源 | benchmark table | 未完成 |
| P8 | 留出初态、温度、尺寸、晶向、材料测试 | 证明泛化而非记忆 | blind test | 未完成 |
| P9 | 多 lag、chunk rollout、长时稳定和事件统计 | 验证长期能力及失效边界 | $T_{valid}$ certificate | 未完成 |
| P10 | DMI、阻挫、联合挑战 | 测试复杂能量景观 | four-domain report | 未完成 |
| P11 | 主文/附录图、数据和代码发布 | 形成可审稿证据链 | reproducibility package | 未完成 |

## 十三、文献复现实验包和图号登记

| 文献 | 必须复现的内容 | 本地数据/图位置 |
|---|---|---|
| Gomonay2024 | Fig.2 自旋波分裂和色散；Fig.3 畴壁局部磁化；Fig.4 磁性尖端力；Fig.5 速度/Walker 行为及 Supplement S7/S9 色散 | `output/literature_reproduction/gomonay_2024/`；当前只完成内部色散和部分壁运动，原文图需下载后与复现图并排 |
| Bauer2011 | 开放链热激活反转、长度依赖、温度/阻尼寿命和 Arrhenius 图 | `output/literature_reproduction/bauer_2011/`；当前短轨迹无越零事件，不能称复现寿命图 |
| Nishino2015 | Fig.1 case A/B 平衡磁化和反转路径；不同噪声/阻尼下平稳分布 | `output/literature_reproduction/nishino_miyashita_2015/`；保留旧失败和新 primary 证书 |
| Hirst2022 | Mn2Au 温度相关磁化/磁化率；ASD 阻尼振荡；AFM-LLB；热梯度畴壁 | `output/literature_reproduction/hirst_mn2au_2022/`；当前壁宽和 AFMR 仍未闭合 |
| Laliena2020/2022 | 修正 BVP 临界 Gamma；螺旋剖面；电流下孤立手性孤子速度/宽度 | `output/literature_reproduction/laliena_crnb3s6_2020/`；当前临界值差约 2.54% |

原论文图像必须保留来源、图号、下载日期和许可证；仓库发布时使用允许再分发的截图或只保存数字化曲线，不能把版权图直接替换为项目图。每个复现目录必须有 `reference_manifest.json`、`digitized_data/`、`comparison.png` 和 `report.md`，并标注“原文图”与“本项目复现”。

新增交叉文献：**[ML_micromagnetics2021] “Machine learning methods for the prediction of micromagnetic magnetization dynamics”**，arXiv:2103.09079，[预印本](https://arxiv.org/abs/2103.09079)；**[ML_magnetoelastic2021] “Data-driven magneto-elastic predictions with scalable classical spin-lattice dynamics”**，npj Computational Materials (2021)，[文章](https://doi.org/10.1038/s41524-021-00617-2)；**[ML_exchange2026] “Smooth overlap of spin orientations: Machine learning exchange fields for ab initio spin dynamics”**，Physical Review B (2026)，[文章](https://journals.aps.org/prb/abstract/10.1103/kknv-7ypx)；**[RuO2_challenge2024] Plouff et al., “Revisiting altermagnetism in RuO2: a study of laser-pulse induced charge dynamics by time-domain terahertz spectroscopy”**，arXiv (2024)，[预印本](https://arxiv.org/abs/2412.11240)。
