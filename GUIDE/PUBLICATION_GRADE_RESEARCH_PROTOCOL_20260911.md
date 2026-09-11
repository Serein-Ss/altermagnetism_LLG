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

随机 LLG 将交换、各向异性、阻尼和热涨落连接到自旋波、畴壁、弛豫和热激活跃迁。Gomonay 论文提供 RuO2 类 d-wave 交错磁的晶向相关色散和畴壁基准；Bauer 研究随机反铁磁畴壁反转和 Arrhenius 统计；Nishino–Miyashita、Hirst 和 Laliena 分别提供有限温自旋、Mn2Au ASD/LLB/AFMR、CrNb3S6 螺旋动力学参照。

机器学习方面，Riemannian Flow Matching 将流匹配推广到流形；Timewarp 用条件流加速分子动力学；TITO 学习多时间尺度、跨体系的转移分布，并同时评估热力学和动力学。现有工作说明流模型学习动力学已有先例，本项目缺口应定义为磁自旋流形、键级相互作用、随机路径复制和磁性条件外推的统一可审计验证。

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

1. Gomonay: https://www.nature.com/articles/s44306-024-00042-3
2. Bauer: https://arxiv.org/abs/1010.4730
3. Rózsa: https://arxiv.org/abs/1808.07665
4. Riemannian Flow Matching: https://arxiv.org/abs/2302.03660
5. Flow Matching: https://arxiv.org/abs/2210.02747
6. Timewarp: https://arxiv.org/abs/2302.01170
7. TITO: https://pmc.ncbi.nlm.nih.gov/articles/PMC13060594/
8. Nature criteria: https://www.nature.com/nature/for-referees/policies-and-processes
9. NMI aims: https://www.nature.com/natmachintell/submission-guidelines/about/aims
10. NCS aims: https://www.nature.com/natcomputsci/natcomputsci/natcomputsci/about/aims
11. PRL criteria: https://journals.aps.org/prl/about

互盲审查：`D:/WORKSPACE/CodePlace/publication_review_20260911/R1.md`、`R2.md`、`R3.md`。训练入口：`scripts/training/train.py`、`scripts/inference/sample.py`。
