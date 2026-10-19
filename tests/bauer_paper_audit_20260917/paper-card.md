# Bauer 2011：逐图解读、复现数据与预算核查

> Source coverage: Full paper（项目内 arXiv v2 PDF及对应HTML）；未取得作者原始数据。
> Extraction confidence: Mixed；正文和图注可读，未完成曲线数字化。
> Locator mode: structure-grounded
> Primary analytical lens: discovery（热激活翻转机制）
> Secondary analytical lens: None
> Context verification: Targeted external check（论文元数据），不是新颖性检索。
> Card completeness: Complete relative to supplied source；严格逐点复现的未公开条件仍未知。

精读脚本因 zrs-mag 缺少 PyMuPDF 失败，按技能回退到 pdftotext 和HTML；不使用PDF页码作为证据定位。记录日期2026-09-17。本次不提交模拟、不修改生产配置。

## 01 基本信息

[Paper] David S. G. Bauer、Phivos Mavropoulos、Samir Lounis、Stefan Blügel，*Thermally activated magnetization reversal in monatomic magnetic chains on surfaces studied by classical atomistic spin-dynamics simulations*，J. Phys.: Condens. Matter 23, 394204 (2011)，DOI 10.1088/0953-8984/23/39/394204。作者机构为Forschungszentrum Jülich及UC Irvine；文中提及juSpinx，未提供可复用原始轨迹数据集。[Paper: Title; Introduction]

[原文HTML](https://arxiv.org/html/1010.4730v2)；[本项目PDF](../../docs/attachments/data/literature_reproduction/bauer_2011/reference/bauer_2011_magnetic_chain.pdf)。PDF标题页存在2018排版日期，而arXiv版本标识为2010，发表记录为2011；不将排版日期当发表年份。

术语：寿命τ=相邻稳定磁化翻转之间的等待时间；过渡持续时间=一次翻转过程本身；畴壁宽度L_DW与关联长度ξ不同；θ=kBT/J是温度，不是旋转角；λ是Bauer LL阻尼，不直接当Gilbert α。

## 02 一句话总结

[Paper] 随机经典自旋动力学将有限磁链的翻转等待时间与端部成核、畴壁随机传播联系起来，并显示三轴各向异性能够在最低能垒不变时改变寿命前因子。[Paper: Conclusions]

## 03 研究问题

[Paper] 无外场时磁链能保持一个磁化方向多久？寿命如何受链长、温度和各向异性影响？有限链是否能被一个整体宏自旋准确代表？[Paper: Introduction; Section II]

## 04 背景和发展路线

[Paper-framed; external verification not performed] 原文从表面原子磁链的双稳态与信息保持出发，对比宏自旋整体旋转和非均匀成核传播；强调经典自旋近似在金属基底体系中的适用性边界。未对其2011年的新颖性做独立检索。[Paper: Introduction]

## 05 核心问题

| 问题 | 原文证据 | 对当前项目的含义 |
|---|---|---|
| 平均磁化不能定位翻转机制 | Figure 2空间时间图 | 保存逐格点自旋，不只保存平均值 |
| 等待时间远长于过渡过程 | Figure 2、Appendix A | 2 ps可视化不够验收寿命 |
| 相同最低能垒不保证相同寿命 | Figure 4–5 | 不能只验证能量或静态分布 |

## 06 核心思想

[Paper] 在L_DW远小于L且L远小于ξ的范围，端部先形成反向磁畴，畴壁或返回消失、或走到另一端完成翻转；动力学概率改变寿命而不只取决于静态能量。[Paper: Section II; Figure 2–3]

[Analysis] 对生成模型，这是同时检查路径形状和路径权重的候选基准，不只是终态预测。

## 07 方法概览

[Paper] 最近邻铁磁交换、单离子各向异性、开放边界、零外场、单位自旋；随机LL方程按Stratonovich解释，噪声只在进动项，强度与λ和温度满足涨落耗散关系。Appendix A使用Milstein–Tretyakov弱随机RK，并用长轨迹连续翻转间隔估计寿命；没有神经网络训练。[Paper: Equations 1–4; Appendix A]

## 08 模块拆解

| 模块 | 输入输出 | 证据与边界 |
|---|---|---|
| 哈密顿量 | 自旋到能量、有效场 | Eq.1、3；无DMI与偶极 |
| 随机动力学 | 状态与独立噪声到后继状态 | Eq.2、Appendix A；不是普通确定性RK4 |
| 事件统计 | 连续路径到等待间隔、平均寿命 | Appendix A；具体驻留判据未完整公开 |
| Arrhenius拟合 | 多温度寿命到能垒和前因子 | Eq.4、Figure 1、4 |
| 空间机制观察 | 逐格点时间序列到成核和传播 | Figure 2；单例不能确定机制发生概率 |

## 09 公式与物理单位

[Paper] τ=τ₀ exp(E_B/kBT)；长链E_B约等于2√(2JK)，L_DW约等于2√(J/K)；短链极限E_B约等于LK。τ₀在长链区近似正比于L−L_DW，不能越过相关长度后仍无限外推。[Paper: Equations 4–5; Section II]

[Paper] 三轴取Kx=0、Ky≤Kz，则Δlow=Kz−Ky，Δhigh=Kz；端部畴壁困陷的连续近似条件为Kedge>K+E_DW/4。[Paper: Equations 6–7; Appendix B]

[Analysis] 为向实际单位换算，可统一选Section IV的J=50 meV作为显示尺度：ℏ/J=0.013164239 ps，J/kB=580.2259 K。这不表示图1–5原本都指定了Co/Pt这套实际材料参数。

## 10 逐图证据链与所需数据

### Figure 1：激活能垒随链长

[Paper] 横轴L、纵轴E_B/J，两组K/J=0.01与0.1。点来自多温度寿命的Arrhenius拟合；水平线是连续近似畴壁能垒，分别约0.28284与0.89443。较短链出现偏离，提示从畴壁机制向整体旋转过渡。[Paper: Figure 1; Section II]

[Analysis] 每个L、每个K需要多温度独立长轨迹、翻转间隔及不确定度；不能用一次静态畴壁优化替代该图。确切温度网格、全部点坐标和每点事件数未公开，需数字化与明确自选设置。

### Figure 2：完整路径中的成功和失败尝试

[Paper] L=100、K/J=0.1、θ=0.11、λ=0.1；总时长750000ℏ/J，三段各250000ℏ/J。上方为总Mz而不是平均mz；下方横轴时间、纵轴站点、颜色为局域sz符号。A/C/F返回，B/D/E翻转；A/D内部成核，B/C/E/F端部成核。原图特意选取同时含这些过程的时间段，不应从它直接估计无偏机制概率。[Paper: Figure 2; Section II]

[Analysis] 取J=50 meV，温度63.82485 K，总时长9.87318 ns。需完整自旋轨迹、总磁化和成核/返回/完成事件表。无需相同随机种子，也不应保证出现同样的字母事件序列。

### Figure 3：寿命随链长

[Paper] 横轴L、纵轴寿命，以百万ℏ/J为单位；K/J=0.1曲线人为乘10以便显示。长链近线性增长，短链有畴壁宽度相关的偏移与拐折；竖线约为6、20个站点。[Paper: Figure 3; Section II]

[Analysis] 每条曲线需要固定温度下的多L寿命。已读正文与图注没有明确两组曲线的温度，不能猜测它们相同，更不能据图宣称较大K反而降低寿命。原参数缺口需向作者核实或明确标注为机制复现。

### Figure 4：高能垒改变前因子

[Paper] L=100、Δlow/J=0.01固定，图例Δhigh/J=0.010、0.011、0.020、0.100。上图τ对温度；下图logτ对逆温度，同一组数据的两种画法。近似平行斜率支持相近E_B，截距差异反映τ₀不同；增加高能垒可以缩短寿命，较大差异后趋于饱和。[Paper: Figure 4; Section III]

[Analysis] 取J=50 meV，Δlow=0.5 meV，高能垒0.5、0.55、1、5 meV；温度轴约17.4–31.9 K，非精确采样表。低温事件稀少，需延长观察并给误差条。自然对数与常用对数的斜率换算不同，应在重画时明确。4组各向异性即4个不同参数的哈密顿量，不是只改温度。

### Figure 5：单自旋轨迹为何受三轴影响

[Paper] 单自旋，横Mx纵My；左(Kx,Ky,Kz)=(0,0,K)，右(0,9K,10K)，相同时长从+z出发。左右差别不只是圆形/椭圆形：右侧更容易到达Mx²+My²=1即Mz=0附近。作者用进动将高能垒方向涨落转向低能垒方向解释Figure 4。[Paper: Figure 5; Section III]

[Analysis] 这是磁矩方向空间，不是原子空间，也不是网页里的格点热图。需要L=1完整三分量轨迹，两种各向异性、多seed配对条件；仅展示一对轨迹不足以验收首次到达时间分布。K绝对数值、温度和总时长未完整给出，不能承诺原图路径逐点一致。

### Section IV与附录

[Paper] Co/Pt估算采用L=50、J=50 meV、Δlow=0.8 meV、Δhigh=1 meV、λ=0.1，给出τ₀约0.6 ps、E_B约18 meV；低温实验可观测性是基于Arrhenius的数量级预测。Appendix B为端部各向异性困陷判据，未另画第6张主图。[Paper: Section IV; Appendix B]

[Analysis] 当前网页正是这套实际参数附近的2 ps核验，并非Figure 2的参数。代入文中近似拟合，50 K寿命约39 ps，说明2 ps不适合认证寿命；10 K约0.000708秒，说明正文“约10 K或更低”不能解读为10 K精确对应1秒，1秒约需7.4 K。该计算是外推，不是新的模拟测量。

## 11 结论应如何理解

[Analysis] 可规划重现五图的机制和统计规律；目前不能保证严格逐点复现，原因包括未公开条件与当前未认证状态。随机路径无需复制原文噪声；需要相同条件下统计规律一致。仅单位模长正确或损失下降不能推出寿命、成核概率正确。

## 12 作者明确指出的限制

[Paper] 双稳态分析要求链长低于关联长度；高温下寿命前因子不再近常数；真实Co/Pt阻尼未确定、未来需考虑更远邻交换和DMI；经典近似的尺寸、温度、时间适用范围仍需研究。低温误差大与有限时间内翻转少有关。[Paper: Introduction; Sections II–V]

## 13 当前项目差距与预算

[Analysis] 已有OpenChain、弱随机RK、CPU Numba核、事件统计和Arrhenius拟合工具，不是从零开始。但快速CPU核目前是单轴模型，campaign入口传入K=.1和λ=.1等硬编码，不能只改YAML复现三轴扫描。最终归一化是项目扩展，仍需检验长期事件统计；缺原文温度点与一致的事件认定协议。旧数据有分片，必须检查完整性与证书再复用。

历史证据：[已完成64轨迹记录](../../assets/literature_reproduction/bauer_2011/experiments/reference_dynamics/20260913_classical_independent_original_dt002/analysis/result.json)。L100、θ=.11、时长750000、dt=.02、64条、fat 8核，23412.45秒=6.503小时；Slurm记录为完成。状态仍是complete_not_certified。

按相同实现、相近IO和8核资源线性估算：dt=.01约13小时，dt=.005约26小时，三档合计45.5个8核任务小时。这不是所有体系已测速度，三轴和不同批量需重测。100–150个条件各一批64条同长轨迹，在粗步长约650–975个8核任务小时，细步长约2600–3900小时；4个同等任务并行约7–10天或27–41天，不含排队、开发、返工和事件数不足的续算。

图4低温寿命可达数十ns，仅每条9.87ns不能满足500个完成事件。500个事件是已有项目门槛，不是原文报告的采样量，也不是无条件精度保证。预算应按总暴露时间、有效事件数、独立副本和误差要求重估；有右删失时不可把完成事件平均当作无偏完整平均寿命。

2026-09-17资源只读快照：fat六节点中五个drain、一个mixed，后者80逻辑CPU、约772GB标称内存；rtx4090分区的实际RTX4090节点drain，另有8张RTX3090节点idle。分区名不保证显卡型号，idle也不保证用户立即能获得全部资源。未做GPU基准、未提交任务，不能承诺GPU加速倍数。

64条L100轨迹按150001帧、三分量float32保存，自旋原始数组约11.52GB；100–150条件一档约1.15–1.73TB。正式统计可在线存事件/观测量，保留稀疏全程与高分辨率事件窗口；训练需要的数据不可只保留成功事件。

## 14 可迁移的知识

Agent-derived knowledge candidates：[Analysis] 等待时间不同于过渡时间；静态能垒与动力学前因子要同时验证；畴壁宽度不同于关联长度；图3的展示乘数必须消除后再比较；零温扰动轨迹不替代有限温寿命样本。

## 15 与当前研究目标的联系

[User] 小体系训练、同一哈密顿量大体系生成、宏观条件控制。[Analysis] Figure 2的机制、Figure 3的长度规律适合做大体系留出检验；Figure 4需要多个哈密顿量各自训练，或明确扩展模型的条件设计，不能假定温度和尺寸足以标识不同各向异性。大体系LLG参照仅作盲测，不混入小体系训练。

## 16 后续候选任务

Agent-derived research candidates；创新状态：unverified；本次不提出已证实的新颖性。

[Hypothesis] 小体系覆盖成核、返回、畴壁传播等局部状态及其无偏随机后继后，模型能在L100恢复Figure 2–3的事件统计。相对原文改变的是路径生成器，不改变物理定义。[Paper: Figure 2–3]

如何验证：先恢复旧原条件数据并核验，补齐三轴快速核，冻结事件定义和温度点，完成小规模多seed吞吐/事件率测试，再生成正式统计数据。候选预算为单轴2种K×8个L×6个温度=96条件，三轴4×6=24条件，Co/Pt额外6温度；合并重复点后约120条件。L与温度点是项目规划，不冒充原文点；小L扩展单独标注。每条件可先64副本，但以事件有效样本和置信区间验收。

可能失败：训练样本不含关键局部状态；有限感受野遗漏集体效应；积分器或事件门槛引入寿命偏差；增加各向异性却不告诉模型体系身份导致目标分布不唯一。优先Figure 2原条件与Figure 5机制检查，再用多温度和多L数据共同生成Figure 1、3、4；最终物理结论须来自生成模型的独立路径统计，而非LLG参考图冒充模型成果。
