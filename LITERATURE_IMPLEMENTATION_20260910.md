# R2–R5 关键模型与运行代码实现记录

_altermagnetism_LLG · 2026-09-10 · 代码交接，不是正式文献复现通过报告_

---

## 📋 当前结论

本轮新增三套约化磁性模型、双子晶格 AFM-LLB 和 Bauer 弱 RK，补上四篇文献的运行、分析及绘图入口。新增代码及相关回归测试 **49 项通过**，四篇文献的 CPU 短程模拟均完成；另完成 LLB 短程模拟、Laliena 剖面和折叠分支计算。

**尚未达到全部正式复现条件。** 生产开关仍为 `false`；代码会拒绝直接运行未启用的生产协议。没有提交大规模任务，没有改写历史数据、冻结代码或 GUIDE，没有推送 GitHub。

测试证据：[JUnit XML](output/literature_reproduction/implementation_tests_20260910.xml)。

## 🔧 本轮实现范围

| 文献 | 物理模型 | 运行与分析入口 |
| --- | --- | --- |
| Bauer 2011 | 开放链约化 LL、弱 RK | `run.py`、`analyze.py`、`fit.py`、`plot.py` |
| Hirst 2022 | 三维四 Mn 晶胞 ASD；双子晶格 AFM-LLB | `run.py`、`llb_run.py`、`analyze.py`、`wall_analysis.py`、`plot.py` |
| Gomonay 2024 | 双层交换；[100]/[110] 晶胞及混合边界 | `run.py`、`analyze.py`、`plot.py` |
| Laliena 2020/2022 | 单轴 DMI 链；电流 LLG；修正 BVP | `run.py`、`analyze.py`、`plot.py` |

各入口位于 `scripts/literature/<paper_id>/`，所有材料参数继续从 `conf/literature/<paper_id>.yaml` 加载。共享代码为：

- [cell_hamiltonian.py](scripts/core/cell_hamiltonian.py)：明确的晶胞位移键表、周期/开放边界、能量和有效场。
- [reduced_llg.py](scripts/core/reduced_llg.py)：新增预测步归一化的 Heun 分支，保留旧积分方法行为。
- [workflow.py](scripts/literature/workflow.py)：独占创建输出、逐帧 HDF5、完整性标记、配置和代码哈希、Slurm GPU 分配检查。
- [animate.py](scripts/literature/animate.py)：从任意一条保存轨迹生成 GIF，不筛选成功轨迹。

### Bauer

保存完整观察窗内所有独立链的预定输出帧，不按反转成功筛选。分析包含带驻留时间的操作性磁盆事件、首达右删失、Kaplan–Meier、限制平均生存时间及轨迹级 bootstrap。重复反转的已完成间隔均值会明确标为截断偏倚统计量，不充当无条件寿命。

Arrhenius/长度拟合要求输入估计量和参考点来源；拒绝把 RMST 自动换成无限时域寿命。新增 `integrator.py` 实现 Milstein–Tretyakov 式 5.9/5.6，并采用式 4.5/4.6 的离散随机变量；包含旋转噪声的 Itô 修正和扩散导数项。原始无投影公式及可选择的最终归一化分开，归一化是本项目的明确扩展。Heun 保留作共同 Wiener 路径对照，不与弱 RK 的离散弱抽样混淆。该方案对应 Bauer 所引用的误差阶，未声称恢复了作者未公开的源代码细节。[^1][^6]

### Hirst

采用正式发表版 Fig.1/Table I，逐位点实现 `J1/J2/J3/J4` 的 `4/4/4/1` 配位。ASD 保持预测和校正归一化；AFMR 先平衡再旋转，两个阶段不重放相同噪声。畴壁长边端面固定、短边周期，输出 `delta0/a` 与 `pi*delta0/a`，避免混淆 31.2 nm 的定义。磁化率采用原文的面内平均定义。[^2]

AFM-LLB 包含横向、纵向右端项和周期空间交换项；提供横向 AFMR 入口。`me`、磁化率和纵向增强参数与动力学分离。当前示例使用明确标注的文献平衡拟合输入，不声称已经用本项目 ASD 标定。LLB 符号遵循原文式 5/12 的耗散约定，未照搬式 18 的不一致符号。[^2]

### Gomonay

实现正式补充材料的 `S1` 哈密顿量、`S44` 动力学与 `S7b/S9` 色散。GUIDE 的公式编号与该版本不一致，未修改 GUIDE 原文件。使用真实行列式为 2 的旋转超晶胞，避免倾斜畴壁跨越不相容的周期接缝。运动初始化使用 `S28/S31/S19b`，随后无阻尼自由演化，不加入额外 SOT。[^3]

运行配置包含多个方向分别执行的 sinc 激励以及 `J_tilde=0` 控制；分析保留有符号频率、不同子晶格通道和多条 BZ 路径。畴壁分析先将两子晶格插值到共同物理坐标，再求磁化及拟合位置、宽度和相位。

### Laliena

实现修正后的式 13/14，使用反射对称半区间消除平移零模，并通过伪弧长延拓越过折叠点。求解器失败会报错，绝不直接当作临界电流。独立离散 LLG 包含 DMI、Gilbert 阻尼和绝热/非绝热电流力矩；`u` 是有明确符号约定的约化输运系数，不是未标注的正 SI 电流。[^4]

约化网格间距 `0.12992957746478873` 对应文献数值网格 1 nm，与原子尺度 `q0_a` 分开记录。周期轨迹分析包含绕数、跨周期中心追踪和相位梯度加权宽度；对于无孤子的状态，中心/宽度不应作物理解释。[^4]

## 📊 验证和可视化证据

| 检查 | 结果 |
| --- | --- |
| 能量自动微分与解析有效场 | 周期及开放边界通过 |
| Mn₂Au 配位、交换和式、周期平移 | 通过 |
| 投影 Heun、固定端面 | 通过 |
| 非线性离散场线性化 vs 文献色散 | 多波矢通过 |
| Laliena 零电流解析剖面 | 通过 |
| 有限电流 BVP vs 独立离散 LLG | 网格减半，残差约缩小四倍 |
| LLB 横向模长与纵向恢复方向 | 通过 |
| 弱 RK 确定性四阶、精确枚举弱矩 | 通过 |
| 删失、重复事件、存盘、拟合 | 合成数据测试通过 |

相关测试文件为 `scripts/tests/test_remaining_literature.py`、`test_remaining_pipeline.py`、`test_hirst_llb.py`、`test_bauer_weak_rk.py`；测试命令还运行了原有约化文献、流程及 Bauer 回归测试。

可以直接检查：

- [Bauer 短程轨迹图](assets/literature_reproduction/bauer_2011/implementation_smoke_20260910/figures/diagnostics.png) 和 [轨迹动画](assets/literature_reproduction/bauer_2011/implementation_smoke_20260910/animations/trajectory_000.gif)。短程图不包含已验证的反转事件。
- [Mn₂Au 子晶格诊断图](assets/literature_reproduction/hirst_mn2au_2022/implementation_smoke_20260910/figures/diagnostics.png)。
- [Gomonay FFT 软件诊断图](assets/literature_reproduction/gomonay_2024/implementation_smoke_20260910/figures/diagnostics.png)。该 smoke 时长不足以解析物理色散，不是论文结果图。
- [Laliena Gamma=0.89 剖面](assets/literature_reproduction/laliena_crnb3s6_2020/implementation_profile_20260910/figures/profile.png)。
- [Laliena 延拓分支](assets/literature_reproduction/laliena_crnb3s6_2020/implementation_branch_20260910/figures/branch.png) 和 [数值记录](data/literature_reproduction/laliena_crnb3s6_2020/derived/implementation_branch_20260910/analysis.json)。

本轮分支计算的最大采样值为 `Gamma=1.2080093138`，最大 BVP 残差 `9.9933e-7`，与文献 `1.2405` 仍有差异，**不予认证通过**。[^4]

## ⚠️ 仍需处理的内容

这些限制没有被通过 smoke test 的结论覆盖：

1. Bauer 弱 RK 的最终投影、步长和长期统计仍需收敛；长度/温度参考点仍需逐点数字化和误差记录，反转阈值需真实磁盆证据；每条件 500 次完成事件尚未生产。Heun 的共同 Wiener 对照与弱 RK 的弱分布收敛需分别报告。
2. Mn₂Au 位移键表仍需独立逐键审计；未假设 Mn 的内部 `z` 坐标。LLB 温度输入及纵向参数尚未标定，完整热梯度 LLB 畴壁运行协议尚未补齐。
3. Gomonay 旋转盒长、sinc 激励、步长、时长及 Walker 扫描尚未进行正式收敛与文献逐点比较。
4. Laliena 临界值差异待定位；本轮没有擅自修改物理参数去贴合文献数值，也没有完成多场强临界曲线认证。
5. 新代码仅做了 CPU 数值验证；CUDA 路径通过 Slurm 分配检查后可调用，但本轮未实测 GPU，未做生产性能或存储容量评估。
6. 新的共享流式入口尚不支持断点续跑；中断文件保留 `complete=false`，分析入口会拒绝读取，重跑须使用新 run ID。

## ⚙️ 运行方式

在项目根目录运行，使用现有 `zrs-mag` 包，不安装到 base：

```bash
bash ../run_zrs_mag.sh -m scripts.literature.hirst_mn2au_2022.run --protocol smoke --run-id new_smoke_001 --device cpu
bash ../run_zrs_mag.sh -m scripts.literature.hirst_mn2au_2022.llb_run --protocol llb_smoke --run-id new_llb_smoke_001 --device cpu
bash ../run_zrs_mag.sh -m scripts.literature.laliena_crnb3s6_2020.run --protocol profile --run-id new_profile_001 --device cpu
```

`run-id` 不可复用覆盖。正式协议会检查 `numerics.validation_extension.production_enabled`；该开关当前全部关闭，不应只为绕过检查而打开。通过验证后再冻结 YAML，并经 Slurm 使用 `--device cuda`。大批量 CPU 数据生产应提交到 fat 节点；本记录不自动提交任何任务。

## 🔗 来源和实现辅助

新增下载及 SHA-256 见 [来源清单](data/literature_reproduction/source_additions_20260910.json)。旧结果和旧引用来源不改写。

本轮使用 Karpathy Guidelines 控制改动范围；PDF 技能辅助核对原文，matplotlib 技能规范约化坐标图，statistical-analysis 技能约束删失与重采样单位，markdown-mermaid-writing 技能用于这份可追踪交接记录。Scientific Agent Skills 的软件引用列于下方；该引用不作为物理模型正确性的证据。[^5]

[^1]: Bauer et al. (2011). J. Phys.: Condens. Matter 23, 394204. https://doi.org/10.1088/0953-8984/23/39/394204
[^2]: Hirst et al. (2022). Temperature-dependent micromagnetic model of the antiferromagnet Mn2Au. Phys. Rev. B 106, 094402. https://doi.org/10.1103/PhysRevB.106.094402
[^3]: Gomonay et al. (2024). Structure, control, and dynamics of altermagnetic textures, including official Supplementary Materials. https://www.nature.com/articles/s44306-024-00042-3
[^4]: Laliena et al. (2020), updated article and 2022 Author Correction. https://www.nature.com/articles/s41598-020-76903-8 ; https://www.nature.com/articles/s41598-022-06147-1
[^5]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065

[^6]: Milstein, G. N., & Tretyakov, M. V. (1997). Numerical methods in the weak sense for stochastic differential equations with small noise. SIAM J. Numer. Anal. 34, 2142–2167. https://doi.org/10.1137/S0036142996278967 ; author-hosted full text: https://www.maths.nottingham.ac.uk/plp/pmzmt/SINUM97.pdf

