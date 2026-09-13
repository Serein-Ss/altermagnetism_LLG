# 文献复现、普通磁体/AFM 对照与模型测试数据计划

> 目标：先用公开文献中的可观测量认证求解器和体系实现，再为普通铁磁体、常规 AFM 对照和交错磁主体系生成相互独立的标准路径数据。文献复现数据、模型训练数据、最终物理研究数据三者用途不同，不能混为一批。

> 本阶段边界：所有主数据均设 `external_field=0`、`SOT=0`，只用 CPU 在 `fat` 分区生成。SOT 是以后单独开展的受驱扩展，不属于当前标准数据集。

## 1. 总体研究逻辑

本工作的核心不是“首次用 LLG 生成轨迹”，而是建立一个可以学习条件随机 LLG **路径分布**的方法，并验证它能否：

1. 在相同初态和宏观条件下采样多条有正确权重、时间顺序和空间结构的随机演化路径；
2. 同时重现终态概率、首达时间、返回事件和相干/成核/畴壁等路径机制，而不只拟合平均轨迹；
3. 在普通磁体、常规 AFM/匹配 AFM 对照和交错磁上使用统一的数据接口；
4. 从小尺寸训练并评估到大尺寸的泛化，但只在物理相关长度、机制和有限尺寸误差允许的范围内声称成功。

因此数据按三层组织：

- **层 A：求解器认证**。解析结果或论文曲线，不能用于展示生成模型性能。
- **层 B：跨体系模型基准**。冻结条件下的大样本随机路径，用于训练/验证/测试。
- **层 C：交错磁物理研究**。在模型通过层 B 后，研究晶向、路径机制和可靠性；不能提前用测试结论选择训练条件。

## 2. 四个体系各自承担什么角色

| 体系 | 文献锚点 | 作用 | 当前代码状态 |
|---|---|---|---|
| 非相互作用磁矩 | Nishino & Miyashita 2015 | 热浴、FDT、积分器和精确平衡分布认证 | 已有简化实现，需按论文完整时长复现 |
| 一维普通铁磁链 | Bauer et al. 2011 | 热激活反转、边缘成核、畴壁传播和寿命标度 | 已有脚本，完整文献时长计算量很大 |
| Mn2Au 常规 AFM | Hirst et al. 2022 | 独立的文献 AFM：平衡序参量、Néel 温度、AFMR、畴壁 | 当前未实现；邻接和 3D 后端完成前为 BLOCKED |
| d-wave 交错磁 | Gomonay et al. 2024 | 复现交错磁色散/晶向指纹；随后研究无场路径分布 | 哈密顿量已有；旧 SOT 扫描不是该文献复现，也不进入本阶段数据 |

另保留 `Gomonay Hamiltonian + Jtilde=0` 作为**匹配 AFM 消融对照**。它控制住晶格、尺寸、积分器、无场初态和热浴，只去掉交错项，适合立即对照训练；但它不是 Mn2Au，也不能写成“复现了一个独立 AFM 材料”。

## 3. 层 A：必须先完成的文献复现

### A1. 解析热浴基准：Nishino–Miyashita 2015

文献设置：独立磁矩、`h=2, M=1, alpha=0.05, dt=0.005, N=1000`，80,000 步，其中前 40,000 步平衡、后 40,000 步测量。

执行要求：

- 数值取样温度可选 `T={0.5,1.0,...,6.0}`；这是为了平滑描出文献曲线的数值网格，不声称是论文逐点原始数据。
- 每个温度至少 10 个独立 run；单个 run 已含 1000 个非相互作用磁矩。
- 在 `dt=0.005` 复现论文设置，并用 `0.0025, 0.00125` 做收敛验证；物理总时长保持 400 不变。
- 输出 `m_z(T)`、精确 Langevin 曲线、`P(cos theta)`、二阶矩、自相关时间、ESS、时间步偏差和置信区间。

通过标准见 `LLG_DATA_TRUSTWORTHINESS_AUDIT.md`。当前 `generate_benchmark_hdf5.py` 的默认短流程与每条件 20 条仅是 pilot，不能直接标记为论文复现。

文献：[Phys. Rev. B 91, 134411](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.91.134411)。

### A2. 普通铁磁路径基准：Bauer 2011 开边界链

论文模型和公开参数：`L=100`、最近邻 `J=1`、易轴各向异性 `K/J=0.1`、`kBT/J=0.11`、阻尼 `lambda=0.1`、开边界；论文给出的观察时长为 `750000 ħ/J`。关注自发反转、从边缘成核的畴壁和畴壁传播，以及寿命随长度/温度的关系。

当前脚本：

```bash
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project
bash run_zrs_mag.sh altermagnetism_LLG/scripts/validation/validate_bauer2011_chain.py
bash run_zrs_mag.sh altermagnetism_LLG/scripts/datasets/generate_bauer2011_paths.py \
  --profile accelerated_pilot --device cpu
```

`accelerated_pilot` 只验证数据通路和空间分类；正式复现必须使用：

```bash
bash run_zrs_mag.sh altermagnetism_LLG/scripts/datasets/generate_bauer2011_paths.py \
  --profile published_fig2 --device cpu
```

注意：论文没有给出当前代码采用的数值 `dt=0.02`。它是数值选择，必须补做 `dt={0.02,0.01,0.005}` 收敛。若保持总时长，单条轨迹分别为 37.5、75、150 million 步，不能把十条短轨迹当作寿命统计。

分两级完成：

1. **定性复现**：每个 dt 至少 10 条完整或足以观察一次反转的轨迹，展示边缘成核和畴壁传播的时空图。
2. **定量复现**：每个长度/温度条件至少获得 500 个反转事件；没有反转的固定时长轨迹作为右删失样本，用 Kaplan–Meier/生存模型估计寿命，不能删除。

至少输出：

- 论文参数表与当前单位换算；
- `m_z(x,t)` 时空图、畴壁位置和速度；
- 势垒与理论畴壁能比较；
- 反转寿命分布、Arrhenius 图、寿命随链长的关系；
- 时间步收敛和事件数/置信区间。

文献：[Bauer et al., J. Phys.: Condens. Matter 23, 394204 (2011)](https://arxiv.org/abs/1010.4730)。

### A3. 常规 AFM 文献基准：Mn2Au

Hirst et al. 给出了足够完整的原子自旋参数，适合做独立 AFM 对照：

| 参数 | 数值 |
|---|---:|
| `a` | 3.330 Å |
| `c` | 8.537 Å |
| `mu_s` | 3.8663 μB |
| `J1` | -5.3422 mRy |
| `J2` | 0.6484 mRy |
| `J3` | -0.6341 mRy |
| `J4` | -6.8986 mRy |
| 同亚晶格 `J0` | 2.5934 mRy |
| 异亚晶格 `J0` | -30.8040 mRy |
| z 向各向异性 | -0.0090 mRy |
| [100] 应变各向异性 | 0.0004 mRy |

论文的每个格点有相应的多壳层邻居计数，平衡模拟使用 `30×30×30` 晶胞、周期边界和 `lambda=1` 加快平衡。AFMR 使用 `lambda={0.05,0.01,0.001}`，将自旋从易轴面外旋转 20° 后释放。

当前仓库尚无 Mn2Au 精确 3D 晶格和邻接模板，因此现阶段状态必须写为 `blocked_parameter_and_implementation_audit`。开始数据生成前必须：

1. 从论文晶体结构构建 3D 晶胞、亚晶格和每一交换壳层；
2. 用邻居计数、键长/方向、能量有限差分和对称性测试认证；
3. 确认 mRy→J、磁矩和有效场单位；
4. 对论文未报告的时间步做稳定性/收敛扫描，不能自称采用“文献 dt”；
5. 扩展当前主要面向 2D `[B,S,X,Y,3]` 的模型接口，或明确把 Mn2Au 仅用于求解器验证。将 3D 数据切成 2D 片层不等价于复现论文。

复现顺序：

- **平衡序参量**：`30^3` PBC、`lambda=1`，复现 `m_e(T)` 和纵向磁化率，拟合 `T_N≈1335 K`；温度网格在临界区加密，实际平衡时长由 Rhat/ESS 决定。
- **AFMR**：在 `T={300,1000,1200} K` 平衡后整体倾斜 20°，运行至少覆盖论文约 1 ps 的衰减振荡并留出频率分辨率；同时做更长窗口检查 FFT 分辨率，比较阻尼拟合和理论 AFMR 关系。
- **畴壁**：在 0 K 复现约 `31.2 nm` 的畴壁宽度，作为空间实现的独立检查。

至少输出：`me_vs_T.png`、`susceptibility_vs_T.png`、`afmr_trace_and_fft.png`、`afmr_frequency_vs_T.png`、`domain_wall_profile.png` 以及对应 JSON/CSV 数值。

文献：[Hirst et al., Phys. Rev. B 106, 094402 (2022)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.094402)。

### A4. 交错磁文献基准：Gomonay 2024

先复现哈密顿量本身的结论，不用当前 0.78 T 脉冲代替文献验证：

- Supplementary Eq. S1/Table I 对应的交换、交错项、各向异性和晶格常数；
- 基态和 d-wave magnon splitting；
- 不同晶向/波矢下的色散分裂及对称性；
- `Jtilde=0` 后分裂消失的消融对照；
- 时间步、尺寸和 FFT 窗口收敛。

现有命令：

```bash
bash run_zrs_mag.sh altermagnetism_LLG/scripts/validation/validate_spinwave.py
bash run_zrs_mag.sh altermagnetism_LLG/scripts/visualization/plot_literature_validation.py
```

输出必须把论文曲线/数值与模拟结果放在同一坐标定义和单位下，并给出频率峰拟合误差。当前的阻尼型 SOT 脉冲是后续数值研究条件，不是 Gomonay 论文已经提供的动力学结果。本阶段只保留 `SOT=0` 的色散和无场弛豫。

文献：[Gomonay et al., npj Spintronics 2, 35 (2024)](https://www.nature.com/articles/s44306-024-00042-3)。

## 4. 层 B：模型基准数据具体生成什么

只有层 A 和可信度审计通过后才生成。各体系单独保存、单独证书，但共享统一 HDF5 字段。

### B1. 普通铁磁链数据

目的：检验模型是否能学习局域成核、畴壁传播、首达时间和右删失概率。

- 训练尺寸：`L={40,60,80}`；验证 `L=80` 的独立初态/seed；尺寸外推测试 `L=100`。
- 条件：围绕文献 `K/J=0.1, kBT/J=0.11, lambda=0.1`，只做少量预注册温度，例如 `T/J={0.09,0.11,0.13}`；其中 0.11 为文献锚点，其他是模型插值/外推条件。
- 每个冻结条件至少 500 条轨迹；若事件稀少，增加轨迹数或观测窗，不进行“只保留已翻转”抽样。
- 训练/验证/测试按 8:1:1 分割 seed 与初态；`L=100` 另作为完全独立的尺寸外推测试，不进入 8:1:1。
- 固定时长由 pilot 的反转时间分布决定，至少覆盖目标条件 `Q99(t_commit)`；记录删失。

### B2. AFM 数据分成两个版本

**可立即生成的匹配对照**：Gomonay 哈密顿量中令 `Jtilde=0`，其余温度、尺寸、无场初态和 seed 与交错磁一一配对，且 `external_field=SOT=0`。每个条件正式建议 1000 条，pilot 128–200 条。它回答“在相同无场条件下，去掉交错项后路径分布如何变化”。

**独立文献 AFM**：Mn2Au。先做 3D 后端和文献复现；随后可生成有限温弛豫数据：

- 温度 `T={300,600,900,1200} K`；
- 初始倾角 `5°,10°,20°`，阻尼 `0.01,0.05`；
- 每条件 pilot 200 条、正式至少 500 条；
- 训练尺寸必须经有限尺寸审计后选定，可从 `16^3,24^3` 开始，`30^3` 作为论文尺寸测试；
- `30^3` 是否能纳入当前模型取决于 3D 等变模型和显存，不能用二维模型直接宣称泛化。

这里的路径是热浴下受扰 AFM 的弛豫/振荡分布，不强行把所有体系都改造成同一种“翻转”任务。

### B3. 交错磁无场主数据

无场下应分别生成三类数据，三类数据不能混合统计：

1. **平衡涨落集**：从认证 5 K 平衡初态池抽样，`field=SOT=0`。用途是学习同一盆地内的有限温路径分布、动态关联和谱；不要求出现翻转。
2. **热激活亚稳态集**：从文献哈密顿量的亚稳态出发，`field=SOT=0`，用长固定窗口记录自发首达、返回和删失。若 5 K 事件概率极低，诚实报告稀有事件，不人为加场。
3. **无场非平衡弛豫集**：对所有轨迹使用完全相同、预注册的倾斜 Néel 初态、局域反向核或畴壁初态，随后无场释放。能量来自初态而非外场；不同热 seed 给出条件路径分布。

每个初态类型单独设置 `condition_id`。为了证明“同一个初态的多路径”，条件内的微观初态必须逐位相同，只改变 `thermal_seed`；从平衡池抽样的实验系综另行统计。

在 Standard V3 候选协议基础上，先完成：

- `dt={0.1,0.05,0.025} fs`；
- 保存间隔 `{2.5,5,10} fs`；
- 总时长 `{1,2,4,8} ps`；
- 尺寸 `L={16,32,48,64,96,128}`；
- 温度诊断 `{0,1,2.5,5,10} K`；
- 初态族 `{平衡池, 固定小角度倾斜, 固定局域反向核, 固定畴壁}`；
- 全部 `external_field=0, SOT=0`。

这些都是**收敛/诊断候选网格**，不是要求做所有笛卡尔组合。先做单因素 pilot，冻结最小合格 `dt`、`saved_dt`、`T_fixed` 和最小无有限尺寸偏差的盒长，再生成正式数据。

每个候选条件 128–200 条；冻结后，每个可能出现多机制或稀有跃迁的正式条件推荐 1000 条，纯盆地内涨落条件可 500 条。0 K 是物理控制，不一定进入生成模型训练。

正式数据中必须保存：完整空间自旋、能量分量、局域 Néel 场、全局序参量、结构因子、畴壁密度、初态类型和路径事件；`drive_mask` 保留为全零字段以维持统一接口。机制标签可以离线更新，但原始轨迹不可覆盖。

## 5. 一个初态需要多少条随机轨迹

生成模型不是用单条样本“代表”一个初态；每次采样得到一条路径，多次采样的经验分布才表示条件路径分布。

- pilot：每条件 128–200 条，用于判断是否多峰和估计概率；
- 二项终态概率的 95% 最坏情况误差约为 `1.96*sqrt(0.25/N)`：`N=100` 为 ±9.8%，`N=400` 为 ±4.9%，`N=1000` 为 ±3.1%；
- 因此正式阈值条件推荐 1000 条，若要 ±2% 则约需 2401 条；
- 稀有事件概率 `p` 至少要看到约 100 个事件才适合比较机制，粗略需要 `N >= 100/p`；
- 评价生成模型时，每个测试条件至少采样与真实集相同数量，并用 bootstrap 比较概率、首达时间和机制分布。

不同轨迹应同时变化热噪声 seed；若研究“固定微观初态的条件分布”，初态必须完全相同。若研究有限温实验系综，则先从认证初态池抽样，再条件化或边缘化，二者必须分别报告。

## 6. 数据格式、切分和模型测试结果

### 6.1 统一最小格式

```text
spins              [N, T, S, X, Y, 3] 或系统对应空间维数
time               [T]
condition           温度、阻尼、初态类型、材料/哈密顿量参数；本阶段外场/SOT恒为0
drive_mask          [T] 或 [N,T]，本阶段全零，为以后受驱扩展保留
energy_components  [N,T,C]
order_parameters   [N,T,Q]
initial_state_id    [N]
thermal_seed        [N]
split               train/val/test/ood_size
metadata            单位、边界、dt、saved_dt、commit、协议哈希
```

变长轨迹若仅用于 pilot 可以用有效长度和 mask；正式训练采用冻结的固定物理时长和统一保存间隔，便于批处理。未决轨迹保留并标为 censored。

### 6.2 8:1:1 切分

- 在生成前按 `initial_state_id` 和 seed block 哈希切分，不能生成后随机逐轨迹打散；
- 相同条件的 80%/10%/10% 用于训练/验证/同分布测试；
- 留出温度或初态幅度作为条件外推测试；无场晶向差异只有在初态/纹理方向有物理定义时才比较；
- 最大尺寸完全留出作为尺寸 OOD；
- 文献复现数据与模型测试数据物理参数可以相同，但随机种子、初态池和文件必须独立。

### 6.3 模型必须报告的结果

不能只展示几条漂亮动画。每个体系至少报告：

1. 单时刻序参量分布和终态盆地概率；
2. 首达/承诺时间分布，包括右删失；
3. crossing-return 概率；
4. 相干、成核、畴壁传播等机制比例；
5. 时空相关函数、结构因子和动态谱；
6. 能量/模长/对称性/PBC 等物理违规率；
7. 不同采样数下分布指标是否稳定；
8. 同分布、条件插值、条件外推和尺寸外推分别报告；
9. 与直接 LLG 的计算成本和统计误差作等精度比较。

普通铁磁链重点看无场畴壁和寿命；AFM 重点看无场 Néel 弛豫/AFMR；交错磁重点看无场动态关联、纹理弛豫以及去掉 `Jtilde` 后的差异。不要要求所有体系用同一标量结论。

## 7. 需要多少物理模拟时间和服务器时间

### 7.1 已知物理时长/步数

| 任务 | 物理或无量纲时长 | 步数 |
|---|---:|---:|
| Nishino 论文设置 | 400 | 80,000 (`dt=0.005`) |
| Nishino dt/2 | 400 | 160,000 |
| Nishino dt/4 | 400 | 320,000 |
| Bauer published | `750000 ħ/J` | 37.5 M (`dt=0.02`) |
| Bauer dt/2 | 同上 | 75 M |
| Bauer dt/4 | 同上 | 150 M |
| 交错磁 4 ps | 4 ps | 40k/80k/160k，对应 `dt=0.1/0.05/0.025 fs` |
| 交错磁 8 ps | 8 ps | 80k/160k/320k |
| Mn2Au AFMR | 至少覆盖论文约 1 ps；建议加长做频率收敛 | 步数由 dt 审计决定 |
| Mn2Au 平衡 | 不预设固定短时长 | 直到 Rhat/ESS 达标 |

### 7.2 CPU-only 的 wall-clock 估算

本轮所有数据生成显式使用 `--device cpu`，不申请 GPU。CPU 型号、线程数、批大小、晶格尺寸和 I/O 会让实际耗时相差数倍。正式提交前对每种 `(system,size,batch,dt)` 做 10,000 步计时，输出 `benchmark_runtime.json`。估算：

```text
single_job_seconds = measured_seconds_per_10000_steps * total_steps / 10000
campaign_walltime  = max(single_job_seconds per shard wave) * shard_waves * 1.3
shard_waves        = ceil(number_of_shards / concurrent_jobs)
```

其中 1.3 是调度和 I/O 裕量。Slurm `--time` 至少取单 shard 估算的 1.5 倍；首次只提交一个 pilot shard，确认峰值内存、文件大小和写盘速度后再开 array。

数据体积也应先估算：

```text
bytes ≈ N * T_saved * S * spatial_sites * 3 * bytes_per_float
```

再加能量和元数据。自旋建议 float32、时间/关键标量可 float64；HDF5 采用按轨迹和短时间块的 chunk、gzip/lzf 压缩。不要通过降低物理保存频率来迁就磁盘，必须先通过保存间隔收敛。

### 7.3 统一的 fat 节点提交头

所有数据任务使用任务名 `zrs_data`，提交到 `fat` 分区：

> **不要提交现有的 `slurm/generate_v3_gomonay_pilots.sbatch` 作为本阶段任务。**该文件是旧的受驱动扫描，包含 `0.60–0.90 T`、任务名 `zrs-mag`，虽然计算设备已经是 CPU，但不符合现在的无场协议。应在无场初态和冻结协议确定后，新建 `zrs_data` 脚本，并向现有生成器显式传 `--drive-t 0.0 --device cpu`。

```bash
#!/usr/bin/env bash
#SBATCH --job-name=zrs_data
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=altermagnetism_LLG/slurm/logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME CUDA_VISIBLE_DEVICES

# 现有生成器必须显式传 --device cpu。
# 无场交错磁生成器当前仍要求 --drive-t 参数，必须传 --drive-t 0.0。
```

提交方式：

```bash
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project
sbatch altermagnetism_LLG/slurm/YOUR_DATA_JOB.sbatch
squeue -u "$USER" -n zrs_data
```

只确认一次状态为 `PD` 或 `R`；之后不轮询、不自动续投、不创建监督任务。完成后由用户检查 `.out` 日志、HDF5 证书和 SHA256。

## 8. 推荐提交顺序

1. 跑全量测试、数据完整性和百万样本热噪声统计。
2. 完成 Nishino 精确基准和几何参照积分器比较；失败则停止所有有限温生产。
3. 做交错磁无场 5 K 的基态/平衡池/亚稳态/固定非平衡初态控制、温度扫描以及 `dt/dt2/dt4` 耦合噪声测试。
4. 完成 Gomonay 色散复现；冻结交错磁哈密顿量实现。
5. 跑 Bauer 定性 pilot；计时后再决定 500 事件定量任务的并行规模。
6. 实现 Mn2Au 3D 邻接并复现 `m_e(T)`/无场 AFMR；在此之前只生成无场匹配 `Jtilde=0` AFM 对照。
7. 完成保存间隔、尺寸和总时长扫描，冻结每个体系协议。
8. 先各生成 128–200 条 pilot 并出证书；全部通过后才生成 500–1000 条/条件的正式数据。
9. 数据按 8:1:1 和尺寸/条件留出拆分，生成 SHA256 清单；此后不再依据测试结果调整训练条件。
10. 最后开始模型训练，并用直接 LLG 的路径分布而非单条轨迹作对照。

## 9. Go/No-Go 条件

只有同时满足以下条件，才允许把某个体系写成“标准数据集”：

- 至少一个解析或文献可观测量通过；
- 哈密顿量/邻接/单位有逐项证书；
- FDT、随机场和积分器弱收敛通过；
- 初态池 Rhat/ESS 通过；
- `dt/saved_dt/size/duration` 已冻结；
- 0 K 基态、有限温无场平衡、亚稳态和固定非平衡初态控制自洽；
- 样本量能给出所声明精度的置信区间；
- 切分无泄漏、文件哈希齐全；
- 对未决/右删失轨迹没有选择性删除。

在此之前统一使用 `pilot`、`candidate` 或 `debug`，不要使用 `production_certified`。
