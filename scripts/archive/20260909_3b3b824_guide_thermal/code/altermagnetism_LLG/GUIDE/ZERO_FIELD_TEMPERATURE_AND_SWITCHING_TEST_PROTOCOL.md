# 无外场温度、Néel 温度与自发翻转测试协议

> 服务器目标目录：`/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG/GUIDE`
>
> 适用体系：当前 `Ruo2DoubleLayerHamiltonian`、其 `Jtilde=0` 匹配 AFM 对照，以及普通铁磁链文献基准。
>
> 核心目标：先测出当前有限尺寸离散模型自己的 Néel 温度，再在保持 Néel 有序的温区寻找无场热激活翻转窗口。只有从一个稳定磁态进入另一个稳定磁态并满足驻留和空间结构判据的事件，才计为真实翻转。

## 1. 术语和研究边界

| 规范术语 | 本文定义 | 不应混用的概念 |
|---|---|---|
| 无外场 | Zeeman 外场为 0，SOT/STT 等确定性外加力矩为 0 | 不等于删除交换场、各向异性场或热随机场 |
| 热随机场 | 满足涨落耗散关系的有限温随机项 | 不是人为驱动场 |
| 模型 Néel 温度 `TN(L)` | 当前离散哈密顿量在有限尺寸 `L` 上的赝临界温度 | 不是实验 RuO2 的体相 Néel 温度 |
| committed switch | 进入相反有序盆地并驻留足够时间 | 不是 `n_z` 瞬时过零 |
| critical sign change | 接近临界区时因 `|n|≈0` 引起的符号变化 | 不能计为稳定磁态翻转 |

本阶段所有正式条件固定：

```text
external_field = 0
SOT = 0
field_like_amplitude = 0
damping_like_amplitude = 0
```

当前生成器仍要求 `--drive-t` 参数，因此必须显式传 `--drive-t 0.0`。极化方向和脉冲长度在驱动为零时没有动力学作用，但元数据中仍应记录为未启用。

## 2. 为什么不能直接指定一个绝对翻转温度

无场热激活翻转时间近似满足：

\[
\tau(T,L)=\tau_0(L)\exp\!\left[\frac{\Delta E(L)}{k_BT}\right],
\]

其中 `ΔE(L)` 是该尺寸、边界和翻转机制对应的最小能垒，`τ0` 是尝试时间。翻转概率还取决于观测窗 `tobs`：

\[
P_{\mathrm{switch}}(t_{\mathrm{obs}})=1-\exp[-t_{\mathrm{obs}}/\tau(T,L)].
\]

因此同一温度下，改变尺寸、周期/开放边界、初态或观测时长都可能使翻转概率发生数量级变化。`TN` 只限定有序相是否存在，不直接决定有限时间翻转概率。

推荐工作区间：

- `T/TN < 0.3`：用于低温稳定性；通常难以在直接 LLG 时间尺度内观察无场翻转。
- `0.5 <= T/TN <= 0.8`：优先寻找有序态之间的热激活翻转。
- `0.8 < T/TN < 0.9`：可作高温压力测试，但必须排除临界变号。
- `T/TN >= 0.9`：主要用于临界对照，不作为干净翻转训练条件。
- `T > TN`：Néel 长程序消失，不定义两个稳定 Néel 盆地之间的翻转。

## 3. 当前参数给出的初步能标

当前代码使用：

```text
J1 = 11.1 meV
J2 = 1.88 meV
Jtilde = 0.8 meV
K = 0.047 meV
mu_s = 1 mu_B
```

对应：

```text
J1 / kB ≈ 129 K
K / kB ≈ 0.545 K
kB*T(5 K) ≈ 0.431 meV
K / J1 ≈ 0.00423
```

5 K 高于单格点各向异性能标，但翻转是集体过程。以相干旋转上界作粗略估计：

\[
\Delta E_{\mathrm{coh}}\approx 2L^2K.
\]

| L | `ΔE_coh` | 5 K 下 `ΔE_coh/(kBT)` |
|---:|---:|---:|
| 8 | 约 6.0 meV | 约 14 |
| 16 | 约 24.1 meV | 约 56 |
| 32 | 约 96.3 meV | 约 223 |

真实最小路径可能通过成核和畴壁降低能垒，所以该表不能替代 GNEB/寿命拟合；但它说明 5 K、1–8 ps 下大尺寸周期体系不应频繁无场完整翻转。若出现这种现象，先检查物理和数值实现，不要直接解释为发现。

## 4. 总体测试流程

```text
基础单元测试
  -> 无场零温稳定性
  -> 多尺寸平衡温度扫描
  -> 确定 TN(L)
  -> 按 T/TN 做翻转 pilot
  -> 区分 committed switch 与 critical sign change
  -> 生存分析和能垒拟合
  -> 冻结温度、尺寸、dt 和观测时长
  -> 生成正式无场路径数据
```

任一步失败都停止后续生产，不允许通过提高温度或缩短驻留阈值绕过。

## 5. Stage 0：基础和无场零温检查

服务器入口：

```bash
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project
bash run_zrs_mag.sh -m pytest altermagnetism_LLG/scripts/tests -q
```

随后对每个尺寸运行 `T=0, drive=0`：

- 从 `+z` 和 `-z` 两个 Néel 基态分别启动；
- 至少运行与后续 pilot 相同的物理时长；
- 自旋模长误差满足既定精度；
- 无噪声、有阻尼时总能量单调不增；
- `+z` 和 `-z` 不应自行跨盆地；
- 两个盆地能量应在数值误差内相同。

任一失败均表示哈密顿量、初态、时间步或积分器存在问题，不进行温度扫描。

## 6. Stage 1：确定当前模型的 `TN(L)`

### 6.1 温度和尺寸网格

第一轮粗扫描：

```text
L = {16, 32, 48, 64}
T = {5, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300} K
external_field = SOT = 0
```

发现 Néel susceptibility 峰后，在峰值左右各增加至少 5 个温度点，间隔取粗网格的 `1/2–1/4`。如果 300 K 时仍保持明显长程序，将扫描扩展到 400 和 500 K；如果 20 K 前已经失序，应在 1–20 K 加密。

### 6.2 初态和采样链

每个 `(L,T)` 至少 4 条独立链：

1. 理想 `+z` Néel 态；
2. 理想 `-z` Néel 态；
3. 独立随机态 A；
4. 独立随机态 B。

确定平衡时可以采用较高阻尼加速，但必须满足正确 FDT，并在若干温度验证不同阻尼得到相同平衡分布。最终翻转寿命必须恢复研究所采用的物理阻尼，不能使用高阻尼平衡链代替真实动力学。

### 6.3 平衡判据

同时分析：

- `|n|`、`|n_z|` 和 `n_z^2`；
- Néel susceptibility；
- 总能量和比热；
- Néel 结构因子峰；
- 自相关时间 `tau_int`；
- split-Rhat 和 ESS。

要求：

```text
split-Rhat < 1.05
每个关键量总 ESS >= 1000
每条链 ESS >= 200
burn-in >= 5*tau_int
保存到初态池的间隔 >= 2*tau_int，推荐 5*tau_int
```

有限温下能量不应收敛为常数，只要求其统计分布平稳。

### 6.4 `TN` 的确定方法

对每个尺寸计算 Binder cumulant：

\[
U_4(L,T)=1-\frac{\langle n_z^4\rangle}
{3\langle n_z^2\rangle^2},
\]

以及 susceptibility：

\[
\chi_N(L,T)=\frac{N}{k_BT}
\left(\langle n_z^2\rangle-\langle|n_z|\rangle^2\right).
\]

以不同尺寸 `U4(T)` 的交点作为主要 `TN` 估计，以 susceptibility 峰和结构因子缩放作为辅助。报告 `TN` 的 bootstrap 置信区间，不只报告单一峰值温度。

如果不同尺寸没有稳定交点，只能报告 `TN(L)` 或 crossover，不能声称得到热力学极限 `TN`。

### 6.5 时间步检查

在低温、susceptibility 峰附近和高温各选一个条件，比较：

```text
dt = {0.1, 0.05, 0.025} fs
```

使用耦合布朗增量比较 `dt/dt2/dt4`。接近 `TN` 的随机场更强、相关时间更长，往往需要更小的时间步和更长平衡时间。

## 7. Stage 2：无场翻转温度 pilot

### 7.1 使用归一化温度，不直接猜绝对温度

确定 `TN(L)` 后，每个尺寸运行：

```text
T/TN(L) = {0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.05}
```

`1.05 TN` 仅作为无序相/临界变号负对照；正式翻转条件优先从 `0.5–0.8 TN` 选择。

在 `TN` 尚未得到前，只允许以下绝对温度作探索性 pilot：

```text
L=8:  T={10,15,20,30,40} K
L=16: T={20,30,40,60,80} K
L=32: 先确定 TN(L) 和能垒，不直接指定生产温度
```

这些不是文献给定的翻转温度，也不能直接进入正式数据。

### 7.2 初态

每个条件建立一个“正 Néel 盆地条件初态池”：

- 初态来自已经认证的无场平衡链；
- 要求 `n_z > +0.8*n_eq(T)`；
- 对固定微观初态实验，同一条件所有轨迹使用逐位相同的初态，仅改变 `thermal_seed`；
- 对实验系综，按初态池成员分层抽样，并单独报告初态内和初态间方差。

不要从未经认证的理想零温构型只热化 0.2 ps 后直接开始正式翻转统计。

### 7.3 样本量和观测窗

每个 `(L,T)`：

```text
pilot trajectories = 200
第一层观测窗 = 10 ps
第二层观测窗 = 100 ps，仅对有候选迹象的温度
第三层观测窗 = 1 ns，仅对仍需寿命统计的少数冻结条件
```

如果 10 ps 内零事件，不立即升温到临界区；先把候选温度延长到 100 ps。如果 100 ps 内仍零事件，则给出概率上限。零事件时，95% 上限可近似写为 `3/N`；`N=200` 时约为 1.5%。

CPU 步数示例：

| 物理时长 | `dt=0.1 fs` | `dt=0.05 fs` |
|---:|---:|---:|
| 10 ps | 100,000 | 200,000 |
| 100 ps | 1,000,000 | 2,000,000 |
| 1 ns | 10,000,000 | 20,000,000 |

因此只对筛选出的少数温度运行 100 ps/1 ns，不能对全部温度和尺寸做完整笛卡尔组合。

## 8. 真实翻转的判据

令 `n_eq(T)` 为该温度平衡态的平均 Néel 序参量模长。`committed_switch` 必须同时满足：

1. 全程 `external_field=SOT=0`；
2. 初态处于 `n_z > +0.8*n_eq(T)`；
3. 轨迹跨过 `n_z=0`；
4. 随后达到 `n_z < -0.8*n_eq(T)`；
5. 在负盆地连续驻留至少 `max(5*tau_corr, 10*saved_dt)`；
6. 末段仍处于负盆地；
7. 前后 `|n|` 均保持有序，建议 `|n| > 0.7*n_eq(T)`；
8. 自旋空间图能够归为相干旋转、成核或畴壁传播中的一种，或明确标为新机制。

其他标签：

- `no_crossing`：从未跨过零；
- `crossing_return`：过零后返回正盆地；
- `unresolved/censored`：观测结束前未完成驻留；
- `critical_sign_change`：变号时 `|n| < 0.3*n_eq(T)` 或体系整体失序；
- `committed_switch`：满足上述全部条件。

阈值需由校准集冻结，并在 `0.7/0.8/0.9*n_eq` 与不同驻留时间下做敏感性分析。

## 9. 空间机制检查

周期边界下没有样品边缘，局域反向畴通常需要产生畴壁对。每条候选翻转至少保存和分析：

- 局域 Néel `n_z(x,y,t)`；
- 反向团簇面积及持续时间；
- 畴壁总长度/密度；
- 结构因子；
- 空间方差和相关长度；
- 全局能量及交换、各向异性能量分量。

机制建议：

- `near_coherent`：绝大多数格点在窄时间窗内同步旋转，空间方差与畴壁密度低；
- `nucleation`：先出现超过预注册最小面积和持续时间的局域反向团簇；
- `domain_wall_pair`：周期边界下形成两条或闭合畴壁并传播；
- `critical_disordering`：全局和局域有序度同时崩溃，随后随机选择符号。

`critical_disordering` 不能用来证明两个稳定有序态之间存在热激活翻转。

## 10. 温度选择的最终 Go/No-Go 标准

某个 `(L,T,tobs)` 只有同时满足以下条件，才可进入正式无场翻转数据集：

```text
0.5 <= T/TN(L) <= 0.85
0.20 <= P_committed_switch <= 0.80
critical_sign_change fraction <= 0.05
unresolved fraction <= 0.20（正式数据建议 <=0.10）
dt/2 与 dt/4 的翻转概率差 <=0.03
dt/2 与 dt/4 的中位首达时间差 <=5%
最大两尺寸的机制比例差 <=0.05，或明确标为尺寸外推测试
```

如果在 `T<=0.85TN`、100 ps 内仍没有足够事件：

1. 先延长到 1 ns；
2. 计算 GNEB/最小能量路径或用寿命拟合估计 `ΔE`；
3. 检查更小尺寸是否仍包含所需空间机制；
4. 保留“无翻转”作为热稳定性结果；
5. 使用 Bauer 普通铁磁链作为无场自发翻转的模型验证基准；
6. 交错磁部分改用无场非平衡弛豫路径，不为了制造事件而加入外场。

如果只有 `T>=0.9TN` 才频繁变号，则本体系在直接模拟窗口内没有干净的无场稳定态翻转条件，不能把临界变号包装成多稳态路径。

## 11. 生存分析和能垒拟合

未在观测窗内翻转的轨迹是右删失样本，不能删除。每个温度至少报告：

- Kaplan–Meier 生存曲线；
- committed-switch 概率及 Wilson 95% CI；
- 中位首达时间及 bootstrap CI；
- crossing-return、critical-sign-change 和 unresolved 比例；
- 机制分层后的首达时间分布。

选择至少 4 个保持相同翻转机制的温度拟合：

\[
\ln \tau=\ln\tau_0+\frac{\Delta E}{k_BT}.
\]

如果温度升高后机制从相干旋转转为成核/畴壁，不得用一条 Arrhenius 直线拟合全部温度。

## 12. 数据目录和输出结果

建议目录：

```text
data/zero_field_temperature_v1/
  metadata/
  tn_scan/
    L16/
    L32/
    L48/
    L64/
  equilibrium_pools/
  reversal_pilot/
  reversal_production/
  certificates/

assets/zero_field_temperature_v1/
  tn/
  survival/
  mechanisms/
  convergence/
```

必须生成的结果：

```text
neel_order_vs_temperature.png
susceptibility_vs_temperature.png
binder_cumulant_crossing.png
specific_heat_vs_temperature.png
autocorrelation_and_ess.png
dt_convergence_near_tn.png
switch_probability_vs_T_over_Tn.png
survival_curves.png
first_passage_distribution.png
mechanism_fraction_vs_temperature.png
representative_spacetime_paths.png
temperature_selection_decision.json
```

`temperature_selection_decision.json` 至少记录：选中的温度、`TN(L)` 及置信区间、观测时长、翻转概率、临界变号比例、时间步收敛、尺寸收敛和最终 `PASS/FAIL`。

## 13. 现有命令与需要补充的脚本

### 13.1 现有生成器可做的无场 smoke test

以下命令只验证 `drive=0` 数据通路。由于默认 0.2 ps 初态准备尚未认证，它不能直接生成正式数据：

```bash
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project
bash run_zrs_mag.sh altermagnetism_LLG/scripts/datasets/generate_v3_gomonay_shard.py \
  --mode pilot \
  --protocol altermagnetism_LLG/data/standard_v3/altermagnet_gomonay2024/protocol.candidate.yaml \
  --output altermagnetism_LLG/data/zero_field_temperature_v1/smoke/L16_T20K_seed0.h5 \
  --size 16 \
  --temperature-k 20 \
  --drive-t 0.0 \
  --condition-id 2000 \
  --shard-index 0 \
  --paths 32 \
  --dt-fs 0.05 \
  --duration-ps 10 \
  --saved-dt-fs 2.5 \
  --equilibration-ps 0.2 \
  --pulse-ps 0.5 \
  --alpha 0.01 \
  --base-seed 20260910 \
  --device cpu
```

### 13.2 正式测试仍需补充

仓库目前没有完整的 `TN` 多链扫描、Binder 分析和无场生存统计脚本。建议后续实现：

```text
scripts/validation/scan_neel_temperature.py
scripts/validation/analyze_neel_temperature.py
scripts/datasets/build_equilibrium_initial_pool.py
scripts/datasets/generate_zero_field_reversal.py
scripts/validation/analyze_zero_field_switching.py
```

这些名称是实施建议，不是当前已经存在的命令。正式生产前，脚本必须支持：多初态链、Rhat/ESS、右删失、`n_eq(T)` 阈值、空间机制、耦合布朗时间步比较及 HDF5 元数据证书。

## 14. CPU Slurm 提交规范

所有数据任务使用 CPU，提交到 `fat` 分区，任务名固定为 `zrs_data`：

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

# 数据生成命令必须显式包含 --drive-t 0.0 --device cpu
```

提交：

```bash
sbatch altermagnetism_LLG/slurm/YOUR_ZERO_FIELD_JOB.sbatch
squeue -u "$USER" -n zrs_data
```

只确认一次任务为 `PD` 或 `R`，之后不轮询、不创建监督任务。完成后由用户检查日志、证书、HDF5 完整性和 SHA256。

## 15. 文献依据和可以声称的范围

1. [Gomonay et al., npj Spintronics 2, 35 (2024)](https://www.nature.com/articles/s44306-024-00042-3)：给出交错磁纹理和动力学模型，但没有给出当前参数集的有限温无场翻转温度；文章也指出 RuO2 体相磁序仍存在讨论。
2. [Yershov et al., Phys. Rev. B 110, 144421 (2024)](https://journals.aps.org/prb/pdf/10.1103/PhysRevB.110.144421)：对相关局域矩交错磁 checkerboard 模型进行随机 Landau–Lifshitz 有限温模拟，给出 `TN`、尺寸和时间步收敛方法；其各向异性参数与当前模型不同，不能直接复制温度。
3. [Weißenhofer and Marmodoro, Phys. Rev. B 110, 094427 (2024)](https://arxiv.org/abs/2405.20921)：使用从头算参数和随机 LLG 研究 RuO2 有限温磁振子输运，不是无场自发翻转研究。
4. [Semenov et al., JMMM 489, 165457 (2019)](https://arxiv.org/abs/1806.11130)：研究普通 AFM 纳米结构的无场热涨落、自发 Néel 矢量翻转和保持时间，支持用能垒与寿命而非只用 `TN` 选温度。
5. [Weiss et al., arXiv:2301.02006](https://arxiv.org/abs/2301.02006)：报告接近自旋重取向临界区的 AFM 自发超快翻转，但材料和双阱机制不同，不能直接作为当前温度参数。

当前可以声称的是“按已发表方法测定模型的有限温有序性并寻找无场路径分布”；在完成上述测试前，不能声称当前 Gomonay 参数集已经具有某个确定的无场翻转温度。
