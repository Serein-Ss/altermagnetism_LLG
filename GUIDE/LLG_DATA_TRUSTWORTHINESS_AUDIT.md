# LLG 轨迹数据可信度审计与重新生成判据

> 适用目录：`/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG`
>
> 目的：在任何生成模型训练之前，证明数据确实来自正确的物理模型、正确的随机 LLG 离散化，并已在时间步、保存间隔、体系尺寸和观测时长上达到所需精度。本文不是“跑通脚本”清单，而是数据是否可以进入论文和模型训练的放行标准。
>
> 当前结论：仓库已有数据只能视为开发/试验数据，不能因为动画合理、能量下降或单元测试通过就标记为正式数据。若下述任一硬门槛失败，应保留原数据用于追溯，但不得混入正式训练集，并按冻结后的协议重新生成。

## 1. 首先明确“可信”的四个层次

1. **实现正确**：哈密顿量、邻接、有效场、随机场、阻尼和驱动力的单位与符号正确。
2. **数值正确**：积分器在选定时间步下给出收敛的统计量，人工归一化没有造成可见的弱偏差。
3. **物理正确**：至少复现一项与当前体系对应的已发表文献可观测量，并通过无驱动、零温等物理控制。
4. **数据正确**：每条轨迹的元数据、随机种子、切分、完整性和路径标签可追溯，且没有训练/验证/测试泄漏。

四层必须全部通过。单元测试只覆盖第一层的一部分。

## 2. 审计原则和目录约定

- 不覆盖任何已有 HDF5、NPZ 或 JSON；审计输出统一写入 `data/audit/<日期_提交号>/`。
- 原始轨迹为只读；若失败，写入 `rejected_manifest.csv`，不要删除。
- 每次审计先记录 Git 提交号、环境、CPU/内存、脚本哈希、协议哈希和输入文件 SHA256。
- 参数分为三类并写进元数据：`literature`（文献直接给出）、`derived`（由文献量换算）、`numerical`（时间步、保存间隔等数值选择）。禁止把数值扫描参数称作文献参数。
- 有限温下能量不会收敛到常数。“达到稳态”应由分布平稳性、链间一致性和自相关时间判断，而不是末段能量斜率接近零。
- 训练/验证/测试按条件组和初态池分组后再作 8:1:1 划分；同一初态、同一随机种子派生轨迹不得跨集合。

服务器上先建立一次审计快照：

```bash
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project
git -C altermagnetism_LLG rev-parse HEAD
git -C altermagnetism_LLG status --short
bash run_zrs_mag.sh -m pytest altermagnetism_LLG/scripts/tests -q
```

应保存：

```bash
mkdir -p altermagnetism_LLG/data/audit
git -C altermagnetism_LLG rev-parse HEAD > altermagnetism_LLG/data/audit/audited_commit.txt
bash run_zrs_mag.sh -m pip freeze > altermagnetism_LLG/data/audit/python_environment.txt
lscpu > altermagnetism_LLG/data/audit/cpu_environment.txt
free -h > altermagnetism_LLG/data/audit/memory_environment.txt
```

说明：当前仓库的测试命令是现有命令；本机 `cgdit` 环境已通过 `test_unified_llg.py` 和 `test_altermagnet_dynamics.py` 共 13 项测试，但服务器仍须在实际生成环境重跑。

## 3. Gate 0：文件、元数据和切分完整性

### 3.1 每条轨迹必须具备的字段

至少保存：

- `trajectory_id`、`system_id`、`condition_id`、`initial_state_id`；
- 独立的 `thermal_seed`，以及生成初态池的 `initial_seed`；
- Git commit、协议文件 SHA256、代码版本；
- 哈密顿量全部参数、单位和参数来源；
- 晶格尺寸、维数、边界条件、邻接模板版本；
- 温度、阻尼、旋磁比、磁矩、驱动形式、驱动方向、脉冲起止时间；
- 内部积分步长 `dt`、保存间隔 `saved_dt`、总时长、积分器名称；
- 完整自旋轨迹，至少为 `[time, sublattice, spatial..., 3]`；
- 能量分量、全局/局域序参量、结构因子或其可重算所需信息；
- 完成状态和异常标记，不能悄悄丢弃未翻转或未收敛轨迹。

### 3.2 硬判据

- 任意 NaN/Inf：失败。
- 任意重复 `trajectory_id` 或重复 `(initial_state_id, thermal_seed, condition_id)`：失败。
- 任意自旋模长误差 `max |||s||-1| > 1e-5`（float32）或 `1e-10`（float64）：失败并排查；不要仅靠再次归一化掩盖。
- 时间轴不严格递增、帧数与协议不符、末帧缺失：失败。
- 8:1:1 切分中存在相同初态池成员或相同随机种子的派生样本：失败。
- HDF5 中的参数与协议文件不一致：失败。

现有脚本可用于基础证书：

```bash
bash run_zrs_mag.sh altermagnetism_LLG/scripts/validation/certify_benchmark_dataset.py
bash run_zrs_mag.sh altermagnetism_LLG/scripts/validation/preflight_standard_v3.py
```

`preflight_standard_v3.py --require-production-ready` 只有在协议冻结和证书齐备后才应通过；不能为了启动生产而绕过它。

## 4. Gate 1：哈密顿量和有效场逐项核对

对每个体系单独建立 `parameter_provenance.csv`，列出：代码变量、物理符号、数值、单位、原文表/公式/页码、换算公式、核对人。

必须做的自动测试：

1. **有限差分梯度**：对随机单位自旋微扰，验证
   \[
   \mathbf B_i^{\mathrm{eff}}=-\frac{1}{\mu_i}\frac{\partial \mathcal H}{\partial \mathbf s_i}.
   \]
   每个能量项分别测试，中心差分相对误差建议小于 `1e-5`（float64）。
2. **邻接计数**：周期边界下每壳层邻居数、键是否重复计数、亚晶格配对必须与论文模板一致。
3. **对称性**：整体自旋反演、晶格平移、论文所要求的点群操作后，能量和相应有效场应按理论变换。
4. **简单极限**：仅保留交换、仅保留各向异性、去掉交错项等极限应回到可解析或明确的对照模型。
5. **零温阻尼**：无驱动力、无热噪声时，阻尼动力学的总能量应单调不增；若积分误差导致可见系统性上升，则时间步不合格。

当前 `Jtilde=0` 是与交错磁共享其余哈密顿量的**匹配 AFM 消融对照**，不是独立文献材料。论文中必须这样标注。

## 5. Gate 2：有限温随机场的统计与单位

当前采用的白噪声应满足零均值和涨落耗散关系。若代码中热场写为每步常值高斯场，其单分量标准差应与所采用的 LLG 约定严格一致；以当前代码约定为例，应核对类似

\[
\langle R_{i\mu}(t)\rangle=0,\qquad
\langle R_{i\mu}(t)R_{j\nu}(t')\rangle
=2D_i\,\delta_{ij}\delta_{\mu\nu}\delta(t-t'),
\]

以及离散化后的 `sigma ∝ sqrt(T/dt)`。`D_i` 中是否包含 `alpha`、`gamma`、`mu_s`、`1+alpha^2` 取决于代码采用的 Gilbert/Landau–Lifshitz 约定，必须从方程逐项推导，不能混用文献公式。

### 5.1 至少做 10^6 个独立分量的纯噪声抽样

对每个代表性 `(T, alpha, dt, mu_s)`：

- 均值 z 分数 `|mean|/(sigma/sqrt(N)) < 5`；
- 方差相对理论偏差小于 `max(1%, 5*sqrt(2/(N-1)))`；
- 不同格点、笛卡尔分量和时间步间的绝对相关系数小于 `max(0.01, 5/sqrt(N))`；
- 将 `dt` 减半后，场的标准差比应为 `sqrt(2)`，容差 1%；
- 使用相同 seed 必须逐位复现，不同轨迹 seed 不得产生相同序列；
- 正式数据统一在 CPU 上生成；不同 CPU/线程数不要求逐位相同，但统计量必须在置信区间内一致。GPU 只在以后确有必要的模型训练中使用，不用于本轮 LLG 数据生成。

现有 `validate_configuration_noise_response.py` 可检查构型相关的瞬时响应，但它不能代替上述完整白噪声审计。纯噪声统计报告脚本目前需要在服务器端补充，建议命名为 `scripts/validation/validate_thermal_noise_statistics.py`，输出 `thermal_noise_statistics.json` 和 PNG；在脚本真正加入仓库前，不要把这个文件名当成现成命令。

### 5.2 Stratonovich 与积分器

随机 LLG 是乘性噪声问题。当前代码采用同一随机场作预测和校正的随机 Heun，并在预测步和完成步归一化。它与 Stratonovich 解释相容，但“显式笛卡尔更新 + 人工归一化”在有限 `dt` 下仍可能改变弱统计量。

因此必须：

1. 保存归一化前的模长误差分布；
2. 用共享布朗增量比较 `dt`、`dt/2`、`dt/4`；粗步增量必须等于两个细步增量之和，不能只比较不同 seed 的两批数据；
3. 补充一个自动保持球面约束的参照积分器，例如隐式 Stratonovich 中点或 Cayley/旋转更新；
4. 不仅比较单条轨迹，还比较平衡分布、翻转概率、首达时间分布和机制比例。

建议放行标准：

- 平滑均值/方差类可观测量在 `dt/2` 与 `dt/4` 间差异小于 2%，且差异落在 95% bootstrap 置信区间内；
- 翻转/返回/未翻转概率的绝对差异不超过 0.03；
- 首达时间分布的 Wasserstein-1 距离不超过中位首达时间的 5%；
- 机制比例的最大绝对差异不超过 0.05；
- Heun 与几何参照积分器在最终选定 `dt` 上满足同一标准。

如果失败，减小 `dt`；如果 Heun 在可承受步长下仍不能通过，应更换生产积分器，而不是继续归一化。

## 6. Gate 3：用解析平衡分布认证热浴和积分器

第一项必须复现 Nishino 与 Miyashita 的非相互作用单磁矩基准。该问题具有精确 Langevin 平衡结果，能同时检测噪声幅值、阻尼约定、温度单位、Stratonovich 离散和归一化偏差。

文献设置：`h=2, M=1, alpha=0.05, dt=0.005, N=1000`，共 80,000 步，其中 40,000 步平衡、40,000 步测量。当前仓库的短流程只能算 pilot，不能替代这一完整复现。

至少输出：

- `m_z(T)` 与精确 Langevin 曲线及误差条；
- 若干温度下 `P(cos theta)` 与精确分布；
- 能量、磁化的自相关时间和有效样本量；
- `dt={0.005, 0.0025, 0.00125}` 的弱收敛图；
- Heun 与几何参照积分器的偏差表。

放行标准：

- 每个温度的 `|m_z - m_z_exact| <= max(0.01, 3*SE_combined)`；
- 二阶矩也在 `max(1%, 3*SE)` 内；
- 基于有效独立样本数的经验 CDF 通过预注册的 5% 显著性检验，或等价地给出 bootstrap 置信带并覆盖精确 CDF；
- 不得通过挑选温度点或剔除异常 seed 达标。

文献来源：[Nishino and Miyashita, Phys. Rev. B 91, 134411 (2015)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.91.134411)。

## 7. Gate 4：初态池确实来自目标平衡分布

正式有限温轨迹不能全部从“理想共线态 + 0.2 ps 热化”开始，除非已经证明 0.2 ps 足以去除初态记忆。

每个 `(system, size, T, boundary)` 至少运行 4 条独立平衡链，从有序、随机、反向和不同畴构型启动。对能量、序参量、结构因子峰、畴壁密度执行：

- split-Rhat `< 1.05`；
- 每个关键量总 ESS 建议 `>=1000`，每条链 ESS `>=200`；
- 丢弃不少于 `5 tau_int` 的 burn-in；初态池抽样间隔不少于 `2 tau_int`，推荐 `5 tau_int`；
- 不同启动方式的末段分布必须一致；
- 初态池与正式动力学随机种子命名空间分离。

若体系在目标温度存在多个长寿命盆地，不要强迫链混成单峰；应报告盆地权重、使用跨盆地采样方法，或明确数据条件化于某一初始盆地。

## 8. Gate 5：无外场优先的物理控制

本阶段正式协议设为 `external_field=0`、`SOT=0`。先研究热浴中的自发涨落、热激活跃迁以及无场弛豫；外场和 SOT 作为后续独立扩展，不能混入无场数据集。

无场数据分成三个物理问题，不能混称为同一种翻转：

1. **平衡涨落**：从认证平衡初态池出发，研究同一盆地内的路径分布；5 K 下可能只有小振动而没有翻转，这也是正确结果。
2. **无偏热激活跃迁**：从一个亚稳态出发，完全依靠热噪声越过势垒；必须给出等待时间/删失统计，不能为了看到翻转而偷偷加入场。
3. **无场非平衡弛豫**：从预先规定的倾斜 Néel 态、局域反向核或畴壁初态释放；能量由初态提供，热噪声决定弛豫路径。这能产生路径多样性，但不等价于平衡态的自发翻转。

无场主协议至少运行：

| 初态 | 温度 | 外场/SOT | 用途 | 正常预期 |
|---|---:|---:|---|---|
| 论文基态 | 0 K | 0 | 基态/积分器稳定性 | 保持初始盆地，能量单调不增 |
| 认证平衡池 | 5 K | 0 | 平衡涨落 | 分布平稳；短窗内可以没有跨盆地 |
| 亚稳态 | 多温度 | 0 | 热激活反转 | 等待时间随温度和势垒呈合理趋势 |
| 固定非平衡态 | 多温度 | 0 | 路径多模态 | 相同初态、不同 seed 给出弛豫路径分布 |

温度建议先做 `T={1, 2.5, 5, 10} K`。如果 5 K 在论文基态、无场短时间内大量跨盆地，先计算势垒和无场寿命；若与 Arrhenius 量级矛盾，优先检查单位、FDT、时间步或初态池。

只有无场数据认证完成后，才另建 `driven_extension/` 做以下四控制，且不得与无场训练集混合：

| 温度 | 驱动 | 用途 | 正常预期 |
|---:|---:|---|---|
| 0 K | 0 | 基态稳定性 | 保持初始盆地，能量不升高 |
| 5 K | 0 | 热稳定性 | 在目标观测窗内应与势垒预期一致；不能默认必然不翻转 |
| 0 K | 目标驱动 | 确定性阈值/分岔 | 判断变化是否由驱动主导 |
| 5 K | 目标驱动 | 有限温受驱路径分布 | 噪声在阈值附近改变盆地选择和首达时间 |

## 9. 为什么旧数据在 5 K 仍出现很大的磁态变化

这件事**不等于已经发现代码错误**，但当前也不能解释为可信的材料预测。

以仓库交错磁参数估算：

- `k_B T(5 K) ≈ 0.431 meV`；
- `k_B T/J1 ≈ 0.039`，相对 `J1=11.1 meV` 的交换能确实低；
- 但 `k_B T/K ≈ 9.2`，相对单格点各向异性 `K=0.047 meV` 并不小；
- `2K/mu_s ≈ 1.62 T`，当前 `0.70–0.80 T` 的阻尼型有效驱动并非可忽略；
- `dt=0.1 fs` 时单步热场标准差约可达 `92 T`，但白噪声场幅随 `dt^{-1/2}` 发散，单步角冲量只有约 `1.6e-3 rad`。因此不能把瞬时“92 T”直接与静态磁场比较。

旧数据的大幅变化最可能的机制是：后来加入的 0.5 ps 确定性 SOT 有效场把小尺寸、周期边界、近相干体系推到分界面附近，不同的弱热噪声选择不同终态盆地。SOT 并不来自原始 `微磁学.txt`，它是为了构造近阈值受驱验证场景而加入；5 K 负责分岔，不一定负责提供全部翻转能量。

因此这批旧 SOT 数据不能回答“无场下是否多模态”。本阶段必须重新生成 `external_field=0, SOT=0` 的独立数据。无场后未必还能得到“正负终态同时大量出现”的相同结论：

- 若 5 K 势垒相对很高、观测仅数 ps，正确结果可能是所有轨迹留在同一盆地，但瞬时涨落和相关时间不同；
- 若初态是亚稳态且观测时间覆盖热激活寿命，可能观察到不同首达时间、返回和少量翻转；
- 若从同一个非平衡初态释放，可观察多条弛豫路径，但必须明确结论是“条件非平衡路径分布”，不是“平衡态自发翻转”；
- 如果需要在可承受 CPU 时间内看到大量无偏稀有跃迁，应选已有热激活文献基准（如 Bauer 铁磁链），不能通过暗中加场或只保留翻转样本解决。

目前真正需要警惕的是：

1. `0.70–0.80 T` 是数值阈值扫描，并非已经由 Gomonay 文献中的材料电流/力矩换算得到；
2. 当前 SOT 对两个亚晶格的作用形式需核对是均匀还是交错力矩，不能默认适用于具体材料；
3. 0.2 ps 初态准备尚未证明达到有限温平衡；
4. 1–3 ps 内到达负的 Néel 分量不等于形成稳定终态；
5. 小周期晶格会抑制成核和畴壁路径，放大近相干翻转；
6. 显式 Heun + 投影的有限步长弱偏差尚未完成认证。

### 9.1 对 5 K 异常的判定树

- `T=0, drive=0` 不稳定：哈密顿量符号、初态或积分器错误，立即停止生产。
- `T=5 K, drive=0` 大量跨盆地：先算势垒和无驱动寿命；若与 Arrhenius 量级矛盾，检查 FDT、单位和 `dt`。
- `T=0, drive=0.78 T` 已翻转：说明旧数据主要是确定性驱动，5 K 多模态是近阈值扰动；该结果只能进入以后受驱扩展。
- 只有 `T=5 K, drive≈threshold` 出现分流，且随温度和 `dt` 收敛：这是合格的受驱验证场景，但不是当前无场主任务。
- 改变尺寸后近相干翻转消失并出现成核/畴壁：原 8×8 数据只描述小体系机制，不能外推到大体系。

## 10. CPU 提交规范

本轮所有 LLG 生成和认证均在 `fat` 分区使用 CPU，Slurm 任务名固定为 `zrs_data`。每个提交脚本至少包含：

```bash
#!/usr/bin/env bash
#SBATCH --job-name=zrs_data
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=altermagnetism_LLG/slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME CUDA_VISIBLE_DEVICES

# 所有生成器显式使用 --device cpu；先写临时 shard，成功后再原子改名。
```

提交后只需确认一次已进入队列/运行：

```bash
sbatch YOUR_SCRIPT.sbatch
squeue -u "$USER" -n zrs_data
```

看到 `PD` 或 `R` 后不再轮询、不启动监督任务；计算完成后由用户自行检查日志、退出码、证书和文件哈希。注意 `fat` 是 CPU 分区，不应请求 `--gres=gpu`。

## 11. Gate 6：时间步、保存间隔、尺寸和总时长

四项按顺序冻结，不能一次同时改变：

1. **时间步**：`dt, dt/2, dt/4`，每个条件至少 128 条共享初态、耦合噪声的轨迹。
2. **保存间隔**：在冻结 `dt` 后比较 `2.5, 5, 10 fs`；首达时间误差应小于保存间隔，机制比例差异不超过 0.03。
3. **尺寸**：至少 `L={16,32,48,64,96,128}`；比较翻转概率、首达时间、结构因子、最大空间方差、畴壁密度和相关长度。
4. **总时长**：先用变时长 pilot 得到承诺时间/弛豫时间分布，再固定正式时长。建议 `T_fixed >= Q99(t_commit)+10 tau_int`，若算力不足至少 `Q99+5 tau_int`，并报告未决比例。

固定时长不要求所有轨迹“能量变成常数”。正式终态可定义为序参量进入目标盆地并连续停留 `t_residence`，同时空间结构量进入稳定统计区间。未满足者标为 `unresolved/censored`，不可删除或强行归类。

放行建议：

- 最大两个尺寸之间关键概率差异 `<=0.03`；
- 中位首达时间相对差异 `<=5%`；
- 空间机制比例差异 `<=0.05`；
- 未决比例 `<=10%`（探索集可放宽至 20%，但不能称正式终态数据集）；
- 相关长度必须明显小于盒长，推荐 `xi < L/6`；否则继续增大尺寸。

## 12. Gate 7：路径类别不是靠肉眼划分

至少保留以下互斥/可组合事件：

- `no_crossing`：全局序参量从未跨过预注册分界面；
- `crossing_return`：跨越后又回到初始盆地，且未满足驻留时间；
- `committed_switch`：进入目标盆地并连续驻留指定时间；
- `unresolved/censored`：在固定观测窗内无法决定；
- `near_coherent`：局域序参量在翻转窗口内高度同步，空间方差和畴壁密度低；
- `nucleation`：局域反向团簇先出现并超过最小持续时间/面积；
- `domain_wall_propagation`：存在稳定界面且界面位置随时间连续移动。

阈值必须在看测试集之前由校准集冻结，并做阈值敏感性分析。现有：

```bash
bash run_zrs_mag.sh altermagnetism_LLG/scripts/validation/analyze_spatial_path_mechanisms.py INPUT.h5
bash run_zrs_mag.sh altermagnetism_LLG/scripts/validation/assess_extended_observation.py \
  --calibration-files CALIBRATION.h5 \
  --evaluation-files EVALUATION.h5 \
  --output REPORT.json
```

若当前数据没有成核/畴壁，不能通过更改标签阈值“造出”这些机制；应先完成尺寸和时长扫描。

## 13. 最终放行表

每个体系单独生成 `audit_decision.yaml`：

| Gate | 失败后的动作 | 是否允许训练 |
|---|---|---|
| 文件/元数据/切分 | 修复写入或重生成受影响 shard | 否 |
| 哈密顿量/邻接/单位 | 修代码，全部相关数据重生成 | 否 |
| 热噪声统计/FDT | 修代码，全部有限温数据重生成 | 否 |
| 解析平衡分布 | 减小 dt 或更换积分器后重生成 | 否 |
| 初态池 | 重建认证初态池及全部派生轨迹 | 否 |
| 物理控制 | 定位驱动、势垒或单位问题 | 否 |
| dt/save/size/duration | 冻结收敛协议后重生成 | 仅可做 debug |
| 全部通过 | 标记 `production_certified: true` | 是 |

最终证书至少包含：参数来源表、测试日志、随机场统计、解析基准、时间步/积分器/尺寸/时长收敛图、初态池 Rhat/ESS、路径类别和置信区间、失败样本清单、数据 SHA256。

## 14. 当前仓库已能做什么，仍缺什么

已有：有效场和能量一致性、周期边界/对称性、零温模长与能量、FDT 幅值、seed 重现、部分尺寸扫描、路径机制分析、Standard V3 preflight。

仍需补充后才可认证：

1. 独立的百万样本热噪声统计报告；
2. 共享布朗增量的 `dt/dt2/dt4` 弱收敛；
3. 隐式中点或 Cayley 几何参照积分器；
4. Nishino 完整 80,000 步和精确分布复现；
5. 认证有限温初态池；
6. 无场 5 K 的基态、平衡池、亚稳态和固定非平衡初态控制，以及温度扫描；
7. 以后若开展受驱扩展，再补具体 SOT 的亚晶格对称性和电流—力矩映射来源；
8. 固定时长、尺寸和保存间隔的最终冻结证书。

关于笛卡尔随机 LLG 和人工归一化的数值风险，参见 [Romá et al., Phys. Rev. E 90, 023203 (2014)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.023203)。该文并不说明笛卡尔坐标本身错误，而是说明绝大多数显式笛卡尔格式不能自动保持模长，有限步长算法必须用平衡统计与收敛性验证。
