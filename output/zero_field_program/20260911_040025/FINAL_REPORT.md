# 本次执行交接：文献补证与无场平衡先导

_2026-09-11；program_id: 20260911_040025。此为提交交接，不是全部研究目标完成报告。_

---

## 📋 提交结果

Slurm 已接受 **91 个作业**：`669173–669263`，名称全部为 `zrs-mag`。
其中 90 个计算子任务、1 个静态分布认证门任务；36 个多尺寸温度先导只有在证书存在、哈希匹配且实际指标通过时才运行。
本次提交之后不再轮询或监督进展，等待用户通知。

逐项作业号、数组索引、命令、依赖和时间申请见 [submission.json](../../../conf/frozen/zero_field_gomonay/20260911_040025/submission.json)。
例如 Bauer 原时长作业是 `669197_14`，认证门为 `669227`。
作业“已接受”不代表“正在运行”或“科学通过”。

## ✅ 已完成与已验证

- 读取并执行新 GUIDE 与旧严格复现计划的共同要求；新指令覆盖冲突的旧清理/停止规则
- 完成 [事实盘点](inventory.md) 和 [逐文件哈希、环境与历史退出码记录](inventory.json)
- 五篇文献分别建立 `reference/target_registry.json`；缺原文点、容差或几何映射的项如实标缺，未虚构外部参考
- 修正 Heun 投影预测子的诊断：记录真正的投影前误差，保留预测子归一化和原积分轨迹
- 新约化流式积分入口保存每条轨迹 RNG、步号、spins、时间、误差累计值和写入游标；身份不符拒绝续算
- 新无场入口复用 `DoubleLayer` 与 `ReducedLLG`，使用 `K_DW`，七类外加驱动显式为零；不调用旧 `UnifiedLLGSolver`
- 实现独立小格点 S1 能量/场对照、CPU Metropolis 静态参考、rank/folded Rhat 与 bulk/tail ESS 检查、哈希证书校验及事件分类接口
- 惰性 loader 用真实小 LLG 分片验证 2/4 位点晶胞、初态、真实时间和 mask；诊断数据默认拒绝进入训练，增加链/初态组划分泄漏检查

| 预检 | 结果 | 作业/证据 |
|---|---|---|
| 全部配置与 CPU 测试 | 122 通过，2 项 CUDA 测试跳过 | `669135` |
| GPU 专项测试 | 16 通过，含精确续算一致性 | `669136` |
| CPU / GPU 实际热浴接口 | 各 1,000,032 分量，注册检查通过 | [CPU](bath_cpu.json)、[GPU](bath_cuda.json) |
| 代表性计时 | 36 组完成，包含压缩/写盘；积分组至少 10,000 步，BVP 使用实际延拓工作单元 | [timing](timing) |

以上只证明对应软件和数值检查范围，不等于相互作用平衡、文献曲线或有限温路径已经认证。
统计实现采用预注册等价带、独立链误差和多重比较规则，不用“不显著”证明等价；Rhat/ESS 使用 ArviZ 0.22.0。[^1][^2]

## 📊 本次计算内容

| 分支 | 子任务数 | 计算范围 |
|---|---:|---|
| Nishino | 8 | theta=2，A/B 各 4 个独立重复；约化时间 1600，前 800 丢弃；保留原失败及用户接受记录 |
| Bauer | 7 | 6 组弱 RK 投影/非投影与 dt 对照；另 8 条 L100、theta=0.11、lambda=0.1、原时长 750000 的完整轨迹 |
| Hirst | 10 | 30³ 三温度平衡、延长 ASD/LLB AFMR、固定端面壁至约化时间 16000；均为数值补证，非完整论文认证 |
| Gomonay 文献分支 | 6 | 100/110 晶向有限长条的延长静态壁及 0.4c/0.9c 自由运动；未冒充原长条或完整临界曲线 |
| Laliena | 13 | 独立改变网格/域长/延拓步长，三盒长螺旋延长到 1000，三空间分辨率电流对照 |
| 相互作用内核 | 10 | 两温度下 8 组 LLG dt/几何积分器对照，2 组独立四链 MC 静态参考 |
| 无场温度先导 | 36 | L16/32/48/64/96/128 × theta=0.1/0.3/0.6/1/1.5/2.5；每条件四条分散初态链，时间 1000；须通过前置证书 |

MC 只用于静态分布对照，MC sweeps 不是 LLG 物理时间。
温度先导不是 128 路径 pilot，更不是正式训练集；没有把均匀/随机初态直接标为平衡初态。
本轮没有新正式训练数据、没有训练新模型，也没有修改旧 checkpoint。

## ⚙️ 资源、配置与恢复

- 环境：仅 `zrs-mag`；实际导入 PyTorch `2.6.0+cu118`、NumPy `2.0.1`
- 新增兼容依赖：ArviZ `0.22.0`、pandas `2.2.3`、xarray `2024.7.0`、xarray-einstats `0.7.0`；未改 base 或服务器全局配置
- 既有 NumPy 包元数据报告 `1.26.4`，实际导入是 `2.0.1`；未为此重装 NumPy，记录实际版本，不依赖错误元数据判定兼容性
- CPU 模拟在 fat，GPU 模拟在 rtx4090 分区申请 GPU；最多 4 个 CPU 模拟、2 个 GPU 模拟，另有 1 个短 CPU 认证作业
- 本次申请时长合计约 358.9 fat CPU 小时、76.7 GPU 卡时，含至少 1.5 裕量；存储估计上界约 171.8 GiB
- 冻结上限：512 CPU 小时、256 GPU 卡时、256 GiB、192 次分配、0 次自动重试；当前批次含预检计时共 129 次分配
- Bauer 原时长按计时估计纯计算约 130 小时，排队另计，不能期待与短时测试同时结束

依据见 [资源估计](resource_estimate.json) 与 [冻结 manifest](../../../conf/frozen/zero_field_gomonay/20260911_040025/manifest.json)。
工作树与冻结副本分开；实际执行冻结 Python/YAML，记录源文件 SHA-256。冻结后工作树仅有尾部空行清理及新增交接查询入口等非物理改动，不能用当前工作树哈希替代冻结执行哈希。

恢复限制：新流式 reduced 轨迹支持同设备/版本精确恢复。历史 Nishino 耦合 runner 与 LLB runner 尚无完整续算支持；若中断需保留失败文件并用新 run 重做，不能宣称从 seed 重启是续算。没有配置自动重提交。

## ⚠️ 尚未完成的科学门与实现

| 阶段 | 真实状态 |
|---|---|
| P0/P2 文献参考 | 目标登记已有；外部逐点曲线、部分几何/逐键审计与正式容差仍缺，不能宣称五篇复现完成 |
| P1 相互作用验证 | 计算与静态比较门已提交，结果未知；小格点绝对等价带仅用于前置先导，不是完整 P4 的 5% 误差证书 |
| P3 平衡与初态池 | 条件先导已提交；有序温区、相关时间、独立盆校准尚待数据；去相关初态池提取与认证入口仍待完成 |
| P4 完整条件路径 | 事件分类接口已有，但没有物理盆阈值；32×4 定长路径、窗口/保存间隔/事件精度认证尚未执行 |
| P5/P6 正式数据 | 五篇矩阵未关闭；生产入口保持不可用，独立生产 certifier、最终协议与原子 schema 封存仍待完成 |
| P7 模型 | 诊断分片 loader 已验证；生产数据适配、模型训练/三随机种子/最终尺寸测试尚未放行 |

Gomonay 500×500/10000×30 的“磁矩数”不能仅凭总数换成当前晶胞尺寸；现有小倾斜运动壁初始化也不能自动外推到阈值以上。
Laliena 约 1.208970 对文献 1.2405 的偏差保持未解释，不调整材料参数拟合。[^3]
Hirst 的原图阻尼及逐位移键审计、Bauer 的文献长度/温度点和 500 事件精度门均未被短时成功替代。

新 raw 分片的完成标记、作业退出码、物理证书是不同层级。当前新格式只开放诊断角色；正式数据完整 schema 的封存/认证不能由 `complete=true` 代替。
所有旧 raw、V2/checkpoint、旧失败报告和 Nishino 人工接受记录保留；本轮未清理或推送远程仓库。

## 📍 用户通知完成后的入口

先进行一次性状态核对，不自动训练：

```bash
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
bash ../run_zrs_mag.sh scripts/workflow/review_zero_field_campaign.py \
  --campaign conf/frozen/zero_field_gomonay/20260911_040025 \
  --output output/zero_field_program/20260911_040025/completion_review.json
```

该命令读取一次 accounting 和执行记录，不是监控器，也不替代后续科学分析。
后续按纸本目标、收敛和删失统计分析，并生成各文献与无场分支的图/动画；未通过的证书保持 fail/inconclusive。
详细阶段状态、明确未实现项、历史预检失败原因和全部作业索引见 [plan_state.json](plan_state.json)。

## 🔗 方法与来源

本次使用统计分析技能区分预注册检验与探索结果，使用文档技能保留可复核阶段状态；这些工具性帮助不构成物理认证。[^4]

[^1]: ArviZ. Rank-normalized split Rhat documentation, version 0.22.0. https://python.arviz.org/en/v0.22.0/api/generated/arviz.rhat.html
[^2]: ArviZ. Effective sample size documentation, version 0.22.0. https://python.arviz.org/en/v0.22.0/api/generated/arviz.ess.html
[^3]: Laliena, V., Bustingorry, S., & Campo, J. (2022). Author Correction: Dynamics of chiral solitons driven by polarized currents in monoaxial helimagnets. https://doi.org/10.1038/s41598-022-06147-1
[^4]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
