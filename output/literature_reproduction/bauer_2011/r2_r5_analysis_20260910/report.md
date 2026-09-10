# bauer_2011 数值验证分析

_本批 9 项；全部量按配置约化；不是正式复现认证。_

---

## 📋 结论与范围

状态：`inconclusive`，生产开关不变。观测窗仅 200，而计划要求 750000；固定 L=100、theta=0.11，没有文献长度/温度扫描。翻转阈值尚未由磁盆分布标定，事件数只作操作性敏感性检查；不能拟合寿命标度、Arrhenius 能垒或认证机制概率。

## 📚 论文、公式与参数

目标及来源以 [GUIDE](../../../../GUIDE/STRICT_LITERATURE_REPRODUCTION_PLAN.md) 和每个 run 的冻结配置为准。DOI：`10.1088/0953-8984/23/39/394204`。旧 reference_manifest 中部分实现状态已过期，本报告不把它当最新代码状态。

| 原参数 | 原值 | 单位 |
|---|---:|---|
| exchange | 1.0 | dimensionless |
| anisotropy | 0.1 | dimensionless |
| theta | 0.11 | dimensionless |
| alpha | 0.1 | dimensionless |

约化定义：

```json
{
  "energy": "J",
  "time": "hbar/J",
  "temperature": "kBT/J"
}
```

约化参数：

```json
{
  "equation_convention": "bauer_ll",
  "exchange": 1.0,
  "anisotropy": 0.1,
  "theta": 0.11,
  "alpha": 0.1
}
```

## 📊 数值结果与可视化

| task | 协议 | 主要数值结果 |
|---:|---|---|
| 0 | trajectory | mz_end=0.846040; crossed=0; events=0 |
| 1 | trajectory | mz_end=0.841894; crossed=0; events=0 |
| 2 | trajectory | mz_end=0.845938; crossed=0; events=0 |
| 3 | trajectory | mz_end=0.853474; crossed=0; events=0 |
| 4 | trajectory | mz_end=0.848316; crossed=0; events=0 |
| 5 | trajectory | mz_end=0.839202; crossed=0; events=0 |
| 6 | trajectory | mz_end=0.837592; crossed=0; events=0 |
| 7 | trajectory | mz_end=0.847971; crossed=0; events=0 |
| 8 | trajectory | mz_end=0.840984; crossed=0; events=0 |

![bauer_2011 的轨迹及数值参数比较](../../../../assets/literature_reproduction/bauer_2011/r2_r5_analysis_20260910/figures/overview.png)

_图 1：各条件均保留；曲线是实际保存帧。频谱概览仅展示 power_fraction≥1e-5 的点，该值为展示规则而非验收阈值；所有点均在 JSON。_

## 🔍 统计、收敛与验收门

单位自旋模长按 GUIDE 的 1e-10 检查，LLB 磁化长度不适用。仅保存了归一化后误差，缺少归一化前误差，不能凭模长通过来认证积分器。

随机轨迹按整条保留，Bauer 每条件 3 个种子×32 条轨迹。生存 bootstrap 为1000次、固定种子20260910、整轨迹重采样、点态95%区间；操作阈值0.7、驻留时间5仅作诊断，同时报告0.6/0.7/0.8与1/5/10敏感性。零事件的退化区间不证明概率为零。没有排除异常轨迹。

确定性比较以最细 dt 为数值参考，保存完整 RMS/max 差异；不是解析真值。随机时间帧相关，窗口均值差仅为描述，未做独立帧 t 检验。AFMR 拟合协方差不是独立重复CI，周期不足时不得强行解释为稳定频率。没有预注册的收敛容差，因此不新增 pass/fail 阈值，也不以不显著推断等价。

本体系的确定性比较见 [comparisons.json](../../r2_r5_analysis_20260910/comparisons.json)。

## ⚠️ 文献对照与未完成项

观测窗仅 200，而计划要求 750000；固定 L=100、theta=0.11，没有文献长度/温度扫描。翻转阈值尚未由磁盆分布标定，事件数只作操作性敏感性检查；不能拟合寿命标度、Arrhenius 能垒或认证机制概率。

没有完整的审定数字化参考曲线及误差，本批不能生成可信的四篇论文逐图数值误差。GUIDE 的标量目标及内部解析关系只作清楚标记的诊断对照；不伪造文献点或文献/模拟并排图。

## 🔗 数据、代码和重现

- task 0：[20260910_110834_validation_000](../../../../data/literature_reproduction/bauer_2011/derived/20260910_110834_validation_000/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 1：[20260910_110834_validation_001](../../../../data/literature_reproduction/bauer_2011/derived/20260910_110834_validation_001/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 2：[20260910_110834_validation_002](../../../../data/literature_reproduction/bauer_2011/derived/20260910_110834_validation_002/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 3：[20260910_110834_validation_003](../../../../data/literature_reproduction/bauer_2011/derived/20260910_110834_validation_003/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 4：[20260910_110834_validation_004](../../../../data/literature_reproduction/bauer_2011/derived/20260910_110834_validation_004/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 5：[20260910_110834_validation_005](../../../../data/literature_reproduction/bauer_2011/derived/20260910_110834_validation_005/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 6：[20260910_110834_validation_006](../../../../data/literature_reproduction/bauer_2011/derived/20260910_110834_validation_006/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 7：[20260910_110834_validation_007](../../../../data/literature_reproduction/bauer_2011/derived/20260910_110834_validation_007/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 8：[20260910_110834_validation_008](../../../../data/literature_reproduction/bauer_2011/derived/20260910_110834_validation_008/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`

[分析清单与代码哈希](../../r2_r5_analysis_20260910/manifest.json)；全部原始文件哈希在逐 run 的 metrics.json 中重新核对。分析工具采用 Karpathy Guidelines 和 Scientific Agent Skills 的原始数据保留、相关性与可视化原则。[^1]

[^1]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
