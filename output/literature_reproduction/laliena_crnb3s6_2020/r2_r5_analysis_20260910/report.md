# laliena_crnb3s6_2020 数值验证分析

_本批 14 项；全部量按配置约化；不是正式复现认证。_

---

## 📋 结论与范围

状态：`inconclusive`，生产开关不变。分支扫描固定一个 h_y，不能复现整条临界曲线；电流轨迹固定 dx，时间步收敛不代表空间收敛。螺旋需检查局部波矢及能量的长期稳定，而非仅看整数 winding。尚缺勘误图数字化点及误差。

## 📚 论文、公式与参数

目标及来源以 [GUIDE](../../../../GUIDE/STRICT_LITERATURE_REPRODUCTION_PLAN.md) 和每个 run 的冻结配置为准。DOI：`10.1038/s41598-020-76903-8`。旧 reference_manifest 中部分实现状态已过期，本报告不把它当最新代码状态。

| 原参数 | 原值 | 单位 |
|---|---:|---|
| A | 1.42 | pJ/m |
| D | 369 | microjoule/meter**2 |
| K | -124 | kilojoule/meter**3 |
| M_s | 129 | kiloampere/meter |
| a | 0.6 | nanometer |
| B_y | 300 | millitesla |
| alpha | 0.01 | dimensionless |
| beta | 0.02 | dimensionless |

约化定义：

```json
{
  "wavevector": "q0=D/(2*A)",
  "length": "1/q0",
  "field": "B0=D**2/(2*A*M_s)",
  "time": "1/(gamma*B0)"
}
```

约化参数：

```json
{
  "equation_convention": "gilbert",
  "q0_a": 0.0779577465,
  "kappa": -5.1726999655,
  "h_y": 0.8071914865,
  "alpha": 0.01,
  "beta": 0.02
}
```

## 📊 数值结果与可视化

| task | 协议 | 主要数值结果 |
|---:|---|---|
| 18 | branch | Gamma max=1.208065226; 偏差=-2.615% |
| 19 | branch | Gamma max=1.208009314; 偏差=-2.619% |
| 20 | branch | Gamma max=1.208065226; 偏差=-2.615% |
| 21 | branch | Gamma max=1.208960328; 偏差=-2.542% |
| 22 | branch | Gamma max=1.208970357; 偏差=-2.542% |
| 23 | helix | q_final=0.88153, 0.62967, 0.62967, 1.00747 |
| 24 | helix | q_final=0.90672, 1.10821, 0.50373, 1.10821 |
| 25 | helix | q_final=0.67164, 0.67164, 1.42724, 0.83956 |
| 26 | current | v_late=1.7395240; v_expected=1.78000 |
| 27 | current | v_late=1.7395240; v_expected=1.78000 |
| 28 | current | v_late=1.7395240; v_expected=1.78000 |
| 29 | current | v_late=0.0000000; v_expected=0.00000 |
| 30 | current | v_late=0.9835943; v_expected=1.00000 |
| 31 | current | v_late=2.1309626; v_expected=2.20000 |

![laliena_crnb3s6_2020 的轨迹及数值参数比较](../../../../assets/literature_reproduction/laliena_crnb3s6_2020/r2_r5_analysis_20260910/figures/overview.png)

_图 1：各条件均保留；曲线是实际保存帧。频谱概览仅展示 power_fraction≥1e-5 的点，该值为展示规则而非验收阈值；所有点均在 JSON。_

## 🔍 统计、收敛与验收门

单位自旋模长按 GUIDE 的 1e-10 检查，LLB 磁化长度不适用。仅保存了归一化后误差，缺少归一化前误差，不能凭模长通过来认证积分器。

随机轨迹按整条保留，Bauer 每条件 3 个种子×32 条轨迹。生存 bootstrap 为1000次、固定种子20260910、整轨迹重采样、点态95%区间；操作阈值0.7、驻留时间5仅作诊断，同时报告0.6/0.7/0.8与1/5/10敏感性。零事件的退化区间不证明概率为零。没有排除异常轨迹。

确定性比较以最细 dt 为数值参考，保存完整 RMS/max 差异；不是解析真值。随机时间帧相关，窗口均值差仅为描述，未做独立帧 t 检验。AFMR 拟合协方差不是独立重复CI，周期不足时不得强行解释为稳定频率。没有预注册的收敛容差，因此不新增 pass/fail 阈值，也不以不显著推断等价。

本体系的确定性比较见 [comparisons.json](../../r2_r5_analysis_20260910/comparisons.json)。

## ⚠️ 文献对照与未完成项

分支扫描固定一个 h_y，不能复现整条临界曲线；电流轨迹固定 dx，时间步收敛不代表空间收敛。螺旋需检查局部波矢及能量的长期稳定，而非仅看整数 winding。尚缺勘误图数字化点及误差。

没有完整的审定数字化参考曲线及误差，本批不能生成可信的四篇论文逐图数值误差。GUIDE 的标量目标及内部解析关系只作清楚标记的诊断对照；不伪造文献点或文献/模拟并排图。

## 🔗 数据、代码和重现

- task 18：[20260910_110834_validation_018](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_018/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 19：[20260910_110834_validation_019](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_019/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 20：[20260910_110834_validation_020](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_020/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 21：[20260910_110834_validation_021](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_021/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 22：[20260910_110834_validation_022](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_022/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 23：[20260910_110834_validation_023](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_023/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 24：[20260910_110834_validation_024](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_024/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 25：[20260910_110834_validation_025](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_025/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 26：[20260910_110834_validation_026](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_026/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 27：[20260910_110834_validation_027](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_027/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 28：[20260910_110834_validation_028](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_028/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 29：[20260910_110834_validation_029](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_029/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 30：[20260910_110834_validation_030](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_030/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 31：[20260910_110834_validation_031](../../../../data/literature_reproduction/laliena_crnb3s6_2020/derived/20260910_110834_validation_031/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`

[分析清单与代码哈希](../../r2_r5_analysis_20260910/manifest.json)；全部原始文件哈希在逐 run 的 metrics.json 中重新核对。分析工具采用 Karpathy Guidelines 和 Scientific Agent Skills 的原始数据保留、相关性与可视化原则。[^1]

[^1]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
