# hirst_mn2au_2022 数值验证分析

_本批 20 项；全部量按配置约化；不是正式复现认证。_

---

## 📋 结论与范围

状态：`inconclusive`，生产开关不变。ASD 平衡为 8³/12³/16³，而目标是 30³；只有一个平衡温度及每条件一条轨迹，不能确定 T_N 或独立重复置信区间。AFMR 仅 8³；LLB 使用文献拟合输入而非本批 ASD 拟合。畴壁尚缺尺寸/长时扫描及逐键外部核验。

## 📚 论文、公式与参数

目标及来源以 [GUIDE](../../../../GUIDE/STRICT_LITERATURE_REPRODUCTION_PLAN.md) 和每个 run 的冻结配置为准。DOI：`10.1103/PhysRevB.106.094402`。旧 reference_manifest 中部分实现状态已过期，本报告不把它当最新代码状态。

| 原参数 | 原值 | 单位 |
|---|---:|---|
| J1 | -5.3422 | mRy |
| J2 | 0.6484 | mRy |
| J3 | -0.6341 | mRy |
| J4 | -6.8986 | mRy |
| J0_same | 2.5934 | mRy |
| J0_inter | -30.804 | mRy |
| d_z | -0.0663 | mRy |
| d_x | 0.0026 | mRy |
| a | 3.33 | angstrom |
| c | 8.537 | angstrom |
| moment | 3.8663 | mu_B |
| theta_300 | 300 | kelvin |
| theta_1000 | 1000 | kelvin |
| theta_1200 | 1200 | kelvin |
| theta_TN | 1335 | kelvin |

约化定义：

```json
{
  "energy_scale": {
    "value": 30.804,
    "unit": "mRy"
  },
  "length": "a",
  "time": "mu_ref/(gamma*E0); gamma_source_audit_required"
}
```

约化参数：

```json
{
  "equation_convention": "gilbert",
  "J1": -0.1734255292,
  "J2": 0.0210492144,
  "J3": -0.020584989,
  "J4": -0.2239514349,
  "J0_same": 0.0841903649,
  "J0_inter": -1.0,
  "d_z": -0.0021523179,
  "d_x": 8.44046e-05,
  "c_over_a": 2.5636636637,
  "theta_300": 0.06168312355777964,
  "theta_1000": 0.20561041185926549,
  "theta_1200": 0.24673249423111857,
  "theta_TN": 0.2744898998321194
}
```

## 📊 数值结果与可视化

| task | 协议 | 主要数值结果 |
|---:|---|---|
| 9 | llb_afmr | omega=0.0818447; RMSE=0.00112668 |
| 10 | llb_afmr | omega=0.0818447; RMSE=0.00112668 |
| 11 | llb_afmr | omega=0.0818447; RMSE=0.00112668 |
| 12 | llb_afmr | omega=0.0457478; RMSE=0.000275236 |
| 13 | llb_afmr | omega=0.0457478; RMSE=0.000275236 |
| 14 | llb_afmr | omega=0.0457478; RMSE=0.000275236 |
| 15 | llb_afmr | omega=0.0283047; RMSE=0.000205065 |
| 16 | llb_afmr | omega=0.0283047; RMSE=0.000205065 |
| 17 | llb_afmr | omega=0.0283047; RMSE=0.000205065 |
| 32 | equilibrium | late |ma|=0.915846 |
| 33 | equilibrium | late |ma|=0.915451 |
| 34 | equilibrium | late |ma|=0.914015 |
| 35 | equilibrium | late |ma|=0.917803 |
| 36 | equilibrium | late |ma|=0.915921 |
| 37 | afmr | omega=0.0788846; RMSE=0.0232569 |
| 38 | afmr | omega=0.0586632; RMSE=0.0339243 |
| 39 | afmr | omega=0.0304775; RMSE=0.025472 |
| 40 | wall | width/a=122.254038 |
| 41 | wall | width/a=122.254038 |
| 42 | wall | width/a=122.254038 |

![hirst_mn2au_2022 的轨迹及数值参数比较](../../../../assets/literature_reproduction/hirst_mn2au_2022/r2_r5_analysis_20260910/figures/overview.png)

_图 1：各条件均保留；曲线是实际保存帧。频谱概览仅展示 power_fraction≥1e-5 的点，该值为展示规则而非验收阈值；所有点均在 JSON。_

## 🔍 统计、收敛与验收门

单位自旋模长按 GUIDE 的 1e-10 检查，LLB 磁化长度不适用。仅保存了归一化后误差，缺少归一化前误差，不能凭模长通过来认证积分器。

随机轨迹按整条保留，Bauer 每条件 3 个种子×32 条轨迹。生存 bootstrap 为1000次、固定种子20260910、整轨迹重采样、点态95%区间；操作阈值0.7、驻留时间5仅作诊断，同时报告0.6/0.7/0.8与1/5/10敏感性。零事件的退化区间不证明概率为零。没有排除异常轨迹。

确定性比较以最细 dt 为数值参考，保存完整 RMS/max 差异；不是解析真值。随机时间帧相关，窗口均值差仅为描述，未做独立帧 t 检验。AFMR 拟合协方差不是独立重复CI，周期不足时不得强行解释为稳定频率。没有预注册的收敛容差，因此不新增 pass/fail 阈值，也不以不显著推断等价。

本体系的确定性比较见 [comparisons.json](../../r2_r5_analysis_20260910/comparisons.json)。

## ⚠️ 文献对照与未完成项

ASD 平衡为 8³/12³/16³，而目标是 30³；只有一个平衡温度及每条件一条轨迹，不能确定 T_N 或独立重复置信区间。AFMR 仅 8³；LLB 使用文献拟合输入而非本批 ASD 拟合。畴壁尚缺尺寸/长时扫描及逐键外部核验。

没有完整的审定数字化参考曲线及误差，本批不能生成可信的四篇论文逐图数值误差。GUIDE 的标量目标及内部解析关系只作清楚标记的诊断对照；不伪造文献点或文献/模拟并排图。

## 🔗 数据、代码和重现

- task 9：[20260910_110834_validation_009](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_009/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 10：[20260910_110834_validation_010](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_010/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 11：[20260910_110834_validation_011](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_011/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 12：[20260910_110834_validation_012](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_012/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 13：[20260910_110834_validation_013](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_013/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 14：[20260910_110834_validation_014](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_014/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 15：[20260910_110834_validation_015](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_015/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 16：[20260910_110834_validation_016](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_016/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 17：[20260910_110834_validation_017](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_017/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 32：[20260910_110834_validation_032](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_032/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 33：[20260910_110834_validation_033](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_033/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 34：[20260910_110834_validation_034](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_034/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 35：[20260910_110834_validation_035](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_035/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 36：[20260910_110834_validation_036](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_036/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 37：[20260910_110834_validation_037](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_037/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 38：[20260910_110834_validation_038](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_038/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 39：[20260910_110834_validation_039](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_039/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 40：[20260910_110834_validation_040](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_040/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 41：[20260910_110834_validation_041](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_041/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 42：[20260910_110834_validation_042](../../../../data/literature_reproduction/hirst_mn2au_2022/derived/20260910_110834_validation_042/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`

[分析清单与代码哈希](../../r2_r5_analysis_20260910/manifest.json)；全部原始文件哈希在逐 run 的 metrics.json 中重新核对。分析工具采用 Karpathy Guidelines 和 Scientific Agent Skills 的原始数据保留、相关性与可视化原则。[^1]

[^1]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
