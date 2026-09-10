# gomonay_2024 数值验证分析

_本批 22 项；全部量按配置约化；不是正式复现认证。_

---

## 📋 结论与范围

状态：`inconclusive`，生产开关不变。色散为 32²/64²/96² 模拟晶胞，不是论文原尺寸。频谱与同代码解析式比较属于内部一致性，不是独立文献认证。仅 110 晶向、512×8 畴壁；一个非零初速度不能证明 Walker breakdown，不能将自由运动标为受驱动稳态。

## 📚 论文、公式与参数

目标及来源以 [GUIDE](../../../../GUIDE/STRICT_LITERATURE_REPRODUCTION_PLAN.md) 和每个 run 的冻结配置为准。DOI：`10.1038/s44306-024-00042-3`。旧 reference_manifest 中部分实现状态已过期，本报告不把它当最新代码状态。

| 原参数 | 原值 | 单位 |
|---|---:|---|
| J1 | 11.1 | meV |
| J2 | 1.88 | meV |
| J_tilde | 0.8 | meV |
| K_SW | 0.0 | meV |
| K_DW | 0.047 | meV |

约化定义：

```json
{
  "energy_scale": {
    "value": 11.1,
    "unit": "meV"
  },
  "length": "a0",
  "time": "mu_ref/(gamma*E0); physical gamma and moment require source audit before SI reporting"
}
```

约化参数：

```json
{
  "equation_convention": "gilbert",
  "J1": 1.0,
  "J2": 0.16936936936936936,
  "J_tilde": 0.07207207207207207,
  "K_SW": 0.0,
  "K_DW": 0.004234234234234234
}
```

## 📊 数值结果与可视化

| task | 协议 | 主要数值结果 |
|---:|---|---|
| 43 | spinwave | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 44 | spinwave | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 45 | spinwave | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 46 | spinwave | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 47 | spinwave | 展示 3/4 已激发诊断点; FFT bin=0.03140 |
| 48 | spinwave | 展示 3/4 已激发诊断点; FFT bin=0.03140 |
| 49 | control | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 50 | control | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 51 | control | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 52 | control | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 53 | control | 展示 3/4 已激发诊断点; FFT bin=0.03140 |
| 54 | control | 展示 3/4 已激发诊断点; FFT bin=0.03140 |
| 55 | spinwave | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 56 | spinwave | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 57 | spinwave | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 58 | spinwave | 展示 1/4 已激发诊断点; FFT bin=0.03140 |
| 59 | wall | v_fit=-0.0000000; width_final=8.903887 |
| 60 | wall | v_fit=-0.0000000; width_final=8.903887 |
| 61 | wall | v_fit=-0.0000000; width_final=8.903887 |
| 62 | moving_wall | v_fit=0.4992708; width_final=8.693706 |
| 63 | moving_wall | v_fit=0.4992708; width_final=8.693706 |
| 64 | moving_wall | v_fit=0.4992708; width_final=8.693706 |

![gomonay_2024 的轨迹及数值参数比较](../../../../assets/literature_reproduction/gomonay_2024/r2_r5_analysis_20260910/figures/overview.png)

_图 1：各条件均保留；曲线是实际保存帧。频谱概览仅展示 power_fraction≥1e-5 的点，该值为展示规则而非验收阈值；所有点均在 JSON。_

## 🔍 统计、收敛与验收门

单位自旋模长按 GUIDE 的 1e-10 检查，LLB 磁化长度不适用。仅保存了归一化后误差，缺少归一化前误差，不能凭模长通过来认证积分器。

随机轨迹按整条保留，Bauer 每条件 3 个种子×32 条轨迹。生存 bootstrap 为1000次、固定种子20260910、整轨迹重采样、点态95%区间；操作阈值0.7、驻留时间5仅作诊断，同时报告0.6/0.7/0.8与1/5/10敏感性。零事件的退化区间不证明概率为零。没有排除异常轨迹。

确定性比较以最细 dt 为数值参考，保存完整 RMS/max 差异；不是解析真值。随机时间帧相关，窗口均值差仅为描述，未做独立帧 t 检验。AFMR 拟合协方差不是独立重复CI，周期不足时不得强行解释为稳定频率。没有预注册的收敛容差，因此不新增 pass/fail 阈值，也不以不显著推断等价。

本体系的确定性比较见 [comparisons.json](../../r2_r5_analysis_20260910/comparisons.json)。

## ⚠️ 文献对照与未完成项

色散为 32²/64²/96² 模拟晶胞，不是论文原尺寸。频谱与同代码解析式比较属于内部一致性，不是独立文献认证。仅 110 晶向、512×8 畴壁；一个非零初速度不能证明 Walker breakdown，不能将自由运动标为受驱动稳态。

没有完整的审定数字化参考曲线及误差，本批不能生成可信的四篇论文逐图数值误差。GUIDE 的标量目标及内部解析关系只作清楚标记的诊断对照；不伪造文献点或文献/模拟并排图。

## 🔗 数据、代码和重现

- task 43：[20260910_110834_validation_043](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_043/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 44：[20260910_110834_validation_044](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_044/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 45：[20260910_110834_validation_045](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_045/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 46：[20260910_110834_validation_046](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_046/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 47：[20260910_110834_validation_047](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_047/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 48：[20260910_110834_validation_048](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_048/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 49：[20260910_110834_validation_049](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_049/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 50：[20260910_110834_validation_050](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_050/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 51：[20260910_110834_validation_051](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_051/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 52：[20260910_110834_validation_052](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_052/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 53：[20260910_110834_validation_053](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_053/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 54：[20260910_110834_validation_054](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_054/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 55：[20260910_110834_validation_055](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_055/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 56：[20260910_110834_validation_056](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_056/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 57：[20260910_110834_validation_057](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_057/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 58：[20260910_110834_validation_058](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_058/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 59：[20260910_110834_validation_059](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_059/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 60：[20260910_110834_validation_060](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_060/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 61：[20260910_110834_validation_061](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_061/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 62：[20260910_110834_validation_062](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_062/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 63：[20260910_110834_validation_063](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_063/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`
- task 64：[20260910_110834_validation_064](../../../../data/literature_reproduction/gomonay_2024/derived/20260910_110834_validation_064/r2_r5_analysis_20260910)，含 `metrics.json`、`series.npz`，频谱任务另有 `spectrum.npz`

[分析清单与代码哈希](../../r2_r5_analysis_20260910/manifest.json)；全部原始文件哈希在逐 run 的 metrics.json 中重新核对。分析工具采用 Karpathy Guidelines 和 Scientific Agent Skills 的原始数据保留、相关性与可视化原则。[^1]

[^1]: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
