# 无场研究计划事实盘点

_2026-09-11；代码、数据和调度状态分开记录_

---

## 📋 状态

| paper_id | 科学状态 | 下一批任务索引 |
|---|---|---|
| nishino_miyashita_2015 | accepted_with_deviation | [0, 1, 2, 3, 4, 5, 6, 7] |
| bauer_2011 | inconclusive | [8, 9, 10, 11, 12, 13, 14] |
| hirst_mn2au_2022 | inconclusive | [15, 16, 17, 18, 19, 20, 21, 22, 23, 24] |
| gomonay_2024 | inconclusive | [25, 26, 27, 28, 29, 30] |
| laliena_crnb3s6_2020 | fail | [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] |

## 🔍 证据与边界

- `nishino_miyashita_2015`: Primary 24/24 pass; old 207/216 convergence and nine accepted stationarity deviations retained. New independent stationarity pending.
- `bauer_2011`: 288 short chains ended at reduced time 200; zero saved crossings. Original duration and 500 events not certified.
- `hirst_mn2au_2022`: Wall still evolving, old width +30.48%; external per-bond audit, temperature/AFMR calibration pending.
- `gomonay_2024`: Candidate discrete S1 implemented; original moment-count geometry, external curves and critical velocity scan unresolved.
- `laliena_crnb3s6_2020`: Candidate corrected BVP fold 1.208970 versus 1.2405; numerical axis separation and external curve audit pending.

完整逐文件 SHA-256、Python 环境、队列、配额查询及历史退出码见 [inventory.json](inventory.json)。
本记录不把任务成功退出当作科学通过；旧失败、用户接受记录和原始数据均保留。
