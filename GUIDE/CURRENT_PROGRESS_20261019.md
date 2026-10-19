# 当前进展与正式目录入口

更新日期：2026-10-19。本页覆盖较早状态快照；已确认的研究合同仍见核心主线、架构合同和实施清单。历史实验不重写。

## 核心目标

从给定 Hamiltonian 体系的小体系样本学习条件有限温随机路径分布，再检验同体系、同边界、同物理时间网格的大尺寸生成。当前先把 Bauer 开放链上的条件短时动力学做准确，再谈尺寸推广和准确性匹配的加速。不是只拟合磁矩动画，也不把所有极端稀有事件概率的恢复作为启动开发的前提。

边界固定于模型包；初态按指定构型或可复现准备规则提供，不训练初态生成器。温度等条件的泛化必须有数据覆盖和独立验证，不能靠修改输入数字宣称完成。

## 当前实现与适用范围

| 项目 | 当前内容 |
|---|---|
| 体系 | Bauer 一维均匀铁磁开放链，L16/25/32 都参与训练 |
| 参数 | J=50 meV，K/J=0.1，阻尼0.1，theta=0.11，T=63.824849668 K，外场0 |
| 参考 | Milstein–Tretyakov 5.9；积分0.25/0.125 fs；存帧0.1 ps |
| 模型 | 分层 SpinPathTransformer，width64/depth4；局域图、区域聚合、8个潜在token、时间注意力、条件调制、切空间输出 |
| 包装 | 连续双锚点轴向规范平均和链序反转平均；保留旧checkpoint合同 |
| 生成 | 学习[0,0.1] ps两帧块；接续10次可执行1 ps，但不代表1 ps准确性通过 |
| 采样 | midpoint128/512个辅助运输步，与LLG物理积分步不同 |
| 权重 | whole-teacher对照，seed11/22/33，各step130240；不是事后挑选的最佳seed |

联合旋转的是磁矩与结构／晶体参考方向。Bauer额外的绕易轴SO(2)和链序反转不等于一般自旋独立SO(3)或反射等变。当前轴向包装不支持两端相应双锚点同时过于接近易轴极点的初态（横向平方和不超过1e-12）。不应把当前包描述成已经验证的通用多体系模型。

## 已完成与当前结论

模型可训练、采样，单位模长、首帧锚定和适用对称性的部分软件检查已通过；但独立开发父状态上的条件路径统计仍有偏差。没有完成小体系定量认证、未见尺寸推广认证或准确性匹配的速度优势证明。

最近完成的 whole 对 patch2 teacher 对照：三种子各2048更新，step128192至130240。条件增量均值RMSE的描述性均值为0.0626902对0.0626321，协方差RMSE为0.00943075对0.00942290；完整1 ps增量方差RMSE为0.0234741对0.0236282，两时统计RMSE为0.0200228对0.0200031。改善很小、全路径效果混合，不能宣布修复有效；保留whole对照。

最近确认的来源差异：旧模型状态库来自seed11 step78144，而当前是三种子step130240。同索引状态平均磁矩角差约0.254/0.162/0.192 rad，说明输入分布已变化，但不是唯一故障原因的证明。

已提取144个当前模型派生状态，仍只有24个祖先TRAIN来源组。当前仅完成其中一个来源的CPU先导：6状态、1920条0.1 ps未来，参考生成约13.11秒；43项先导软件检查通过。不能把计划中的46080条或正式刷新训练记成已完成。

## 任务状态：停止，禁止自动续提

用户已要求停止项目任务。672358_0–1、672359_0–1、672360、672361已取消；GPU预检和正式刷新对照未完成，没有新的正式训练结果。本次仅整理、软件迁移回归和Git发布，不恢复上述任务。历史execution记录描述的是当时状态，最新停止事实见 `tests/on_policy_refresh_repair/cancellation.json` 与精简进展JSON。

## 正式目录与测试边界

| 内容 | 正式位置 |
|---|---|
| 骨干／Bauer包装／球面采样／经验teacher | `scripts/model/` |
| 惰性窗口与TRAIN条件库 | `scripts/datasets/bauer_path_windows.py`、`bauer_conditional.py` |
| 训练目标、teacher与可执行训练入口 | `scripts/training/bauer_path_*.py`、`train_bauer_path.py` |
| 生成入口 | `scripts/inference/bauer_path.py` |
| Bauer能量、短时统计及描述性评估入口 | `scripts/analysis/bauer_*.py`、`evaluate_bauer_path.py` |
| 物理／模型／训练参数、数据与checkpoint索引 | `conf/research/bauer_path/` |
| 实际使用的数据，本地保留不上传 | `data/research/datasets/bauer_open_chain/` |
| 三种子开发权重，不是科学验收通过的最终模型 | `output/development/bauer_path/step130240/seed{11,22,33}/checkpoint.pt` |
| 精简统计、CSV和网页 | `assets/research/bauer_path/progress_20261019/` |
| GPU模板，尚未提交 | `slurm/train_bauer_path.sbatch`，固定rtx4090 |
| 迁移回归、冒烟、历史实验代码和冻结证据 | `tests/` |

新的正式模块不导入 `tests`。冻结实验保留原路径和校验和：它们属于测试／诊断档案，不是正式运行入口。原数据和权重保留在冻结实验中，同时登记正式副本，避免破坏历史来源校验。没有大规模删除或改写历史实验。

## 使用方式

在项目根目录，所有Python均使用zrs-mag：

```bash
# 仅软件迁移回归，包含旧实现对照；不代表物理准确性验收
bash ../run_zrs_mag.sh -m pytest tests/unit/test_bauer_path_release.py -q

# 显式给定单位磁矩初态[N,3]；原始输出保存data，不放assets/output
bash ../run_zrs_mag.sh -m scripts.inference.bauer_path \
  --checkpoint output/development/bauer_path/step130240/seed11/checkpoint.pt \
  --initial /path/to/initial.npy --branches 16 --duration-ps 1 \
  --flow-steps 128 --device cpu \
  --output data/research/generated/bauer_path/paths.h5

# 同一实际初态、同网格的独立分支比较；只评估首个0.1ps，不伪称完整1ps评估
bash ../run_zrs_mag.sh -m scripts.analysis.evaluate_bauer_path \
  --generated /path/to/generated.h5 --reference /path/to/reference.h5 \
  --output assets/research/bauer_path/native_lag_metrics.json
```

CPU冒烟训练需显式 `--fresh-rng`，因为CPU与CUDA随机数状态算法不同，不可声称CPU是逐位复现CUDA续跑。正式训练入口从已登记checkpoint开始一个新的优化器重启阶段，不会自动选择模型、生成参考或提交下游任务。YAML描述当前受支持配方，并不意味着任意体系参数变化都能复用此checkpoint。Slurm模板仅备好，重新计算仍需用户明确授权。

## 后续优先级（未启动）

1. 如获重新计算授权，先完成当前模型状态库的受控刷新预检，核验祖先来源分组、未来噪声独立性及原生抽样比例。
2. 控制其他变量，对照旧状态与各自seed当前状态的teacher；分项比较条件增量、空间协方差、能量、两时统计和完整接续偏差。
3. 保留失败来源未决的结论，不盲增预算。达成预先声明的物理指标后冻结模型，再用独立同尺寸LLG验证未见尺寸。
4. 当前复用开发集不能成为未触碰最终测试；已知joint-v2近似区间覆盖不足，不标成已认证95%区间。事件须用户确认原始证据和定义，不作自动通过门槛。

## 发布原则

本次发布软件验证：12项迁移回归及23项相关合同检查通过，CPU生成／评估命令行通过。另完成单更新CPU训练冒烟，有限梯度、优化器和checkpoint保存通过，训练部分48.44秒；不是完整平衡实验。16更新CPU冒烟和一项重复大梯度测试因超出短测预算中断，不能记成通过；完整限制登记于 `assets/research/bauer_path/progress_20261019/release_validation.json`。这与先前43项先导检查是不同批次，均不等于物理认证。

GUIDE、正式代码、测试代码／协议、精简分析和开发checkpoint进入Git；原始data、测试轨迹、巨型网页、缓存、任务原始日志不进入本次新提交。Git LFS保留，checkpoint仍通过LFS管理。历史远程研究方案正常合并，不强制覆盖历史。
