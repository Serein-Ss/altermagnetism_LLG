# 目录重构验证记录

基于重构前提交 `b166157e0ea19cb096770d3bd3fe4f19bf35c0d5`。
本次没有提交 Slurm 任务、重训模型、重新生成科学数据或推送远程。

## 文件与溯源

- 990 个文件通过同一文件系统 rename 迁移，逐项验证大小与 inode。
- 后续复核 950 个应保持不变的数据、日志、可视化和快照文件，大小/inode 一致。
- 正式审计 submission.json 记录的 77 个源码及 batch 文件 SHA-256 全部一致。
- 历史报告、HDF5 属性、checkpoint 内部旧路径保留；当前读取入口使用显式迁移表解析。
- 8 个大文件的 LFS 源文件/分片路径和忽略规则同步更新；未重新计算或更改已有分片哈希。
- data 下未发现混放的 .py、.out、.png、.gif；output 下未发现 .h5、.h5 分片、.npz、.png、.gif。

## 软件验证

- 重构前：44 passed。
- 重构后最终：51 passed，6.62 秒；192 条 Matplotlib/Pyparsing 弃用警告，无失败。
- 新测试覆盖产物分流、历史路径解析、LFS 位置/大小、Slurm 语法/日志位置、旧目录移除、
  mock sbatch 快照提交（不实际提交）、绘图及 GIF 输出（仅临时目录）。
- 38 个活动 argparse CLI 的 `--help` 检查通过。
- 6 个既有 V2 三种子 flow/deterministic checkpoint 使用新代码在 CPU 上加载成功。
- 迁移后的 3 ps L16 测试数据通过惰性加载器读取：spins `(301,2,16,16,3)`，
  标量条件 `(10,)`，向量条件 `(5,3)`。
- 活动 Python 文件 AST 解析和当前 Slurm 脚本 bash 语法检查通过。

验证通过父目录 `bash run_zrs_mag.sh ...` 使用 zrs-mag。
完整 CLI 子进程检查首次遇到 MKL/OpenMP 线程层冲突；仅对验证命令设置
`MKL_THREADING_LAYER=GNU`，复查通过。未修改服务器全局环境配置。

最终测试命令（在项目父目录）：

```bash
MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 bash run_zrs_mag.sh -m pytest \
  altermagnetism_LLG/scripts/tests -q -p no:cacheprovider
```

这些检查验证重构兼容性，不是新增 GPU 训练验证或物理认证。
GUIDE 中尚未完成的研究阶段和历史未通过的认证门槛保持原状。
