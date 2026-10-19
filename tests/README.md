# 测试与冻结实验

当前正式入口见 [最新进展](../GUIDE/CURRENT_PROGRESS_20261019.md)。可复用模型、采样、数据加载和训练代码已登记到 `scripts/`，参数到 `conf/`，开发checkpoint到 `output/development/`，汇总结果到 `assets/research/bauer_path/`。正式模块不依赖本目录。

本目录保留：

- `unit/`：软件单元测试，包括 `test_bauer_path_release.py` 的冻结实现迁移对照。
- 各日期或repair目录：历史测试代码、受控实验、失败诊断和不可变来源校验。不能在本次整理中删空或改写，否则历史source_lock失效。
- `release_20261019/`：本次发布回归、CPU训练冒烟及接口验证输出，不是新的正式训练结果。
- 原始轨迹、测试checkpoint、大型网页和缓存仅本地保留，Git忽略；因此远程检出不包含运行全部历史实验所需的数据。

`on_policy_refresh_repair/cancellation.json` 覆盖先前submission/execution的任务状态：用户取消任务后没有授权恢复。不要依据旧的自动续提脚本直接重启历史链。

迁移回归需要本机冻结数据；无数据的远程检出可以导入正式模块、读取已上传checkpoint及使用自己准备的合合同网格数据。软件通过不等于条件路径物理准确性通过。
