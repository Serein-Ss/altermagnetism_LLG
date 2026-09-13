# 任务日志

- `slurm/`：普通 Slurm 任务输出。
- `audit/<batch>/`：审计任务、子进程输出与运行环境记录。

历史日志保留原文，因此包含迁移前路径。对应关系见根目录 DIRECTORY_MIGRATION.json。
新任务按任务 ID/批次写日志；不把训练 checkpoint 或数据轨迹写到这里。
