# 旧基线证据源码副本

这些文件由2026-10-15仓库外归档操作保留，供后续技术基线审查：

- `archive_summary.json`：服务器归档清单摘要、完整old/new路径映射及30份辅助源码原始SHA256。
- `archive_training_outputs.py`：本次只执行一次的可恢复归档脚本；默认预演，执行需显式参数。不应对当前仓库盲目重跑。
- `scripts/`：从历史实验/临时管理目录逐字节保存的30份辅助源码，不包含模型、图像数据、视频、运行日志或重复整仓库快照。
- `validation.json`：语法/源码完整性核验结果。

副本保留原始内容和绝对路径，不从此目录执行历史脚本。完整历史树及全部源码版本仍在服务器`/data/zyf/video2door-archives/20261015-baseline-switch-v1/tree/`。恢复方法及清理边界见[基线切换归档说明](../../FLOAT_PUSH_BASELINE_SWITCH_ARCHIVE.md)。
