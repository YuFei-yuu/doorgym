# 工作区整理记录

2026-10-13。当前入口为[新对话交接](FLOAT_PUSH_TRAINING_NEXT.md)。

本次清理了旧LeRobot转换目录、20更新拟合试验、未采用的pilot大文件、短程诊断产物、随机兼容性模型、重复checkpoint，以及服务器项目pip HTTP缓存和Git传输包。

保留两套Conda环境、Blackwell工具链/构建源码/扩展缓存、机器人与门资产、全部训练执行代码。当前18回合完整数据、前后对照视频、旧raw训练/验证数据和最终5000更新模型均保留。

旧进展和历史预算从日常交接文档中移除，关键结论收敛到当前方案及简短旧模型说明。历史文档、试错元数据和删除清单存于：
- 本地 reproduction/pro5000/maintenance/20261013-handoff-cleanup/
- 服务器 .repro/maintenance/20261013-handoff-cleanup/

清理前后对保留模型、原始数据及当前验证轨迹核对SHA256。运行源码未改动，后续训练入口和数据转换工具保持可用。
