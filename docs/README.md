# 项目文档

## 当前阅读顺序

1. [当前状态与下一步](FLOAT_PUSH_TRAINING_NEXT.md)：最新结果、待办、预算与冻结边界。
2. [正式训练与成功视频复现](FLOAT_PUSH_LONG_TRAINING_REPORT.md)：5000→25000 验证、部分独立测试、15000/seed 4008 补录与证据。
3. [环境配置](环境配置.md)：唯一维护的 pro5000 环境、运行和 Blackwell/CUDA/Vulkan 恢复依据。
4. [学习接口](../high-level/float_ik/learning/README.md) 与 [正式工具](../high-level/float_ik/learning/long_tools/README.md)：当前末端目标 → IK → 伺服契约，以及历史关节工具的兼容边界。

## 历史资料

- [首轮摘要](FLOAT_PUSH_SMOKE_REPORT.md)：5000 步起点、数据来源、原始指标与证据路径。
- [早期失败分析 JSON](FLOAT_PUSH_EARLY_FAILURE_ANALYSIS.json)：保持原始数值。
- [正式实验原授权提示词](FLOAT_PUSH_LONG_TRAINING_PROMPT.txt)、[首轮旧计划](bak/FLOAT_PUSH_TRAINING_NEXT_SMOKE.md)：历史已执行计划，不重新授予预算。
- `bak/` 的其他报告涉及旧 RL/奖励任务，保持原样。

2026-10-12 整理：状态、实验结果和环境说明分别集中维护；重复环境/训练前交接文件保留短入口以兼容旧链接。原完整版本可用 `git show 1f0f112:docs/<文件名>` 查看。压缩叙述和重复表格不改变实验数值、源码、模型或原测试分母。

服务器 `ssh pro5000`、仓库 `/data/zyf/video2door`；本地报告 `/Users/yu/Data/Video2Door/code/docs`，产物 `/Users/yu/Data/Video2Door/reproduction/pro5000`。历史实验报告、源码快照与 SHA256 封存不改写；新增补录有独立清单，预算仅追加并保留原前缀。
