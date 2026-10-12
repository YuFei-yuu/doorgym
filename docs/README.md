# 项目文档

## 浮动底座推门

- [当前状态与下一轮入口](FLOAT_PUSH_TRAINING_NEXT.md)：最新训练/测试状态、预算边界和新对话提示词。
- [正式受控续训报告](FLOAT_PUSH_LONG_TRAINING_REPORT.md)：5000→25000 步、选模及异常停止的部分独立测试。
- [首轮训练报告](FLOAT_PUSH_SMOKE_REPORT.md)：5000 步起点与首轮对照。
- [早期失败分析 JSON](FLOAT_PUSH_EARLY_FAILURE_ANALYSIS.json)。
- [正式实验原始授权提示词](FLOAT_PUSH_LONG_TRAINING_PROMPT.txt)：历史已执行计划，不代表重新授权预算。
- [训练前部署交接](HANDOFF_FLOAT_PUSH_TRAINING.md)、[环境配置](环境配置.md) 与 [历史环境交接](Video2Door_pro5000_环境配置与项目交接.md)。
- [首轮历史训练比较与计划](bak/FLOAT_PUSH_TRAINING_NEXT_SMOKE.md)。

服务器统一通过 `ssh pro5000` 连接，仓库为 `/data/zyf/video2door`。
固定使用 `.repro/blackwell/activate.sh`，不根据原始上游 README 重新建立 Isaac Gym 环境。
本地维护的报告位于 `/Users/yu/Data/Video2Door/code/docs`；实验产物位于
`/Users/yu/Data/Video2Door/reproduction/pro5000`，不提交模型、数据或视频。

服务器与本地的源码、调度器及文档按 Git 同步。历史封存中的旧路径是当时记录，保留原字节和哈希；维护报告更新了连接名称和本地链接，不改变实验数值或测试分母。
