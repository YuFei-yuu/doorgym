# 新对话交接入口

更新：2026-10-13。读取 [复现方案](FLOAT_PUSH_REPRODUCTION_PLAN.md)、[指标与绘图](FLOAT_PUSH_METRICS.md)、[环境配置](环境配置.md)，执行提示词见 [NEXT_PROMPT](FLOAT_PUSH_NEXT_PROMPT.md)。

当前已清理旧训练/执行适配与训练产物。原DP、全身推门入口、资产、环境、通用记账和分析工具保留。下一步实现原全身50Hz执行链的浮动底座后端，采集新演示并训练5000更新。

服务器预算继承文件：`/data/zyf/video2door/.repro/retained-evidence/20261013/budget.jsonl`；已用176.587700/7200s。历史种子集合在同目录`used_seeds.json`。

本地精简历史证据：`/Users/yu/Data/Video2Door/reproduction/pro5000/retained-evidence/20261013/`。当前原始任务代码的Git HEAD以两端实时核对为准。
