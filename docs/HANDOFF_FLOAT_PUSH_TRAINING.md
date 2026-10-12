# 浮动底座推门交接入口

按以下顺序阅读：

1. [当前状态与下一步](FLOAT_PUSH_TRAINING_NEXT.md)：预算、待办和冻结边界。
2. [正式训练报告](FLOAT_PUSH_LONG_TRAINING_REPORT.md)：选模、部分独立测试及视频复现。
3. [学习接口](../high-level/float_ik/learning/README.md) 与 [正式工具](../high-level/float_ik/learning/long_tools/README.md)：实际源码契约。
4. [环境配置](环境配置.md)：运行、资产、Blackwell/CUDA/Vulkan 恢复依据。

原训练前交接全文保存在 Git：`git show 1f0f112:docs/HANDOFF_FLOAT_PUSH_TRAINING.md`。其中“尚无训练接口、数据或视频”等描述已过时；原低层 DP/RL 入口、低层 debug 权重不能替代浮动底座学习接口。
