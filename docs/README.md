# 项目文档

新对话按需读取以下文档；默认从第一项开始，不加载历史归档。

1. [当前状态与下一步](FLOAT_PUSH_TRAINING_NEXT.md)：唯一维护的训练交接入口。
2. [本次对话工作总结](FLOAT_PUSH_SESSION_REPORT.md)：完成的工作、提交、产物与清理边界。
3. [与原仓库的源码对比](FLOAT_PUSH_UPSTREAM_AUDIT.md)：复用范围、影响模型表现的差异与排查优先级。
4. [末端接口短程验证](FLOAT_PUSH_EE_SHORT_REPORT.md)：专家质量、2000 步模型、失败诊断和对比视频。
5. [学习接口](../high-level/float_ik/learning/README.md)：动作/观测、执行路径及采集契约；[环境配置](环境配置.md)：pro5000 环境与恢复依据。

关节直出方案的活动代码和旧产物已删除，见 [清理记录](FLOAT_PUSH_CLEANUP_REPORT.md)。新对话可直接使用 [下一轮 prompt](FLOAT_PUSH_NEXT_PROMPT.md)。`bak/` 的其他资料属于更早的 RL 工作。

本地仓库 `/Users/yu/Data/Video2Door/code`；服务器 `ssh pro5000`、仓库 `/data/zyf/video2door`。大产物留在仓库外 `/Users/yu/Data/Video2Door/reproduction/pro5000`。
