# 2026-10-13工作区清理与交接

本轮完成清理、文档收敛和Git同步；下一阶段按 [复现方案](FLOAT_PUSH_REPRODUCTION_PLAN.md) 实现原全身DP路径的浮动底座后端。

## 删除内容

- 本地5个旧实验/包装器目录，文件内容合计4,392,269,768字节；服务器9个目录，合计4,397,379,485字节。
- 旧39D/11D和60Hz v2的训练/执行代码、参考限幅、调度器及对应测试。训练数据和checkpoint已删除。
- 重复会话总结、2000更新报告、v2报告、旧源码对比、旧RL资料及过时audit表，合并为当前方案、指标与环境文档。

完整删除路径和文件大小保存在两端精简证据目录的cleanup_manifest.json；仓库内 [汇总](audit/FLOAT_PUSH_CLEANUP_20261013.json) 记录范围与验证。

## 保留内容

原 `high-level/dp/`、全身push入口、原独立浮动脚本、机器人/门资产与私有Blackwell环境保持不变。

保留通用资源记账、数据检查、原控制器离线分析、绘图及配对视频工具；这些工具从记录的字段元信息读取数据。原训练/推理模块作为下一阶段实现主体。

精简证据约37MB：原CUDA700失败回合的原始CSV/NPZ/视频、一组专家/模型视频、v2专家轨迹/视频/源码来源、历史文档副本、预算账本和历史种子集合。

- 本地：`/Users/yu/Data/Video2Door/reproduction/pro5000/retained-evidence/20261013/`
- 服务器：`/data/zyf/video2door/.repro/retained-evidence/20261013/`

GPU账本保留累计176.587700/7200s；本轮未启动GPU任务。

## 验证与后续

4项数据/统计/配对分析测试通过；6类超时/锁/预算记账检查通过；保留的r0轨迹重新生成分析图成功。精简证据哈希核对通过，原源仓库模块哈希保持一致。源码、文档和删除范围同步到关联远程`YuFei-yuu/doorgym`，最终提交以Git HEAD为准。

新对话只需读取 [复现方案](FLOAT_PUSH_REPRODUCTION_PLAN.md)、[指标绘图](FLOAT_PUSH_METRICS.md)、[环境配置](环境配置.md)。可复制 [提示词](FLOAT_PUSH_NEXT_PROMPT.md) 继续实现和训练。
