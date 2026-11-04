# WC4流程与结果入口

当前交接请先读[流程文档集中目录](flow_handoff_20261025/README.md)，最新为[新专家采集与ACT训练流程报告（2026-11-04）](flow_handoff_20261025/WC4_新专家采集与ACT训练_20261104.md)。

已验收专家完成50条／77,356帧及100条／154,667帧数据。50条ACT训练至300k；100条ACT独立从头训练至200k，最终4/16，120k主线最高6/16。成功按实测开门≥80°且最终合法重锁；其他事件单列。本次整理归档不授权下一轮实验，预算由用户另行确认。目录中的2026-10-25报告保留历史语境，不代表当前状态。

## 历史基线：仅开门Joint9 ACT

以下保留已完成的旧开门基线。其[完整复现手册](REPRODUCTION_GUIDE.md)用于查阅环境和历史配方；其中训练量及开门成功口径不自动适用于新的完整任务或专家重写。

| 项目 | 已验收结果 |
|---|---|
| 数据 | wc4 skill，50条成功Joint9轨迹，27500帧，前置/腕部depth |
| 最终模型 | ACT baseline，50k更新；保留完整模型、optimizer、RNG和统计 |
| 闭环 | 48/64（75%），最大绝对门角≥80° |
| 范围 | 不宣称独立泛化、穿越、碰撞合理性或停车已通过 |

[BASELINE.json](BASELINE.json)保存版本、最终模型SHA、服务器路径和W&B链接。最终模型及完整数据在服务器，局部数据镜像、最终结果、gzip原始日志和两段原始整体视频在本地 `reproduction/wc4-act/`；模型不进入Git。

该旧开门基线的核心产物目录为：

- `wc4-joint9-skill50-s62000-20261015`：成功NPZ、LeRobot数据、数据清单及实际采集配置。
- `wc4-act-resume40k-50k-s1000-20261015`：最终50k checkpoint和Door导出、最终64次完整原始日志、成功/失败代表视频、有效配置和训练进度。

目录名仅是产物定位符，不要求按名称重演历史恢复过程。该旧开门路线的早期试验、失败准备、旧DP和中间checkpoint曾按用户要求清理；这不是当前完整任务／新专家历史证据的删除授权。最终checkpoint内的历史父路径只是原始元数据，完整恢复以该50k目录自己的training_state为准。

服务器 `/data/zyf/video2door` 仅保留Gym/Blackwell支持及已安装editable依赖源码；它不再是旧主线工作区。新源码和ACT环境使用 `/data/zyf/video2door-wc4-act`。新任务不能修改或覆盖已验收数据、模型和结果。

[reference_tools](reference_tools/README.md)保留成功基线所用的检查实现参考；新run使用新参数，不直接执行其中历史路径常量。环境构建材料见[reproduction_support](reproduction_support/README.md)。
