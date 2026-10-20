# 已完成：wc4 Joint9 ACT成功基线

用户已确认模型训练任务完成。唯一维护入口为[完整复现手册](REPRODUCTION_GUIDE.md)，训练量30k以上，可按需扩大至40k～50k。

| 项目 | 已验收结果 |
|---|---|
| 数据 | wc4 skill，50条成功Joint9轨迹，27500帧，前置/腕部depth |
| 最终模型 | ACT baseline，50k更新；保留完整模型、optimizer、RNG和统计 |
| 闭环 | 48/64（75%），最大绝对门角≥80° |
| 范围 | 不宣称独立泛化、穿越、碰撞合理性或停车已通过 |

[BASELINE.json](BASELINE.json)保存版本、最终模型SHA、服务器路径和W&B链接。最终模型及完整数据在服务器，局部数据镜像、最终结果、gzip原始日志和两段原始整体视频在本地 `reproduction/wc4-act/`；模型不进入Git。

保留的实验目录只有：

- `wc4-joint9-skill50-s62000-20261015`：成功NPZ、LeRobot数据、数据清单及实际采集配置。
- `wc4-act-resume40k-50k-s1000-20261015`：最终50k checkpoint和Door导出、最终64次完整原始日志、成功/失败代表视频、有效配置和训练进度。

目录名仅是产物定位符，不要求按名称重演历史恢复过程。早期试验、失败准备、旧DP和中间checkpoint已按用户要求删除；最终checkpoint内的历史父路径只是原始元数据，完整恢复以该50k目录自己的training_state为准。

服务器 `/data/zyf/video2door` 仅保留Gym/Blackwell支持及已安装editable依赖源码；它不再是旧主线工作区。新源码和ACT环境使用 `/data/zyf/video2door-wc4-act`。新任务不能修改或覆盖已验收数据、模型和结果。

[reference_tools](reference_tools/README.md)保留成功基线所用的检查实现参考；新run使用新参数，不直接执行其中历史路径常量。环境构建材料见[reproduction_support](reproduction_support/README.md)。
