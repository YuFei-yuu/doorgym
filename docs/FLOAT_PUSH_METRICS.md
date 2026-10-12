# DP训练、推理与任务记录

更新：2026-10-13。方法参照固定源提交`0acc2f050479fb54a0670ccb90bc72b98b52a54a`。主目标和执行配置见 [复现方案](FLOAT_PUSH_REPRODUCTION_PLAN.md)。

## 源仓库直接记录的指标

| 环节 | 来源/字段 | 保存与绘图 |
|---|---|---|
| 训练 | `train_door_dp.py`：train/step、train/loss、train/lr、train/elapsed_sec | 每100更新保存JSONL/CSV；loss-step、lr-step、累计耗时-step；可同时使用原wandb日志 |
| 数据成功 | 原push recorder的 `pass_started`，正向门角达到80° | 每次采集的seed、门资产、是否开门、结束原因；成功数/尝试数 |
| 完成通过 | 原push任务的 `pass_done`、through目标与0.12m容差 | 原阶段标记、首次80°/通过时刻、最终门角与目标距离；开门成功率、完整任务成功率 |
| DP动作 | `make_door_dp_log_record`：dp_action_raw、applied_action、robot_command | 10通道随时间曲线，标注每16步重规划时刻 |
| 底座 | base.xy、height、lin_vel、ang_vel | 世界XY路径叠加门及通过目标；速度/yaw-rate曲线 |
| 末端 | ee.target_pos_world、actual_pos_world、target_quat、actual_quat、pos_error | 世界目标/实际XYZ、三维轨迹、位置误差范数、四元数姿态角误差 |
| 夹爪 | gripper.target、actual_pos | 目标/实际曲线；标注转把、开门与通过时刻 |
| 门/阶段 | door.dof、phase_id/name，原入口extra中的pass_started/pass_done | 门角/把手角、阶段时间轴，与末端和底座曲线对齐 |

原raw recorder把阶段保存在subtask_index作可视化。状态输入使用真实机器人字段。

## 浮动适配增加的观测记录

| 数据 | 用途与图 |
|---|---|
| obs_t、a_t、obs_t+1、frame/next_frame、sim_time | 验证50Hz时间对齐、回放和窗口边界；逐帧检查表 |
| IK输出、最终伺服目标、实际q/dq | 七关节目标/实际、速度及IK残差曲线 |
| 实际root位姿、URDF安装位姿、末端刚体位姿 | 核验世界/底座/安装原点坐标转换 |
| 原snap_arm_to_zero及freeze事件 | 在关节和末端图上标记原开门后处理触发 |
| dp_action_applied、图像有效性、命令来源 | DP控制帧数/占比、脚本命令帧数和相机缺帧 |
| GPU/进程资源、单次推理耗时 | 每次启动的资源快照、预算账本、推理延迟分布 |
| 结束原因、异常类型、完成物理步数 | 按全部尝试统计完整任务、超时和仿真异常 |
| 实际手指/把手contact body pair及可用的接触力 | 连续接触时间段图；几何距离另列 |

这些补充字段用于排查适配和解释任务过程，不改变原优化目标或设置额外精度门槛。

## 专家观测离线分析

固定最后5000更新checkpoint，使用验证集原16帧观测窗口和原100步DDPM推理，评估实际执行的16步动作块。

保存各动作通道RMSE、世界末端位置向量RMSE、归一化四元数的姿态角误差分布、vx/yaw-rate/夹爪误差；绘制预测/专家动作曲线与误差分布。按原subtask_index分段统计只用于分析。

数据统计使用原compute_stats/normalize_batch，训练集合均值/标准差及样本数单独保存。检查验证集隔离和z-score覆盖范围。验证噪声MSE若记录，应在冻结checkpoint上用固定窗口/噪声计算并独立列为诊断。

## 完整任务交付

- 同初态专家/模型、同资产和同执行版本，保存左右并排视频与原视频。
- 每个完整回合记录原开门成功、原pass_done、最终通过目标距离、首次事件时刻及结束原因。
- 汇总开门成功数/全部尝试数、完整任务成功数/全部尝试数；视频与指标使用相同seed和checkpoint。
- 训练loss表示原噪声预测损失；任务曲线与成功率来自仿真回合。日志、图表和视频分别放入run的training、evaluation、analysis目录。
- 关键产物下载本地，检查SHA256和完整视频解码。

现有 `checks.py` 检查数据，`offline_eval.py` 直接调用原控制器，`analyze.py/short_tools/deliver.py` 提供图表和配对视频基础。新的浮动入口按本表写出原JSONL及绘图用CSV/NPZ，分析工具随后适配新增字段。
