# 训练与任务指标

成功标准：正确方向门角达到80°，并完成原 pass_done（门后目标距离≤0.12m）。所有完整尝试计入分母。

| 环节 | 记录 | 产物 |
|---|---|---|
| 训练 | train/step、loss、lr、elapsed_sec | metrics.jsonl/csv、训练曲线 |
| 任务 | opened、pass_done、门角、目标距离、事件帧、结束原因 | summary.json、成功率表 |
| 数据 | 执行前state/action/image、next_state、帧序列、snap事件 | trajectory.npz、raw/ |
| 动作 | 原始/应用10D动作、重规划时刻、DP/脚本帧数 | execution.jsonl、动作曲线 |
| 执行 | 底座XY/速度/yaw、末端目标/实际位姿、q/dq/IK目标、接触 | 任务与关节诊断图 |
| 直接通过 | 首次80°后的反向行程、累计转向、pass_align帧数 | 多初态结果表 |
| 推理 | 每次重规划耗时 | 延迟P50/P95 |

## 检查与分析入口

- checks.py、dataset_contract.py：字段、维度、帧对齐、集合隔离、训练集统计和checkpoint更新数。
- offline_eval.py：用原控制器和100步DDPM，对专家观测生成动作，记录各通道RMSE和姿态误差。
- analyze_v3.py：每回合动作、关节、门角、底座路径、接触图及CSV。
- validate_forward_expert.py：预定18回合位置网格验证、全量分母和视频解码。
- deliver_v3.py：旧5000更新实验的汇总与同初态视频模板。

训练噪声MSE和末端误差用于诊断，任务成功由仿真回合判定。训练统计只从训练集合计算。

每个run保存源码/配置、种子与集合划分、模型更新数、SHA256和同初态视频。新训练建议继续保留训练曲线、离线动作误差、完整回合成功率、门角与底座路径图。
