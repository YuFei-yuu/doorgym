> 文档维护（2026-10-12）：服务器连接统一为 `ssh pro5000`，报告归入 `code/docs`，本地链接指向 `reproduction/pro5000`。仅维护名称与链接，实验数值、预算和测试分母不变。实验目录中的原始 `REPORT.md` 与 SHA256 封存保持原样。

# 浮动底座推门：受控续训实验报告

实验 `20261012-005832-controlled`；状态：`stopped`。停止原因：`Final test infrastructure failure`。

本轮累计占卡进程墙钟 **2.383 GPU 小时**（上限 6 小时），新增优化 **20000 步**，累计到 **25000 步**。完整进程计时包含加载、保存、采集、评估和视频；账本也保守包含 CPU 预检命令。

资源停止后经用户明确授权接续：保留原 15000/21000 步模型文件，沿用已完成的 21000 步验证，直接从该 checkpoint 继续优化，不重复旧验证；原预算前缀、模型 SHA256 与配置核对见 resume_state.json。恢复前已计费 1.636 GPU 小时，本次接续新增计费 0.747 小时，二者累计受同一 6 小时上限约束。恢复时沿用历史连续无改善计数 0，随后按完整验证结果更新为 2。旧选模与交付记录封存在 `resume_events/1791784292846482382`，此前恢复事件见 prior_resume_events。

本次接续新增优化更新 **4000 步**。新增训练从原 checkpoint 恢复，恢复审计与逐步日志保留。

## 独立闭环结论

| 方法 | 成功/已启动 | Wilson 95% | 完成/计划 | 最终平均门角 | reach / grasp / 解锁 |
|---|---:|---|---:|---:|---|
| expert | 8/8 | [67.6%, 100.0%] | 8/30 | 89.999° | 8 / 8 / 8 |
| untrained | 0/8 | [0.0%, 32.4%] | 7/30 | 0.000° | 0 / 0 / 0 |
| best | 4/7 | [25.0%, 84.2%] | 7/30 | 59.460° | 2 / 0 / 5 |
| last | 0/7 | [0.0%, 35.4%] | 7/30 | 51.427° | 2 / 0 / 4 |

原计划 120 回合，实际启动 30 回合、完成 29 回合、未启动 90 回合。逐回合状态见 planned_test_attempts.csv/JSON；各组未启动数为 expert 22、untrained 22、best 23、last 23。异常回合保留在成功率分母中，未重试。

这些比例和 Wilson 区间仅描述已启动的部分测试；因基础设施故障提前终止且各组样本数不等，不能当作预定每组 30 回合的完整性能或证明稳定闭环成功。

停止证据：未训练模型 seed 5007 已保存 294 个完整执行步；下一步后观测有限性断言失败。日志先出现 PhysX 碰撞阶段错误码 700，继而出现 CUDA illegal memory access 和求解器 kernel 失败。已保存轨迹及待执行动作均为有限值；故障后状态未保存，无法定位具体非有限通道或确定 CUDA 故障触发原因。仅能确认该回合发生仿真基础设施故障，不能归因于输入契约、显存不足或硬件故障。见 stop_diagnostics.json 与 logs/0166_test_untrained_seed_5007.log。按原停止规则退出，剩余预算没有继续使用。

优化已达到累计 25000 步上限，连续无改善计数为 2/3；选模已在独立测试首次启动前冻结。独立测试的提前停止原因与训练步数上限分别记录。

最佳模型 15000 步；最后模型 25000 步。

best 的 4 个数值主成功回合中，同时通过 reach 与 grasp 保持指标的有 0 个。主成功判据未要求稳定抓取，必须结合阶段指标与视频解释，不能将数值成功直接称为稳定抓把推门。

主成功：已解锁、最终正确方向门角 ≥80°、尾端净距离 ≥0.52 m。所有已启动回合均进入分母；未启动回合不计为已评估。原 `open_80deg` 兼容字段实际默认 20°，另行保留，不参与主成功判断。

## 前 1 秒诊断

同步、坐标、上一执行动作、模型历史/队列、深度及归一化契约检查：**True**。检查明细和逐通道原始数据保留。

| 5000 步模型回合 | 首次超训练状态范围 | 首次 ≥50% 通道限幅 |
|---|---:|---:|
| policy_5000_d100_seed_4000 | 0.017 s | 0.000 s |
| policy_5000_d100_seed_4001 | 0.017 s | 0.000 s |
| policy_5000_d100_seed_4002 | 0.017 s | 0.000 s |
| policy_5000_d25_seed_4000 | 0.033 s | 0.400 s |
| policy_5000_d25_seed_4001 | 0.033 s | 0.400 s |
| policy_5000_d25_seed_4002 | 0.017 s | 0.333 s |

本轮诊断专家完成 1 回合，主成功 1/1；这是控制与观测参照，不是预定 30 回合独立专家测试。

25/100 去噪对照使用相同 checkpoint 与种子 4000–4002，仅改变采样步数；100 步结果仅作诊断。录制专家观测下的离线动作误差保存在 offline_diagnostics.json，不能替代真实闭环表现。训练集包含夹爪启动速度瞬态，单个较大 z-score 本身不能证明实现错误。状态名 q0/dq0 到 q5/dq5 对应 joint1 到 joint6，q6/dq6 对应夹爪。

![前 1 秒诊断](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled/plots/early_closed_loop_diagnostics.png)


| 去噪步数（诊断） | 闭环成功 | 平均最终门角 | 专家观测离线底座 vx RMSE | 机械臂关节 RMSE 范围 |
|---:|---:|---:|---:|---:|
| 25 | 0/3 | 0.000° | 0.01186 m/s | 0.027–0.059 rad |
| 100 | 0/3 | 0.000° | 1.79073 m/s | 9.412–23.514 rad |

实装 DDPM 的 25/100 步时间表首步分别是 t=96/99；预测 x0 对 epsilon 误差的首步放大系数分别约为 21.4/2029.2。100 步引入更低信噪比的终端步，在 clip_sample=False 下可能放大尚未充分拟合的噪声预测误差。此为配置与公式支持的解释，尚未单独干预验证因果。

100 步诊断在三个回合的首个动作即出现多数通道限幅，且专家观测离线预测误差同样很大；因此它的异常不能仅归咎于环境状态偏离。25 步则先出现关节速度越界，再出现大范围限幅。

policy_5000_d100_seed_4000 首次越过训练范围的通道：base_x_initial_m, base_y_initial_m, base_vx_body_mps, base_vy_body_mps, base_wz_radps, dq0_radps, dq2_radps, dq3_radps, dq4_radps, dq5_radps, previous_base_vx_body_mps, previous_base_vy_body_mps, previous_base_wz_radps, previous_joint1_target_rad, previous_joint3_target_rad, previous_joint4_target_rad, previous_joint5_target_rad, previous_joint6_target_rad；policy_5000_d100_seed_4001 首次越过训练范围的通道：base_x_initial_m, base_y_initial_m, base_yaw_sin, base_vx_body_mps, base_vy_body_mps, base_wz_radps, dq0_radps, dq2_radps, dq3_radps, dq4_radps, dq5_radps, previous_base_vx_body_mps, previous_base_vy_body_mps, previous_base_wz_radps, previous_joint1_target_rad, previous_joint3_target_rad, previous_joint4_target_rad, previous_joint5_target_rad, previous_joint6_target_rad；policy_5000_d100_seed_4002 首次越过训练范围的通道：base_x_initial_m, base_vx_body_mps, base_vy_body_mps, base_wz_radps, dq0_radps, dq2_radps, dq3_radps, dq4_radps, dq5_radps, previous_base_vx_body_mps, previous_base_vy_body_mps, previous_base_wz_radps, previous_joint1_target_rad, previous_joint3_target_rad, previous_joint4_target_rad, previous_joint5_target_rad, previous_joint6_target_rad；policy_5000_d25_seed_4000 首次越过训练范围的通道：dq1_radps；policy_5000_d25_seed_4001 首次越过训练范围的通道：dq1_radps；policy_5000_d25_seed_4002 首次越过训练范围的通道：dq1_radps。

当状态越界先于多数动作限幅且契约检查通过时，证据支持闭环偏离后的反馈放大；这仍是时序证据，不能单凭本轮将根因唯一归结为关节速度、采样器或训练量。
## 受控续训与选模

沿用原 14 条训练、4 条验证演示，manifest 与归一化不变，不新增训练数据。schema=float_push_joint_v1；batch=16，AdamW lr=1e-4、weight_decay=0.01，梯度裁剪=1，观察/预测/执行窗口=4/16/4，DDPM 训练 100 步、推理 25 步、clip_sample=False。恢复模型、优化器、采样器及 Python/NumPy/Torch/CUDA RNG。每 250 步固定离线验证，每 500 步不可覆盖保存。

| 累计优化步 | 完整验证成功 | Wilson 95% | 最终平均门角 | reach / grasp / 解锁 | 验证噪声 MSE |
|---:|---:|---|---:|---|---:|
| 5000 | 0/10 | [0.0%, 27.8%] | 0.000000° | 0 / 0 / 0 | 0.020601 |
| 7000 | 0/10 | [0.0%, 27.8%] | 0.000000° | 0 / 0 / 0 | 0.018280 |
| 9000 | 0/10 | [0.0%, 27.8%] | 0.000000° | 0 / 0 / 0 | 0.016700 |
| 11000 | 0/10 | [0.0%, 27.8%] | 17.993398° | 0 / 0 / 3 | 0.015389 |
| 13000 | 0/10 | [0.0%, 27.8%] | 0.000000° | 0 / 0 / 0 | 0.013246 |
| 15000 | 5/10 | [23.7%, 76.3%] | 50.711315° | 2 / 1 / 7 | 0.013009 |
| 17000 | 1/10 | [1.8%, 40.4%] | 44.499912° | 1 / 0 / 7 | 0.012278 |
| 19000 | 0/10 | [0.0%, 27.8%] | 46.528584° | 0 / 0 / 6 | 0.012436 |
| 21000 | 1/10 | [1.8%, 40.4%] | 80.228450° | 0 / 0 / 9 | 0.010524 |
| 23000 | 0/10 | [0.0%, 27.8%] | 35.578440° | 2 / 0 / 4 | 0.009132 |
| 25000 | 0/10 | [0.0%, 27.8%] | 73.998036° | 3 / 0 / 9 | 0.009487 |

上述 Wilson 区间描述固定验证集的回合比例；该集合已反复用于选模，不能把这些区间或验证成功率当作独立测试性能。

最后 25000 步模型有 8/10 回合最终门角达到 80°，其中 8 回合尾端净距离不足 0.52 m；reach/grasp 分别为 3/10、0/10。平均门角提高不能代替完整过门成功。

候选 15000 步的 5 个数值主成功回合中，同时通过 reach 与 grasp 指标的只有 1 个。主成功判据没有要求稳定抓取；这些结果不能直接称为稳定抓把推门。阶段指标与原始轨迹一并保留。

| 累计优化步 | 前 1 秒触发多数限幅的回合 | 已触发者平均首次时间 | 前 1 秒平均限幅比例 |
|---:|---:|---:|---:|
| 5000 | 10/10 | 0.407 s | 45.52% |
| 7000 | 7/10 | 0.743 s | 19.03% |
| 9000 | 8/10 | 0.871 s | 9.62% |
| 11000 | 1/10 | 0.933 s | 7.63% |
| 13000 | 1/10 | 0.833 s | 10.08% |
| 15000 | 2/10 | 0.908 s | 10.03% |
| 17000 | 0/10 | —（全删失） | 9.48% |
| 19000 | 0/10 | —（全删失） | 10.75% |
| 21000 | 0/10 | —（全删失） | 7.92% |
| 23000 | 0/10 | —（全删失） | 4.88% |
| 25000 | 0/10 | —（全删失） | 12.13% |

未在前 60 步触发的回合按右删失保留，不填为 1 秒或无穷大；早期限幅指标只用于诊断。

![续训早期闭环对照](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled/plots/continuation_early_comparison.png)


连续三个完整验证点相较历史最佳成功次数和最终平均门角均无提高则早停（门角容差 0.001°）；loss、reach 和限幅时间不重置计数。最佳按成功率、最终门角、验证 loss 依次选择；同分取较早 checkpoint。测试种子 5000–5029 在选模冻结后使用，旧 3000–3009 未作为新测试。

![完整验证停止指标](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled/plots/validation_stopping_metrics.png)

![训练与离线验证](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled/plots/training_validation.png)

![闭环验证](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled/plots/closed_loop_success.png)

![独立测试](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled/plots/test_comparison.png)

![实际回合门角、把手与末端误差](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled/plots/rollout_metrics.png)

## 动作与运行诊断

| 方法 | 平均限幅比例 | 最终 EE 误差 | 决策 p50 / p95 |
|---|---:|---:|---:|
| expert | 0.00% | 1.085 m | 未完成 / 未完成 ms |
| untrained | 80.63% | 3.291 m | 32.32 / 40.40 ms |
| best | 43.46% | 3.102 m | 31.71 / 40.05 ms |
| last | 27.42% | 0.525 m | 31.52 / 39.20 ms |

untrained 最终门角/末端误差仅有 7/7 个有效末态样本（已启动 8 回合），异常回合没有伪造末态。限幅统计按 20244 个已记录执行步平均，包含异常回合保存的前 294 步；时延包含异常回合已发生的网络决策。reach/grasp/解锁仅计已记录确认事件，异常缺失不当作确认成功。

已完成固定验证的全回合统计（限幅按所有执行步/通道平均，时延按网络决策汇总）：

| 累计步 | 全回合平均限幅 | 平均最终 EE 误差 | 决策 p50 / p95 |
|---:|---:|---:|---:|
| 5000 | 77.39% | 29.421 m | 28.65 / 36.43 ms |
| 7000 | 78.57% | 18.958 m | 29.96 / 36.43 ms |
| 9000 | 67.03% | 25.730 m | 31.25 / 39.16 ms |
| 11000 | 44.13% | 1.429 m | 28.53 / 37.52 ms |
| 13000 | 40.09% | 1.383 m | 31.38 / 40.07 ms |
| 15000 | 45.56% | 2.779 m | 30.00 / 39.29 ms |
| 17000 | 35.75% | 0.696 m | 29.77 / 38.82 ms |
| 19000 | 42.99% | 0.870 m | 31.07 / 41.43 ms |
| 21000 | 41.79% | 0.861 m | 32.22 / 41.13 ms |
| 23000 | 46.01% | 1.392 m | 30.86 / 36.77 ms |
| 25000 | 33.88% | 0.482 m | 29.24 / 34.91 ms |

动作、末端误差与决策时延的原始记录保留在对应 metrics.csv、trajectory.npz 和 early_diagnostic.npz 中。未启动测试显示为未完成。推理时延仅统计真正网络决策，四步队列的取动作不算推理；首调用包含在原始记录中。

## 视频、预算与交付核验

完整预定测试：False；视频全部完整解码：True。每 2 仿真步采样，30 fps；完整视频应为 1425 帧、47.5 秒。首/中/尾帧分别对应仿真步 2/1426/2850，抽帧与逐步日志关联保存。12 个抽帧画面与对应时刻/门角/成功标注的人工核验：True；明细见 visual_review.json 与 video_frame_alignment.json。

| 预算阶段 | 完整进程墙钟 GPU 小时 |
|---|---:|
| preflight | 0.0086 |
| diagnostic | 0.1561 |
| validation | 1.6776 |
| train | 0.0860 |
| test | 0.4547 |

工程核验：预算正常退出、TERM 超时计费、互斥、最终预留、死进程恢复及不重复记账测试通过；25/100 步缓存推理与完整参考前向的最大误差均为 0（默认推理仍为 25）。完整 RNG 恢复审计=True；模型专家函数调用保护=True；自动开门扭矩为零=True。明细见 logs、resume-audit 和 training/resume_*.json。

本次恢复检查：原预算前缀、配置和 15000/21000 步模型哈希保持一致，预算命令 ID 不重复、训练步日志连续，既有完整验证点哈希不变；见 delivery_checks.json 与 logs/resume_cpu_checks*.log。

本轮新增训练循环累计约 261.5 秒（含循环内离线验证/保存）；与上表包含初始化和全部仿真的占卡进程墙钟分开报告。

- [test/expert_seed_5000/rollout_labeled_h264.mp4](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled/test/expert_seed_5000/rollout_labeled_h264.mp4)：1425 帧，30.0 fps，47.5 秒，解码核验 True。
- [test/untrained_seed_5000/rollout_labeled_h264.mp4](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled/test/untrained_seed_5000/rollout_labeled_h264.mp4)：1425 帧，30.0 fps，47.5 秒，解码核验 True。
- [test/best_seed_5000/rollout_labeled_h264.mp4](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled/test/best_seed_5000/rollout_labeled_h264.mp4)：1425 帧，30.0 fps，47.5 秒，解码核验 True。
- [test/last_seed_5000/rollout_labeled_h264.mp4](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled/test/last_seed_5000/rollout_labeled_h264.mp4)：1425 帧，30.0 fps，47.5 秒，解码核验 True。

四个视频均为 seed 5000：专家成功，未训练/最佳/最后模型失败，保留了完整代表性失败。测试中最佳模型的数值成功回合未开启视频；基础设施停止后未补录，不能用已有失败视频展示其成功。异常 seed 5007 未录视频，保存了部分轨迹和完整错误日志。视频抽帧与日志对照见 video_frame_alignment.json 和 video_contact_sheet.png。

服务器目录：`/data/zyf/video2door/experiments/float-push/20261012-005832-controlled`；本地目录：`/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-long-training/20261012-005832-controlled`。配置、源码快照、环境清单、原始 CSV/JSONL、分步轨迹与 PNG/PDF 图表均按 run_id 保存。本轮已核验 4 个 MP4，完整性逐一列于上文。必要模型包括未训练、5000 步起点、最佳和最后模型；全部分步 checkpoint 保留服务器。SHA256 清单用于同步校验。

原始 20 条数据、manifest、训练日志、预算日志及起点/未训练模型共 25 个文件与历史 SHA256 封存逐一核对：True；见 reused_data_provenance.json。

旧数据缺少未经处理的 Gym 原始深度与显式末步后状态；本轮仅复用其训练输入，不追补或伪造。新诊断专家回合和前 60 步记录保存原始 Gym 深度、处理深度、动作前观测、执行动作与显式后状态，且不进入训练 manifest。

## 结论边界

脚本专家成功、训练接口可运行、噪声 loss 下降和独立闭环成功分别以对应证据判断。此实验只覆盖门 99650089960001 和小范围初态随机化；浮动底座由运动学直接移动，未加载低层行走或平衡策略。模型使用 Gym 语义分割辅助的双相机 mask/depth。尾端净距离判据未检查门洞横向容纳或足接触，不能外推真实机器人或其他门泛化。

保留 2 次非零退出记录，均已计费；逐项见 budget.jsonl。历史预检曾因 diffusers 内部默认字段列表顺序触发配置误报，修正为比较实际扩散参数后重跑；本次独立测试的 PhysX CUDA 故障另见上述停止证据，未重跑。

本轮未完成全部实验：Final test infrastructure failure。已保存实际结果与未完成项，不宣称收敛或完整测试。
