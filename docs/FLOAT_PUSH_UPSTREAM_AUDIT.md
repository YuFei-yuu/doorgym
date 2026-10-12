# 浮动底座 DP 与原仓库源码对比

核对日期：2026-10-12。本文对比基线为清理前 `d0b7390`；后续移除旧执行分支及分析工具适配见 [清理报告](FLOAT_PUSH_CLEANUP_REPORT.md)，当前训练/控制参数仍沿用此处所述的末端基线。原仓库在线 `main` 为 [`0acc2f050479fb54a0670ccb90bc72b98b52a54a`](https://github.com/Sivan666666/doorgym/tree/0acc2f050479fb54a0670ccb90bc72b98b52a54a)，本地/服务器源码为 `d0b739098aded310da23d9034c0bfbdfe6557b64`，整理前工作区均干净。本地 `origin` 指向 `YuFei-yuu/doorgym`，不是本次比较的 Sivan 原仓库。

通过 `git ls-remote` 核对原仓库引用，并确认该提交已存在于本地且是 HEAD 的祖先；直接比较固定提交的 Git tree 与本地实际文件。原有 `high-level/dp/`、浮动专家/IK、全身执行及机器人资源的跟踪源码未改变；新增学习适配集中在 `high-level/float_ik/learning/`。其余差异主要是文档、忽略配置和删除 `.DS_Store`。提交间 diff 共 70 个文件条目，包含文档移动；这不是 70 个算法改动。完整路径和关键文件 SHA256 见 [机器可读核对记录](audit/FLOAT_PUSH_UPSTREAM_AUDIT.json)。未将 Git 忽略的实验产物、外部依赖或环境安装视为相同。

**结论：复用了原网络和原仓库浮动 IK，但没有原样复用原全身 DP 训练/执行配置。** 网络文件相同不等于参数形状、动作语义、时序、数据分布或闭环行为相同；本轮没有重新训练原全身 DP，因此下面不能作为两种方法性能优劣的实测结论。

## 两条“原仓库”路径必须区分

原仓库同时有全身 DP 路径和独立浮动脚本。本次结合前者的网络与后者的物理/IK 工具：

| 路径 | 底座如何移动 | 机械臂如何移动 |
|---|---|---|
| 原 `dp/play_door_dp_policy.py --mode push` | 调用 `play_b1z1_push_with_door_asset_camera.py`；DP 给 vx/yaw 指令，低层策略输出腿部动作，经 PD 力矩和足地接触产生运动 | DP 给世界系 EE 位置/四元数，`B1Z1DoorAsset.step()` 调用继承的 `_control_ik`；当前关节角加 IK 增量成为位置伺服目标。低层网络的手臂动作被清零，手臂力矩通道也置零，手臂依靠位置驱动 |
| 原独立 `float_ik/*push_door.py` | 脚本轨迹设置浮动 actor 位姿，无行走/平衡策略 | 原通用浮动 IK 加位置伺服；部分回家阶段原来有直接关节插值 |
| 当前 `learning/FloatEnv` | 模型给 body vx/vy/yaw-rate；60 Hz 积分后用 `set_robot_base_pose` 更新运动学底座，z 固定 0.60 m | 模型给世界 EE 目标，调用原浮动 `set_ik_target` / `update_arm_ik_targets`，再 `set_actor_dof_position_targets`；专家回家阶段也先 FK 生成末端标签，再经过同一 IK |

核对源码：[原 DP 分派](../high-level/dp/play_door_dp_policy.py)、[全身 push 执行](../high-level/play_b1z1_push_with_door_asset_camera.py)（`step`，约 1184 行；低层 policy 调用，约 2331 行）、[原低层 IK/力矩](../low-level/legged_gym/envs/manip_loco/manip_loco.py)（`_control_ik` / `_compute_torques`）、[当前执行](../high-level/float_ik/learning/float_env.py)（`expert_action` / `step`）。不能把另一个 `high-level/envs/b1z1_base.py` 当作此 DP wrapper 实际调用的环境。

## 影响表现的具体差异

“原 DP”数值为固定源码的默认配置，不代表其作者成功模型一定使用这些参数；“当前”来自代码及本轮保存的实验配置。

| 项目 | 原 DP 路径 | 当前浮动适配 | 影响与判断 |
|---|---|---|---|
| 网络 | `DoorDiffusionPolicy`，四路视觉编码和条件去噪网络 | 同一文件逐字节复用，SHA256 `7fbb04bb32f1d60fc11581b0920bd0ef023fa0df4030b952bc44b5fefd9d9cbc`；输入/动作维度和窗口改变 | 不是另一种网络，但权重形状和条件分布不同，不能直接搬 checkpoint |
| 动作 | 10D：vx、yaw-rate、世界 xyz/xyzw、gripper；vy=0 | 11D：增加 body vy，其余仍是世界末端目标 | 自由度增加；底座速度实际由运动学强制实现，与有惯性/足地接触的行走不同 |
| 状态与反馈 | roll/pitch、角速度、全身 q/dq、上一低层动作、脚接触、末端位置/姿态和夹爪；无专家阶段输入 | 39D：相对初始底座位置/yaw、上一底座指令、7 个 q/dq、基座系 EE 位姿、上一条已执行高层指令；无门 GT/阶段/时钟 | 上一世界末端指令直接进入下一次预测；大目标会污染闭环输入，现有有限性检查挡不住巨大但有限的值 |
| 坐标表述 | 动作为世界目标；观测 EE 位置相对臂基座、姿态取 `env.ee_orn` | 动作为世界目标；观测 EE 位姿在基座系，底座位置相对初始 | 都不是简单“全局状态→全局动作”；当前 reset 随机位移未作为显式绝对原点输入，需检查视觉是否足以消除歧义。是待测风险，不能断言坐标实现错误 |
| IK 与伺服 | `_control_ik` DLS damping=0.05，外部位置/姿态增益默认 1.5/1；手臂 stiffness/damping=400/40；夹爪单独 160/16 | 复用浮动工具：DLS damping=0.08，位置/姿态增益 1/0.7，旋转行权重 0.1；IK 求解增量上限 0.06 rad；位置驱动 80/8 | 当前位置/姿态控制更柔和，姿态权重也不同；接触跟踪和抖动均可能改变。关节边界投影可能让最终目标与实测之差超过 0.06，不能把它当绝对关节跳变上限 |
| 控制时序 | 低层配置 dt=0.005、decimation=4，默认 DP/记录每 0.02 s 即 50 Hz；同一 IK 目标保持四个仿真步 | 60 Hz 每步更新底座、IK 和物理 | 同一帧数代表的时间不同；控制频率、刚度和接触响应需一起看，不能单比轨迹帧间位移 |
| 窗口/重规划 | 默认 obs/pred/action=16/32/16 | 4/16/4 | 观测跨度从约 0.30 s 降至 0.05 s；执行块从 0.32 s 降至约 0.067 s。更短历史与更频繁重新采样可能影响阶段判断和连续性，尚无隔离对照 |
| 扩散采样 | DDPM cosine、epsilon、训练 100 步；默认推理 100 步；`clip_sample=True` | 相同噪声目标，训练 100 步，推理 25 步；`clip_sample=False` | 当前去噪结果无标准化范围裁剪，是发散排查的高优先项。原裁剪默认作用于标准化样本，配合 z-score 可能压掉超出均值±1σ的真实专家动作；不能未经离线覆盖检查直接恢复 |
| 优化和步数 | 默认 batch=64、AdamW lr=1e-4、weight decay=1e-6、100000 更新 | batch=16、同 lr、weight decay=0.01、此次仅 2000 更新 | batch、正则和训练量均变。噪声 MSE、梯度裁剪 1 相同；两份训练实现都没有 EMA 或学习率调度。短训是验证，不是原训练成绩复现 |
| 归一化/采样 | 提供数据集所有帧统计，std floor=1e-6；shuffle 有效窗口，开头不足历史的窗口排除 | 只用合格 train 回合统计，floor=0.01；先均匀抽回合再抽窗口，开头重复初帧补历史 | 当前数据隔离更明确，但标准化尺度/采样权重不同，低方差状态与动作通道尤其受影响；当前完整回合同长度，回合抽样差异较小 |
| 观测—标签时序 | push recorder 在 `env.step` 后记录新观测与刚执行/保留的末端目标，训练从同一帧索引取动作 | 明确保存动作执行前 obs_t、应用动作 a_t 及 next_state | 是实际语义差异：原录制路径具有一步偏移风险，不能为“复用”直接照搬；应保留当前前后帧契约并用回放核验 |
| 视觉 | 腕/前相机 handle mask + masked depth，54×96，灰度复制 RGB，每帧有效深度 min/max 映射 | 同通道语义与分辨率、同网络编码；直接取浮动 Gym segmentation/depth，灰度在模型入口展开 | 大体复用，但相机挂载、浮动几何、裁剪和视野不同；两者均有分割辅助，不是纯 RGB。逐帧深度映射不保留统一绝对深度尺度 |
| 专家/数据覆盖 | 全身脚本、低层行走及其接触条件，原 DP 训练实际数据量/成功率本次未核验 | 浮动脚本加冻结接触时暂停底座、参考 6 mm/步/0.035 rad 限速；新采 16 train+3 val，仅单门及小初态扰动 | 新教师已经改变；当前默认策略没有同样参考限制，训练标签平滑不代表策略执行有相同约束。部署偏离专家状态后的覆盖不足仍是待测因素 |
| 物理/任务判据 | 全身机器人、行走动力学；原记录脚本按 pass_started/pass_done 等阶段保存 | 分离浮动 actor、无行走；门自动开启力=0；主成功只需解锁、最终正向≥80°、尾端净距≥0.52 m | 不能跨环境直接比较成功率；当前主判据不要求稳定抓把或侧向完整穿门，可能出现“成功但动作错误” |

训练差异依据：[原 train](../high-level/dp/train_door_dp.py)、[原 controller/特征](../high-level/dp/door_dp_common.py)、[当前 schema](../high-level/float_ik/learning/schema.py)、[当前 train](../high-level/float_ik/learning/train.py)、[当前 policy](../high-level/float_ik/learning/policy.py)。执行差异依据：[原浮动 IK](../high-level/float_ik/isaacgym_visualize_b1z1_basearn.py)、[当前 action 处理](../high-level/float_ik/learning/ee_actions.py)、[当前采集](../high-level/float_ik/learning/rollout.py)。

当前推理缓存视觉条件和时间编码以减少重复计算；固定窗口、固定噪声的 CPU 全网络/缓存路径对比差异为 0，仅证明该测试的计算等价，不代替 GPU 全链路证明。学习适配未改写原网络文件，也没有以 PPO/DAgger 训练当前模型。

## 有证据的问题与尚未确认的原因

已观察到：专家观测下 xyz 向量 RMSE 约 0.221 m；原始模型首次目标偏移约 0.401 m，第 12 个索引目标范数超过 10 m，随后增长到约 2.08e36 m，再出现 CUDA 700。动作仍可为有限数，位置误差的 float32 运算却已溢出。这证明当前检查和目标范围约束不足，不能把故障简单归咎于硬件、显存或 IK 本身。

后加诊断限幅器把世界目标限速到 0.36 m/s，但底座允许 0.6 m/s；顺序做“实测附近投影”和“上一目标限步”不能同时满足两者。模型三个回合跟踪误差 P95 达 7.91–9.09 m，均无 reach/grasp。它没有成为默认配置，不能作为已修好的执行器使用。

尚未确认：`clip_sample=False`、短时间窗口、上一指令反馈、数据覆盖、IK/伺服参数分别贡献多少；尚无单因素实测归因，CUDA 错误的精确触发机制也未证明。下一步应先修复并验收执行边界，再做冻结数据/模型的离线及短闭环对照，最后决定训练改动。不要同时恢复所有原参数、把增加训练步数当已证实答案，或把新 IK 失败证据也清掉。具体目标见 [当前交接](FLOAT_PUSH_TRAINING_NEXT.md)。
