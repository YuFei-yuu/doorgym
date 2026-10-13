# 浮动底座推门 DP 复现方案

更新：2026-10-13。主线：采用 Sivan DoorGym 的全身 DP 推门路径，将四足行走执行替换为运动学浮动底座，其余部分优先直接调用源仓库模块。

实施状态：20/20专家采集、5000更新训练和冻结模型续评已完成。模型验证开门1/3、通过0/3；测试开门0/3、通过0/3，尚未达到最终目标。专家及精确回放能达到90°并通过。用户已授权提高评估预算，续评未重新训练。详见 [结果报告](FLOAT_PUSH_V3_RESULTS.md) 和 [v3进展](FLOAT_PUSH_V3_PROGRESS.md)。冻结执行配置与源任务顺序保留，PhysX运行在单GPU上。

## 已确认的目标

用户已确认：

1. 任务成功标准为门沿正确方向达到80°，且机器人通过门。
2. 采用原全身 DP 的执行路径和参数，只替换底座运动方式。
3. 从随机初始化固定训练5000更新，依据任务结果再决定是否训练至原默认100000更新。
4. 原 Git 仓库已有的开门后处理保留；当前工作区新增的处理退出主线。

末端位置/姿态误差、关节速度、接触持续性用于解释执行表现，取消4cm、1cm、3cm等误差值对采集、训练和闭环评估的前置限制。专家能完成任务，其成功回合即可作为演示。

## 源仓库与复用范围

源码基线：[`Sivan666666/doorgym@0acc2f050479fb54a0670ccb90bc72b98b52a54a`](https://github.com/Sivan666666/doorgym/tree/0acc2f050479fb54a0670ccb90bc72b98b52a54a)。实施开始时再核对远端引用及两端源码。

| 环节 | 复用模块与方法 |
|---|---|
| 推门入口、任务与专家 | `high-level/play_b1z1_push_with_door_asset_camera.py` 的 `ManipLocoDoorAsset`、专家轨迹、门/把手状态、通过门流程 |
| DP 网络 | `high-level/dp/models/door_diffusion_policy.py`，直接导入 `DoorDiffusionPolicy` |
| 图像处理 | `high-level/dp/door_dp_common.py` 的 `dp_image_inputs_from_camera_tensors`；原腕/前相机、mask和masked depth、54×96 |
| 动作应用与推理 | 原 `apply_door_dp_action`、`DoorDPPolicyController`；保留原10D动作和窗口队列 |
| 采集与转换 | 原 raw recorder、`convert_door_raw_to_lerobot.py` 和 `DoorDPLeRobotRecorder` |
| 数据窗口与训练 | 原 `DoorDPSequenceDataset`、`compute_stats`、`normalize_batch` 及训练循环 |
| IK | `ManipLoco._control_ik` 与原 push `step` 的位姿误差构造、关节位置伺服 |
| 记录与交付 | 保留的 `checks.py/offline_eval.py/analyze.py/short_tools/deliver.py` 负责实验记账、指标和视频 |

原全身环境构造依赖四足模型、腿部传感器和低层策略。适配入口保留门、机械臂、相机与任务模块，将低层策略加载和腿部执行替换为浮动底座后端。能直接调用的函数直接调用；环境对象耦合的部分抽取成原入口和浮动入口共用函数，原入口默认行为保持一致。

## 执行配置

| 项目 | 新主线配置 | 源码位置 |
|---|---|---|
| 物理/控制频率 | dt=0.005s，decimation=4，控制和记录50Hz | `LeggedRobotCfg`、`B1Z1RoughCfg`、原push入口 |
| PhysX | substeps=2，位置/速度迭代12/4 | 原push参数 |
| IK | DLS damping=0.05；位置/姿态增益1.5/1 | 原 `_control_ik`、原push `step` |
| 手臂伺服 | stiffness/damping=400/40 | 原push资产设置 |
| 夹爪 | stiffness/damping=160/16，joint friction=120 | 原push参数与资产设置 |
| 夹爪接触 | contact/rest offset=0.018/0.003m，friction=8 | 原push参数 |
| 仿真接触 | contact/rest offset=0.02/0.002m，max depenetration velocity=0.5m/s | 原push参数 |
| 门物理 | 沿用原门锁/把手、阻力0.2、门摩擦0.5/阻尼0.2、开启阻尼0.05、自动开启力0 | 原push参数与 `_door_torques` |
| 回合长度 | 原录制默认2500控制步，即50s | `record_door_dp_dataset.py` |

完整原参数配置随run保存，包括资产路径、缩放、相机外参、碰撞参数、专家时序和初态。门资产从原全身路径解析；机器人采用源浮动split loader的 `b1z1_basearn.urdf`，机械臂安装位姿从实际加载URDF读取（0.20/0/0.10m）。按用户确认，机械臂初始化和行走默认姿态采用源浮动零位；停车距离采用源浮动默认0.25m。

浮动后端接收同一 body vx、yaw-rate，在四个物理步中推进底座位姿；vy固定0。IK使用同一时刻的末端测量和Jacobian。保留原IK误差和位置驱动逻辑，以坐标与时序适配处理运动学底座。

上一轮新增的6mm参考限速、6cm跟踪球、0.035rad姿态限步、0.03rad关节目标限步、8cm预测残差门槛，以及15cm/2rad/s经验停止条件退出新主线。数值合法性、URDF关节范围和仿真异常处理随执行后端保留。

## 原入口的开门后处理与适配前上游基线

已对固定上游 Git tree 与当前文件逐字节核查：

- `play_b1z1_push_with_door_asset_camera.py` SHA256为 `442d6e716313bd88662ee5e0035f4596a295a6970d5af2d5104784cb63c3cde7`，两者相同。
- `update_pass_state`、通过门导航和 `snap_arm_to_zero` 均来自该上游提交。
- `door_dp_common.py` 也与该提交相同。`apply_door_dp_action` 会覆盖脚本的底座/末端命令并清除freeze标志；此前执行的关节归零仍已生效。

GitHub当前HEAD核查仍为上述commit。直接下载的两个源文件与本地Git tree及工作区哈希一致。原push入口第2243–2294行更新任务与导航，第2295–2307行调用DP并应用输出，第2331–2332行调用低层策略和环境step。原common第542–546行重写底座与末端命令，第556–560行设置夹爪并清除freeze标志。

源码：[原入口调用顺序](https://github.com/Sivan666666/doorgym/blob/0acc2f050479fb54a0670ccb90bc72b98b52a54a/high-level/play_b1z1_push_with_door_asset_camera.py#L2243-L2332)、[原动作应用函数](https://github.com/Sivan666666/doorgym/blob/0acc2f050479fb54a0670ccb90bc72b98b52a54a/high-level/dp/door_dp_common.py#L537-L560)。

原入口在图像可用时应用DP输出；图像不可用时保留本帧脚本命令。新记录增加DP控制帧数、脚本命令帧数与图像有效性。

按用户确认保留原调用顺序：更新专家/任务状态及原通过门处理 → 读取DP观测 → DP输出覆盖命令 → 浮动执行后端 → 物理仿真。报告同时记录DP动作、原开门后处理触发时刻和实际执行状态。

实施后原全身入口新增可选backend、记录与计时钩子。原专家、导航、门力矩、相机和 `snap_arm_to_zero` 函数体保持与固定上游AST一致。源浮动推门目标曾经用户确认抽取共享函数作对照，其6种分支输出与固定源代码一致；正式采集使用原全身专家，不启用该对照目标。源码快照与哈希随每个回合保存。

## 状态、动作与数据

动作保持原10D：body vx、yaw-rate、世界末端xyz、世界xyzw、夹爪目标。

浮动状态保留原字段的真实27D子集：底座roll/pitch2D、body角速度3D、臂/夹爪实际q和dq各7D、相对机械臂安装原点且沿底座轴表达的末端位置3D、世界末端xyzw4D、实际夹爪1D。四元数归一化并保持符号连续。原腿部状态、足接触和上一低层网络动作随行走后端移除。

新版本为 `float_push_upstream_ee_v3`，执行版本为 `upstream_wholebody_float_v3`；50Hz、全身IK/接触配置及原开门后处理写入数据和checkpoint元信息。主数据重新采集。

复用原raw记录字段与LeRobot转换。附加记录执行前 `obs_t/images_t`、IK前命令 `a_t`、物理步后 `obs_t+1`、IK/伺服关节目标和实际关节；训练窗口使用执行前观测与对应命令。原容器、图像处理和窗口索引保持复用，浮动入口负责采集时刻与状态字段适配。

训练/验证/测试仍使用分离的初态种子。沿用此前16个训练初态、4个离线验证初态、3个闭环验证初态及3个完整任务测试初态的规模；实施前扫描留存记录，分配新的完整种子块。名义专家先确认可完成任务，随后采集预定集合。原raw recorder按pass_started保存开门成功回合；训练/验证manifest选取完成原pass_done流程的完整专家演示，每次尝试保存任务结果。

## 训练和推理

| 项目 | 配置 |
|---|---|
| 网络、损失 | 原 `DoorDiffusionPolicy`，epsilon noise MSE |
| 优化器 | AdamW，batch64，lr1e-4，weight decay1e-6，梯度裁剪1 |
| 观测/预测/执行窗口 | 16/32/16 |
| 扩散 | DDPM cosine，训练100步，推理100步，clip_sample=True |
| 归一化 | 原z-score和std下限1e-6；原统计函数只接收训练集合 |
| EMA/学习率调度 | 原训练循环均未使用，保持此行为 |
| 更新/种子 | 随机初始化，seed42，固定5000更新 |
| checkpoint | 原入口完成5000更新后保存的model_latest.pt；复制冻结并记录更新数和SHA256 |
| 运行线程 | 两线程；DataLoader workers=0，配合现有资源约定 |

上述训练参数取自源仓库默认。100000更新作为5000更新验证后的下一阶段选项。

实施中优先走原raw→LeRobot→原train入口；原环境的依赖通过现有运行环境解决。若必须使用NPZ容器，则继续继承原序列类和统计函数，记录容器适配。小样本拟合、专家观测动作误差和不同采样设置用于定位训练接口，不设置厘米级误差门槛阻断主任务验证。

已建立服务器独立Conda训练环境，继续使用原LeRobot数据读取。其解码图像为[0,1]浮点，原读取函数增加恢复uint8像素的兼容处理，逐像素往返检查通过。用户确认不使用帧缓存，增加训练时限以保持原读取方式。

## 评价任务完成情况

原入口在正向门角达到80°时锁存 `pass_started`。通过门目标沿用原几何：

- 首次达到80°时锁存侧 `s=sign(robot_x-door_root_x)`。
- 目标 `x=door_root_x-s×1.40m`。
- 目标 `y=door_root_y+pass_center_y_offset-s×0.10m`。
- 底座到目标的平面距离≤原 `pass_align_pos_tol=0.12m`。

主成功记为本回合曾达到正向80°，随后机器人到达原通过门目标；原pass_done阶段标记一并记录。专家和DP使用同一几何记录；任务标记保存首次开门、首次通过及最终状态。

固定最后checkpoint先做专家观测动作误差统计，再运行60、300、900步短程，随后运行2500步完整回合。短程用于检查运行和观察执行，完整回合按开门并通过的任务结果计成功。报告成功数/全部尝试数、门角、底座路径与视频；位置/姿态误差、关节速度和接触持续性作为配套曲线。

至少交付一组同初态、同执行版本的专家/模型并排视频。所有新日志、数据、图和视频写入新run目录并下载本地。

## 实施顺序和资源

建议run_id：`20261013-upstream-float-v3`。旧训练/执行适配已清理；现保留原仓库模块、通用分析与资源工具，下一阶段实现浮动执行适配并训练。

1. 核对两端HEAD/工作区、上游commit、活动进程；保存原参数与源码哈希。
2. 适配原全身环境的浮动后端，保持原任务、相机、IK和开门后处理。
3. 跑名义专家完整任务及动作回放，确认能够开门并通过；冻结执行版本。
4. 采集新成功演示，检查字段、时序、图像和集合划分；转换原训练容器。
5. 随机初始化完成5000更新，冻结最后checkpoint。
6. 执行离线、短程和完整任务评估，交付同初态专家/模型视频及任务统计。
7. 根据5000更新的任务结果讨论100000更新阶段。

用户已将GPU进程墙钟总预算提高至19800s，阶段累计上限为执行1800/采集1500/拟合300/训练14400/评估1800s。新run继承服务器 `.repro/retained-evidence/20261013/budget.jsonl` 的176.587700s历史消耗；实际用量以run账本为准。授权与调配依据保存在 `budget_authorization.json`。每次GPU启动前资源采样，单GPU、nice10、两线程，使用现有launch/bounded工具。

## 文档与记录入口

- [数据指标与绘图](FLOAT_PUSH_METRICS.md)：源日志字段、任务指标和可视化。
- [环境配置](环境配置.md)：pro5000运行与依赖。
- [新对话提示词](FLOAT_PUSH_NEXT_PROMPT.md)：可直接复制执行。
- [清理记录](FLOAT_PUSH_CLEANUP_REPORT.md)：旧内容删除范围与精简证据位置。
