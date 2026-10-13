# 当前方案：浮动底座推门 DP

## 目标与已验证状态

门沿正确方向达到80°，机器人沿接近方向穿门并完成 pass_done。当前专家在9组位置×2个种子的18回合中全部成功；forward方案尚未训练DP。

本地和服务器入口见[交接文档](FLOAT_PUSH_TRAINING_NEXT.md)。固定源提交为 Sivan666666/doorgym@0acc2f050479fb54a0670ccb90bc72b98b52a54a。

## 复用关系

| 环节 | 实现 |
|---|---|
| 原全身任务与专家 | high-level/play_b1z1_push_with_door_asset_camera.py |
| 浮动底座执行 | high-level/float_ik/learning/wholebody_float.py |
| 浮动资产与底座位姿 | 源split loader、b1z1_basearn.urdf、set_robot_base_pose |
| IK与关节伺服 | ManipLoco._control_ik及原全身位置驱动 |
| 相机、门与把手物理 | 原全身环境模块 |
| DP网络 | high-level/dp/models/door_diffusion_policy.py |
| 图像、动作应用与推理 | high-level/dp/door_dp_common.py |
| 数据转换与训练 | convert_door_raw_to_lerobot.py、train_door_dp.py |

控制顺序：原专家/任务/导航更新 → DP观测与输出覆盖 → IK/浮动执行 → 物理仿真。保留源 snap_arm_to_zero 开门后处理。

## 直接向前通过

浮动入口默认 float_pass_mode=forward：

1. 达到80°后直接进入 pass_through，跳过门前重新对齐。
2. 用初始接近侧 s 确定门后目标，避免底座已经越过门平面时目标反向。
3. 目标仍为 x=door_x−s×1.40m，y=door_y+center_offset−s×0.10m；距离≤0.12m时完成。

可显式使用 float_pass_mode=source 重现旧流程；原四足入口继续默认source。两种模式使用不同 execution_profile。

## 执行配置

| 项目 | 当前值 |
|---|---|
| 底座后端 | --float_execution source；源set_robot_base_pose |
| 仿真 | CPU tensor pipeline、单GPU PhysX |
| 频率 | dt=0.005s、decimation=4，即50Hz；substeps=2 |
| PhysX迭代 | 位置12、速度4 |
| 初态 | x=4.1、y参数=0、z=0.50m、yaw=π；手臂为源浮动零位 |
| 接近与推门 | 停车0.25m、推门vx=0.25m/s、专家步长0.06m |
| IK | DLS=0.05，位置/姿态增益1.5/1 |
| 伺服 | 手臂400/40；夹爪160/16 |
| 任务时长 | 最多2500控制步；专家成功后结束 |

robot_y由源入口叠加把手中心偏移，实际底座初态写入记录。门、接触、相机和URDF安装位姿等完整配置保存于每回合 config.json。

## 数据与训练

- 状态27D：底座roll/pitch、角速度、臂/夹爪q与dq、末端相对安装原点的位置、世界xyzw、夹爪位置。
- 动作10D：body vx、yaw-rate、世界末端xyz、世界xyzw、夹爪目标。
- 四路图像：腕/前相机的把手mask和masked depth，54×96。
- 同步保存执行前观测/动作、执行后状态、IK目标、接触与源snap事件。
- 当前schema为 float_push_upstream_ee_v3，执行profile为 upstream_wholebody_float_v3_forward_pass。

原训练参数：batch64、AdamW lr1e-4、weight decay1e-6、梯度裁剪1；观测/预测/执行窗口16/32/16；100步DDPM cosine，epsilon MSE；z-score统计仅来自训练集合。原循环无EMA及学习率调度。

保留原LeRobot逐帧读取。训练读取函数已处理LeRobot浮点[0,1]图像到uint8的转换。新训练使用forward演示及独立集合；旧source模式数据/模型保留作对照。

## 验证

使用完整回合的80°开门和 pass_done 统计全部尝试。训练loss、末端误差与接触曲线用于解释表现。保留同初态视频、源码哈希、数据划分和checkpoint更新数。当前专家结果见[验证报告](FLOAT_PUSH_FORWARD_EXPERT_VALIDATION.md)。
