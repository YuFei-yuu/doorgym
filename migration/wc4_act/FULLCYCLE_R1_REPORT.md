# wc4 完整任务专家首轮：停止交付，未通过完整任务验收

日期：2026-10-20（Asia/Shanghai）。依据用户首轮计划书，最多3回合、单环境、同一名义初态、每回合最多90秒；一次完整成功或任一停止条件触发即停止。本轮没有训练、模型推理、规模采集、状态回放或失败替换。

## 结果与停止依据

实际启动 **1回合，完整成功0回合**。`attempt_000` 在第569步（零起算），仿真11.399999745秒停止。没有启动后续回合。

停止时阶段为 `rotate_handle`。PhysX返回两条 `z1_arm_articulated:gripperStator / door_0:fixed_frame` 非预期接触记录，均为 `lambda=0`；`minDist≈0.04m`，`initialOverlap≈0.03944/0.03927m`。这是保守接触守卫触发，**不构成承力碰撞、冲击或穿透深度的证明**。接触距离与真实几何间隙的关系需要独立审查，不能把 `initialOverlap` 直接解释为穿透深度。本轮没有修改守卫、接触裕量或物理参数后重跑。

全身通过未达成；未进入远侧调整、关门、重新锁闭、释放或最终保持窗口。最大实测门角仅0.036778°；解锁标志出现5步。这些局部状态不构成完整任务完成，也不能据此断言协调关门不可行。

## 版本、隔离与运行绑定

- 起点：本地及服务器干净 `main@87c5a8665f78b0627c95ea9be6a0e756fc359b11`。
- 开发分支：`expert/wc4-fullcycle-r1-20261020`，没有合并到main。
- 实际仿真代码提交：`cf67d071bdf045a07e1cfb295cdda1ceb9780de1`，本地/服务器一致。后续交付提交只增加证据分析入口及本报告，没有继续执行仿真。
- 固定作者源：`096938c44ecc1c7602304d42502f96658e46eb25`。新增实现位于 `migration/wc4_act/fullcycle_*.py`，没有修改原作者核心代码、资产、原skill或成功基线。
- 服务器run：`/data/zyf/video2door-wc4-act/.repro/runs/wc4-fullcycle-nominal-r1-20261020`。
- 本地完整镜像：`/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-fullcycle-nominal-r1-20261020`。
- GPU：物理2，UUID `GPU-2c8bfc35-6335-164d-bba2-5c97c3d9cfd0`。启动前连续三次检查显存<1024MiB、利用率<10%、无compute进程，持有共享UUID锁至进程退出。CUDA/Vulkan映射为逻辑0；已释放租约。没有影响其他GPU任务。
- Gym Python3.8.20、Torch2.3.1+sm120、NumPy1.23.5；复用 `/data/zyf/video2door/.repro/envs/b1z1-blackwell`。ACT环境仅用于CPU证据解码，未加载模型。没有安装包或重建环境。

`source-head-attempt_000.txt`、`worktree-attempt_000.patch`、`environment-attempt_000.txt`、资源快照和 `attempt_000/command.json` 记录实际绑定。参数预检发生在代码提交前，当时HEAD仍为87c5a86，不能与正式回合版本混淆。服务器缺少rg，盘点使用grep；本地部分克隆的历史对象比较受网络限制，固定作者保留源码比较在服务器完成。

## 初态、控制与物理

seed62000，单环境，名义root `(4.1,0.18,0.5)`，pitch0，yawπ；门配置wc4，door_x2.5、door_y0、z_offset0.01，实际铰链XY `(2.467,-0.463)`，关闭角0。初态、相机及物理随机化关闭；保留现有配置默认值和wc4倍乘，不手工调整物理。实际值见 `attempt_000/effective-config.json`、`reference.json`、`initial-state.json`。

门铰链自动力矩、开门阻力、开门阻尼、锁力、摩擦和阻尼均0；**没有门板自身复位力帮助关门**。把手弹簧刚度0.25、阻尼0.05，把手关节摩擦/阻尼均0.025；把手弹簧作用于把手，不是门板闭合力。

开门段复用原skill目标和gym_jacobian DLS，将关节与底盘命令统一按25Hz生成、50Hz零阶保持执行。相较原专家50Hz更新IK，这存在控制时序差异；因此本次接触不能直接归因为已验收基线本身的必然失败。底盘每步严格调用原Joint9前进/转向积分器，无侧移；浮动底座按既有抽象设置root姿态，不属于动力学轮式底盘控制验证。

六轴home为全0；既有控制范围裁剪后gripper home为-1.177500rad，初始实测夹爪为-1.570796rad。首个夹爪目标存在0.393296rad阶跃，已保留，不将整条命令称为C2平滑轨迹。旧专家程序在同初态下的目标终向为π；完整旧专家的实测终态未另跑，本回合也未达到终态。

## 离线方案比较与边界

在服务器Gym环境CPU执行两次离线检查；没有创建仿真器。第一版筛候选，第二版补充远侧绕行、展臂和固定底盘比较；第二版才是正式候选依据。使用URDF primitive/mesh包围角点、六轴全姿态IK和31个关门角采样；不是连续空间/动力学可行性证明。

选择协调圆弧：接触半径0.65m、底盘后置距离0.75m、末端高度0.95m；六轴目标约 `[-1.18915,1.98205,-1.18389,-0.79816,0.33245,1.57080]rad`。关门运动由前进速度和转向速度表达，固定机械臂姿态随底盘绕铰链运动。离线姿态IK残差约6.1e-14m / 4.6e-13rad；非末端几何到门板无限平面的最小采样裕量0.2063m。远侧绕行及展臂采样最大X为2.38130m，没有保守包围盒与90°门板相交候选。关闭扫掠中最大X为2.462m，使用门洞中平面X2.5作为远侧检查面。

相同接触轨迹、底盘固定在中段候选姿态的比较，最大位置残差0.35323m、姿态残差0.43688rad。这只排除了这个固定姿态候选，**没有证明所有固定底盘方案无解**。

几何离线检查未验证门框PhysX接触距离，也没有证明接近/转把手阶段无接触；本回合正是在这部分触发停止。门洞中平面与门框远侧表面不同，40mm接触距离、几何裕量、1°关闭容差的综合兼容性尚未完成验证，不能据离线角点判为全任务可行。

新入口实现了关闭角误差≤1°且速度≤1°/s保持25步后锁存实测角；随后释放、收臂和同时满足全部最终条件100步的门槛。锁存不是吸附0。**这些新增运行阶段均未被本回合执行验证**，不能视为已完成工程能力。

原始初始锁抽象保留，在首次解锁前锁存初始实测角0，最大纠正量0.000641898rad（0.036778°）；最后一次初始锁写发生在565步，该步后首次报告解锁。没有重新锁闭写入，也没有在关门阶段写角度。逐条写入记录见 `lock-writes.jsonl`。

## 连续性、动作和观测核验

- 连续570条状态、570条接触日志；`trajectory.npz`为570×27，逐值与JSONL一致。状态记录是每步执行后实测值，动作是该步使用的命令；学习前必须明确时序，不能直接假定与训练样本对齐。
- 保存动作重建的底盘命令轨迹误差0，实测root XY与命令误差0；没有侧向动作。没有重新运行固定动作回放，因此只证明动作可表达及日志一致，未证明完整动力学复现。
- 六轴相邻指令最大增量0.025683rad；夹爪最大0.075358rad（不含首次初始化阶跃）。六轴最大实测/指令差0.051132rad，夹爪0.270638rad。末端目标/实测最大位置差0.134513m，不能用IK目标连续代替实际跟踪。
- 从实测关节/root重建末端FK与原始实测末端位置最大差7.56e-7m，支持本回合坐标变换一致性；不是所有碰撞网格和接触裕量的验证。
- 双深度均保存为640×480、25fps补充视频，深度范围0.2～1.5m、legacy相机。由于实际没有关门段，**双深度能否区分开门/关门未验证**；未增加阶段标签、未来信息、策略回退或网络输入。
- 整体原始RGB为1920×640双侧视角，已检查原片提取的全程接触表，全身和门在接近侧画面可见；远侧画面在本回合有门体遮挡。三段视频均完整解码285帧、25fps、11.4秒，时间戳连续。录像覆盖实际已执行的回合，不是90秒补片或状态回放。

## 证据导航与命令

run根目录：`attempts.jsonl`、两版离线结果/日志、预检日志、正式日志、退出码、源码版本、GPU快照。`attempt_000/`：初态、有效配置、body-map、参考、570步 `states.jsonl`、全部原始 `contacts.jsonl`、初始锁写入、`trajectory.npz`、两路深度及 `overview/env_0000/rollout.mp4`。`audit/`：视频解码/动作/接触核验、FK复核、轨迹逐值检查、原片接触表。最终 `SHA256SUMS` 覆盖交付全部证据。

正式命令在服务器项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B migration/wc4_act/fullcycle_offline.py --output .repro/runs/wc4-fullcycle-nominal-r1-20261020/offline
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B migration/wc4_act/fullcycle_offline.py --output .repro/runs/wc4-fullcycle-nominal-r1-20261020/offline-v2
bash migration/wc4_act/fullcycle_launch.sh preflight
bash migration/wc4_act/fullcycle_launch.sh attempt_000
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LD_LIBRARY_PATH=$PWD/.repro/envs/door-act/lib .repro/envs/door-act/bin/python -B migration/wc4_act/fullcycle_audit.py .repro/runs/wc4-fullcycle-nominal-r1-20261020
```

第一版离线检查在未提交代码期间运行，未另存执行前源码快照；其结果和日志保留，最终源码提交对应第二版。不能把最终脚本冒充第一版执行源。命令不可直接覆盖已有输出。实际回合完整argv以 `attempt_000/command.json` 与启动脚本为准。初始化、参数预检、CUDA最小运算和离线分析均非额外专家回合。

停止交独立验收。本轮不宣称名义完整专家成立、鲁棒、可规模采集或可开始训练。未提出或执行下一阶段实验。
