> 历史首轮报告。当前正式实验已到 25000 步，最新状态见 [下一轮入口](FLOAT_PUSH_TRAINING_NEXT.md) 与 [正式报告](FLOAT_PUSH_LONG_TRAINING_REPORT.md)。服务器连接与本地链接已按 `pro5000` 维护；原始实验封存未改写。

# 浮动底座推门首轮训练与独立评估报告

生成时间：2026-10-12 00:09:02 中国标准时间

本轮完成了原专家复测、同步数据录制、原仓库 Diffusion Policy 网络训练、断点恢复、独立 Gym 闭环评估和视频交付。**这是一轮有界可行性实验，不是正式 6 GPU 小时训练。正式训练尚未启动。** 早期录制未完整保留原始传感器深度和最后一步后状态，详见数据限制；本轮模型闭环结论以真实 rollout 为准。

最终独立测试：专家 **10/10**，未训练 DP **0/10**，按验证集选出的 **5000 步模型 0/10** 达到最终开门并过门的数值条件。训练/验证 loss 的下降与闭环表现分别报告。

## 独立闭环结果

测试初态种子固定为 3000–3009，与采集和选模验证种子不重叠。所有已启动回合均进入分母；每回合 2850 步、47.5 秒仿真时间。成功定义为已解锁、最终正确方向门角 ≥80°、底座尾端净距离 ≥0.52 m。

| 策略 | 成功次数 | 成功率及 Wilson 95% 区间 | 最终平均门角 | reach / grasp / 解锁 |
| --- | --- | --- | --- | --- |
| 脚本专家 | 10/10 | 100.0% [72.2%, 100.0%] | 89.137° | 10 / 10 / 10 |
| 未训练 DP | 0/10 | 0.0% [0.0%, 27.8%] | 0.000° | 0 / 0 / 0 |
| 5000 步 DP | 0/10 | 0.0% [0.0%, 27.8%] | 0.000° | 0 / 0 / 0 |

未训练 DP 使用随机网络权重，但沿用训练集归一化统计、相同动作执行器和推理配置；它不是与示范数据完全无关的均匀随机动作基线。

| 固定验证 checkpoint | 成功次数 | 最终平均门角 | 验证噪声 MSE |
| --- | --- | --- | --- |
| 0 | 0/5 | 0.000° | 未训练 |
| 100 | 1/1 | 89.818° | 0.188451 |
| 500 | 0/5 | 0.000° | 0.089621 |
| 2000 | 0/5 | 0.000° | 0.040132 |
| 5000 | 0/5 | 0.000° | 0.020601 |

100 步 checkpoint 只做过一个接口 smoke 回合，不能与五回合结果直接比较，也不参与最终选模。该回合数值成功但 reach/grasp 保持指标未通过，说明不能仅凭门角和过门距离宣称稳定抓取。完整验证使用固定种子 2000–2004；选模规则依次为最终完整成功率、最终平均门角、较低验证 loss。测试结果没有用于选模。

所选模型失败分类：`{'unreached': 10}`。
所选模型测试平均动作通道限幅比例为 **77.34%**。该指标按所有时间步和通道统计；原始网络动作与实际下发动作均保留。
测试各回合决策时延 p50 范围 **26.23–32.17 ms**，p95 范围 **32.00–40.94 ms**。一次决策生成并执行 4 步动作；未将排队动作当作网络推理。仿真在推理时等待，因此视频 1 倍仿真速度不代表实际墙钟实时性。

每个新进程的首决策（包含预热影响）为 120.59–142.16 ms；该值单独保留，不与模型加载时间混淆。
补充计时覆盖 9 个所选模型回合：每步物理执行、相机采样与指标计算合计 p50 为 6.95–8.65 ms。首批回合未拆分该项耗时，原有完整决策时延均已保存。
模型完整回合吞吐为 0.82–0.95 仿真秒/墙钟秒，分母含仿真及模型初始化。

原始回合统计：[episode_results.csv](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/episode_results.csv)；[aggregate.json](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/aggregate.json)；[逐回合原始记录](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/test)。

## 专家基线与数据

本地与服务器起始提交均为 `9fdeae89613e0597603c605a0f5171723efedca0`，仓库起始干净。原专家入口未修改。原脚本三次复测均成功，最终门角约 89.98025°、尾端净距离约 0.55 m。新接口的专家轨迹和禁用专家函数的动作回放也通过，最终结果一致。原脚本无 seed CLI，三次复测是同一确定性设置的重复性验证，不是泛化测试。

原专家实际阶段步数：walk 260、initial_hold 150、grasp 150、grasp_hold 100、close_gripper 120、rotate_handle 300、push_door 1080、return_home 360、hold_home 330。最后阶段由 2850 步总上限决定。

资产为门 `99650089960001`，缩放 1.2，门位置 `(2.5,0)`；机器人标称 x=4.1 m、z=0.60 m、yaw=π，y 由原把手几何与 robot_y=-0.06 m 合成。初始门关闭，负铰链角为推门方向，signed angle=-原始角度。把手上限 45°、解锁阈值 40/45，自动开门扭矩始终为 0。

采集种子 1000–1019，20 次尝试全部保存且全部成功。按回合预分配训练 14、验证 4、留出测试 2；成功筛选发生在分配之后。共 57000 条 60 Hz 样本，训练 39900、验证 11400、留出 5700。窗口不跨 episode，统计量仅由训练回合计算。每回合独立初始化，初态均匀随机 x±0.05 m、y±0.03 m、yaw±2°；门资产、位置、缩放和物理参数固定。

数据包含动作前机器人状态、同一帧双相机 mask/depth、实际动作、帧号/时间，以及动作后的指标。米制深度另存，但早期 depth_m 已经过非有限值替换和 2 米裁剪，不能作为未经处理的原始传感器深度。后续记录器增加了显式 next_state 与原始 Gym 有符号深度 gym_depth_m；早期 episode 的相邻动作前状态给出非末尾 transition 的后状态，最后一步仅保留终止指标，训练不使用 next_state。额外 `recording-audit/` 回合验证补充字段，仅用于记录器审计，未加入训练或测试成绩。完整训练数据留在服务器 `data/`，[manifest](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/manifest.json) 指明路径、种子、拆分与成功状态。

这些结果只覆盖同一门和小范围初态变化，不证明新门、不同物理参数或真实机器人泛化。没有补齐缺失门资产，也没有使用低层行走模型。

一个已核验的物理诊断例子：未训练模型 seed 2000 在第 300 步出现约 90.31° 的瞬时门角，但 unlocked=False，随后原锁机制恢复关闭，最终失败。因此最大门角或 ever_open80 不能单独证明推门成功；原锁的逐步状态钳制与运动学根移动也是本环境的局限。

## 控制接口与无专家辅助边界

动作 schema 为 `float_push_joint_v1`：机体系底座 vx、vy、yaw rate，加 joint1–joint6 与夹爪位置目标，共 10 维；它与原 DP 的 10 维末端位姿动作不兼容。速度单位 m/s、rad/s，关节单位 rad。底座速度上限分别 0.6、0.3、0.4；位置目标按 URDF 限位裁剪。专家内部保留原 IK 的 range_scale=0.75 与 IK 最大增量 0.06 rad；模型直接输出关节伺服目标，不调用专家 IK 或回收阶段。

38 维状态包含初始坐标系下底座平面位姿、执行器速度、七关节位置/速度、底座系末端位姿及上一执行动作。四元数为 xyzw，归一化并保持符号连续。视觉使用腕部与前置相机各 54×96 的把手 mask、masked depth，共四路输入；模型输入与录制使用同一转换。把手分割来自 Isaac Gym 的语义 segmentation，这是明确的仿真感知辅助，尚未训练现实视觉分割器。

策略输入不包含专家阶段、时间进度、门/把手 GT 位姿或角度。GT 只用于原任务锁机制和独立统计。策略进程把专家轨迹函数、IK 更新和 IK 控制器构建替换为报错；相同执行器已通过独立动作回放。门的唯一持续状态写入是原有未解锁关闭锁机制；没有直接设置开门角度或自动开门扭矩。

底座仍是原仓库的运动学浮动执行器，固定高度，直接移动根刚体。模型负责其速度命令，但这不是具有接触平衡和四足步态约束的行走控制。底座数值跨越门平面也不单独算成功，必须同时满足解锁与最终开门阈值。

## 训练配置与证据

复用原 `high-level/dp/models/door_diffusion_policy.py`，未重建 PPO/DAgger。训练直接读取 NPZ，不引入 LeRobot。观测/预测/执行窗口为 4/16/4；DDPM 训练扩散步 100、推理步 25、squaredcos beta、epsilon prediction。状态/动作按训练均值和标准差归一化，标准差下限 0.01；禁用归一化动作的 [-1,1] 扩散裁剪。相较原管线，这些适配明确保存在源码与 checkpoint 中。

batch=16，AdamW，lr=1e-4，默认 betas=(0.9,0.999)、weight_decay=0.01，梯度裁剪 1.0，训练种子 42。验证使用固定窗口与固定扩散噪声。每步记录原始 loss/lr/elapsed，每 250 步离线验证，每 500 步独立保存 checkpoint；另保留 0/100 步。checkpoint 包含模型、优化器、数据 manifest、归一化、schema 和完整 RNG 状态，已实际从 100→500→2000→5000 步恢复。

实际训练到 5000 步；首条训练噪声 MSE 1.249941，最后一条 0.018670，末次验证 MSE 0.020601。单点 loss 带有扩散噪声采样波动；图中保留未平滑曲线。

缓存相机编码和扩散时间嵌入，并将调度索引留在 CPU，减少同步开销。与原逐次完整网络前向相比，同输入同随机种子的动作最大绝对差为 0；见 `logs/policy-equivalence.log`。

![](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/plots/training_validation.png)
![](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/plots/closed_loop_success.png)
![](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/plots/test_comparison.png)
![](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/plots/rollout_metrics.png)

各图均有同名 PDF。逐步原始数据见 `training/train.jsonl`、`training/train.csv` 和各 rollout 的 `metrics.csv`、`trajectory.npz`；动作图与限幅图保存在 `plots/`。离线动作预测误差见 `offline_action_diagnostics.json`，该诊断使用录制观测，不能冒充闭环评估。

离线预测审计使用两条验证轨迹的 20 个固定时刻，未使用独立测试观测。各通道误差保留真实单位，完整数值在 `offline_action_diagnostics.json`。

| checkpoint | base vx RMSE (m/s) | base vy RMSE (m/s) | yaw-rate RMSE (rad/s) | 6 个机械臂关节 RMSE 范围 (rad) | 夹爪 RMSE (rad) |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.259 | 0.286 | 0.454 | 15.299–28.877 | 6.979 |
| 100 | 0.240 | 0.026 | 0.026 | 2.083–4.277 | 1.475 |
| 500 | 0.085 | 0.011 | 0.009 | 0.614–1.884 | 0.421 |
| 2000 | 0.016 | 0.001 | 0.000 | 0.149–0.185 | 0.059 |
| 5000 | 0.007 | 0.000 | 0.000 | 0.031–0.074 | 0.022 |

同步计时的离线审计中，所选模型预处理加视觉/状态编码均值 4.86 ms，25 步去噪加动作还原均值 26.05 ms（20 次，含首调用；逐次原始值已保存）。这项计时不替代上面的真实闭环决策时延。

## 指标口径与验证

| 指标 | 实际定义及单位 |
| --- | --- |
| 原 open_80deg 兼容字段 | 实际 abs(angle)≥20°，不是 80° |
| 原 passed_door 兼容字段 | 上述 20° 条件加 door_x-base_x>0.7 m |
| 本轮主成功 | 已解锁、最终 signed angle≥80°、尾端净距离≥0.52 m；0.52=0.55−0.03 原验证容差 |
| reach / grasp | 分别 0.08/0.12 m 阈值连续 10 步；grasp 另需命令闭合>0.65、实际闭合>0.35。事件按 episode 锁存，独立于原 RL curriculum |
| signed_open_ratio | 复用浮动专家的 abs(q-closed)/关节范围并裁剪 [0,1]；这个兼容名字的实现并非带符号，方向另由 signed_angle_deg 判定 |
| handle_open_ratio | (q-lower)/(upper-lower)，解锁阈值 40/45；原始角度另以度记录 |
| EE / FK 误差 | 实际 EE 到把手距离；关节目标经 FK 得到的目标距离和跟踪误差，单位 m。FK 指标为新增，不冒充原 Cartesian ee_goal 字段 |
| phase | 分别报告 reach、grasp、解锁、开门、最终成功；诊断结果不控制动作 |
| action | 原始/执行动作、逐通道限幅、L2 与差分；混合单位 L2 只作诊断，逐通道轨迹优先 |
| loss / lr / elapsed | 原 DP 噪声预测 MSE、AdamW 学习率、累计训练循环秒数；不等于奖励或任务成功 |

`ever_open80`、最终 open80、原 20° 兼容开门/过门字段都保留在逐回合 CSV/JSON 和逐步 metrics.csv；曾达 80° 不参与最终成功判定。尾端距离采用门法线方向投影，尚不检查门洞横向容纳或四足接触，因此主成功应理解为本浮动仿真的数值任务判据。

逐项源码来源、公式和适配差异见 [metric_sources.json](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/tools/metric_sources.json)，源码版本及资产哈希见 `provenance/`。检验结果：**21/21 项通过**，包括动作/相机对齐、独立回放、同种子 reset、坐标变换、禁止专家调用、成功阈值与方向、归一化训练集隔离和视频解码。

## 视频与本地产物

- [adapter-nominal/rollout_labeled_h264.mp4](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/adapter-nominal/rollout_labeled_h264.mp4)：1425 帧，30.0 fps，47.5 秒，完整解码 通过。
- [test/expert_None_seed_3000/rollout_labeled_h264.mp4](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/test/expert_None_seed_3000/rollout_labeled_h264.mp4)：1425 帧，30.0 fps，47.5 秒，完整解码 通过。
- [test/policy_0_seed_3000/rollout_labeled_h264.mp4](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/test/policy_0_seed_3000/rollout_labeled_h264.mp4)：1425 帧，30.0 fps，47.5 秒，完整解码 通过。
- [test/policy_5000_seed_3000/rollout_labeled_h264.mp4](/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results/test/policy_5000_seed_3000/rollout_labeled_h264.mp4)：1425 帧，30.0 fps，47.5 秒，完整解码 通过。

交付版视频在真实 rollout 帧上补充了 run_id 标签；原始未转码文件同时保留。目录、源码哈希与视频 SHA256 保存关联。视频每两仿真步采样一帧，30 fps、1 倍仿真速度，编码 H.264/yuv420p。画面标注模型训练步、种子、仿真时间和回合标识；配置、SHA256 和首中尾抽帧共同关联到对应 rollout。未训练模型视频作为代表性失败；所选模型的失败视频同样保留，不替换为专家或回放视频。

| 产物 | 服务器相对实验根目录 | 本地 |
| --- | --- | --- |
| 完整数据 | data/，manifest.json | manifest 已同步，大体积数据保留服务器 |
| 分步 checkpoint | training/checkpoints/ | 已同步未训练、100 步、所选模型与最后模型；其余保留服务器 |
| 原始训练与评估 | training/、validation/、test/ | 日志、CSV/JSONL、逐回合轨迹已同步 |
| 图表视频 | plots/、各 rollout 目录 | PNG/PDF、H.264、视频核验记录已同步 |
| 复现资料 | provenance/、tools/、budget.jsonl | 环境清单、源码包、补丁和预算记录已同步 |

服务器根目录：`/data/zyf/video2door/experiments/float-push/20261011-224552-smoke`。本地根目录：`/Users/yu/Data/Video2Door/reproduction/pro5000/float-push-training/results`。每个服务器产物的大小与 SHA256 见 `artifact_sha256.json`；同步文件另做本地校验。

## 环境、代码与复现命令

服务器 Ubuntu 22.04，RTX PRO 5000 72GB Blackwell；驱动 580.126.20 保持不变。环境 Python 3.8.20、PyTorch 2.3.1+sm120、私有 CUDA 12.8、NumPy 1.23.5、diffusers 0.24.0，Gym Preview 4。GPU PhysX、CPU tensor/API pipeline，CPU 数学库和 PhysX 各 2 线程，nice 10；单场景、单卡串行。仿真 dt=1/60 s、Z 轴向上、gravity=(0,0,-9.81)、PhysX solver_type=1、位置/速度迭代=6/1；其余 Gym 默认值由 Preview 4 版本确定。腕相机 link06 局部位置=(0.0955,0.22,-0.03175) m、旋转=(-1.57,0,-1.07) rad；前相机 trunk 局部位置=(0.425,0.04,0.12) m、旋转=(0,-30°,0)，均为 69° 水平 FOV、96×54。完整解析参数在每回合 config.json。精确 pip/conda 清单在 provenance。系统 CUDA、驱动、base 和其他环境均未改动。

新增源码在 `/Users/yu/Data/Video2Door/code/high-level/float_ik/learning/`：仿真与动作接口、URDF FK 诊断、共享 schema、录制/播放入口、原 DP 推理适配、NPZ 训练、检查与绘图。原专家、原 DP 模型、原低层控制器均未修改。新增文件尚未提交 Git；完整补丁和源码包在 provenance。

以下命令用于查看和复现单个结果；不要覆盖原实验输出目录。每次换 GPU 必须在激活前设置 VIDEO2DOOR_GPU，保持 CUDA/Vulkan 同卡映射。

```bash
ssh pro5000
export VIDEO2DOOR_GPU=0  # 先按当时三次资源采样选择
source /data/zyf/video2door/.repro/blackwell/activate.sh
run=/data/zyf/video2door/experiments/float-push/20261011-224552-smoke
# 独立模型回合，输出目录必须不存在
nice -n 10 timeout 150 python high-level/float_ik/learning/rollout.py \
  --mode policy --seed 3000 --video \
  --checkpoint "$run/training/checkpoints/step_005000.pt" \
  --output "$run/manual-new-rollout"
# 只读数据检查与图表重建
python high-level/float_ik/learning/checks.py --root "$run"
python high-level/float_ik/learning/analyze.py --root "$run"
```

真实执行命令逐条保存在 budget.jsonl。训练恢复入口为 `train.py --manifest ... --output ... --resume ... --steps ... --seconds ...`；正式训练需要先获得确认，并使用新 run_id、新预算账本和新场景种子，不能直接继承本次硬编码 2 小时账本无限追加。

## 计划差异与记录限制

首轮实际采集 20 次（上限 40），已满足至少 8 条训练和 2 条验证成功轨迹。原始深度与显式末帧后状态在早期录制中不足，后续仅修复记录器并用独立审计回合验证，没有覆盖历史数据或回填不存在的测量。未训练、100 步、5000 步 checkpoint 已同步，其他分步模型留在服务器。

执行耗时拆分是在最终测试途中增加的只读计时；不改变模型权重、观测输入、动作、物理或终止条件，逐回合源码哈希区分版本。完整预处理/编码和去噪分解另在固定录制验证观测上测量，并明确标为离线诊断。

## 预算与实际结论

| 阶段 | 实际占卡进程墙钟 | 调整后上限 |
| --- | --- | --- |
| 基线与接口 | 4.27 分钟 | 5 分钟 |
| 数据采集 | 12.47 分钟 | 25 分钟 |
| 训练与周期评估 | 21.03 分钟 | 60 分钟 |
| 最终评估与视频 | 24.28 分钟 | 30 分钟 |

累计 **1.034 GPU 小时**，低于 2 GPU 小时。用户已确认把未用基线预算的 15 分钟调拨给最终评估；因此阶段上限从 20/25/60/15 分钟调整为 5/25/60/30 分钟。按本任务进程墙钟累计，包含激活和启动等开销，未按 GPU 利用率折算；CPU 绘图、校验和同步不占 GPU。

**当前 DP 路线已经可录制、可训练、可恢复、可独立闭环运行，但本轮所选模型未形成可靠推门能力。** 不建议直接把相同配置延长至正式 6 小时并期待自然收敛。离线录制观测上的动作误差已明显下降，而闭环仍出现大幅限幅和轨迹偏离，这更支持优先检查首次偏离、误差累积与输入分布变化。限幅现象本身不证明归一化代码存在错误。下一步可在新的小规模验证场景中分别测试训练支持范围内的动作约束、观测/动作归一化稳健性与更长历史或显式记忆；不要同时重做所有算法。这些是待验证假设，不是本轮已证实的修复。

本轮未启动 PPO、DAgger 或其他算法，没有额外门资产下载，也没有启动正式训练。首轮结果交付后停止，等待用户确认下一阶段。
