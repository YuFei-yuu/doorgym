# Video2Door 浮动底座推门训练交接文档

> 2026-10-12 更新：下文保留的是训练前部署交接。首轮 DP 数据、5000 步训练和独立评估现已完成，专家测试 10/10、5000 步模型 0/10。请先阅读 [首轮结果、训练对照与下一轮提示词](FLOAT_PUSH_TRAINING_NEXT.md)，不要再按本文的“尚无训练接口/数据/MP4”判断现状。额外正式训练尚未启动。

更新日期：2026-10-11，记录已完成的部署与验证。**当前完成的是脚本专家推门，不是训练策略推门。** 

## 1. 当前结论与任务范围

- 服务器 `ps1` 上的项目和独立 conda 环境已经就绪，Blackwell GPU 的 Isaac Gym、PyTorch 和相机兼容问题已解决。
- 原项目浮动底座入口完整执行了 2850 步：靠近、抓取、压下把手解锁、推门、底座过门、机械臂回收，最终开门角度约 89.98°。
- 手腕和前置相机的 RGB、深度、把手分割采集通过，无需桌面或 OpenCV 窗口。
- 不需要下载缺失的 663 组门资产、完整的低层行走控制器。先使用已经验证的门 `99650089960001`，其余现有门可用于后续扩展。
- **尚未完成**：浮动底座训练接口、专家状态/动作数据集、正式模型训练、收敛曲线、模型闭环推理评估、最终效果视频。

## 2. 服务器、仓库与隔离要求

| 项目 | 位置或状态 |
| --- | --- |
| 本地工作区 | `/Users/yu/Data/Video2Door` |
| 本地项目源码 | `/Users/yu/Data/Video2Door/code` |
| 本地 SSH 配置 | `/Users/yu/.ssh/config`，已配置别名 `ps1` |
| SSH 连接 | `ssh ps1`，对应 `ubuntu@172.81.242.38` |
| 服务器项目 | `/data/zyf/video2door` |
| 仓库 origin | `https://github.com/YuFei-yuu/doorgym.git` |
| 部署提交 | `0acc2f050479fb54a0670ccb90bc72b98b52a54a` |
| conda 程序 | `/data/miniconda3/bin/conda` |
| 推荐运行环境 | `/data/zyf/video2door/.repro/envs/b1z1-blackwell` |
| 推荐激活脚本 | `/data/zyf/video2door/.repro/blackwell/activate.sh` |
| 本阶段辅助脚本 | `/data/zyf/video2door/.repro/float-push` |
| 本地辅助脚本与报告 | `/Users/yu/Data/Video2Door/reproduction/ps1` |

约束：

1. 所有新下载、克隆和新环境放在 `/data/zyf` 下；本项目产物优先放在 `/data/zyf/video2door` 内。
2. 使用 conda 隔离。不要修改 base、其他 conda 环境、系统 CUDA、NVIDIA 驱动、全局 Vulkan 配置或 shell 启动文件。
3. 不停止、不重启、不修改其他人的 GPU 任务。开始训练前重新查看 GPU 占用，控制并行环境数、CPU 线程和显存。
4. 原项目源码目前在服务器上保持干净，`.repro` 通过 `.git/info/exclude` 忽略。本地源码存在改动，不要覆盖。
5. 训练适配需要改代码时，保留原入口与可复现基线，记录补丁、提交和具体命令；不要删除旧数据、日志或模型。
6. Isaac Gym 默认 VHACD 几何缓存位于 `/home/ubuntu/.isaacgym/vhacd`；不要清空共享缓存。本项目其余主要缓存位于 `.repro`。

本次交接时重新通过 SSH 核对了部署提交、服务器 Git 干净状态和成功验证的 `summary.json` 路径。

## 3. 硬件与可用环境

服务器为 Ubuntu 22.04 / x86_64，256 CPU 线程，内存约 743 GiB。GPU 为 8 张 NVIDIA RTX PRO 5000 72GB Blackwell，计算能力 12.0 / `sm_120`。驱动为 580.126.20，系统 CUDA 为 13.0，均保持原样。

已验证环境：

| 组件 | 版本或说明 |
| --- | --- |
| Python | 3.8.20，兼容仓库 Isaac Gym Preview 4 的 Python 绑定 |
| PyTorch | `2.3.1+sm120`，基于 v2.3.1 源码私有编译，仅编译 `sm_120` |
| 私有 CUDA | conda 12.8 系列，nvcc 12.8.93 |
| cuDNN | 9.7.1.26，运行时版本 90701 |
| NumPy | 1.23.5 |
| torchvision | 0.18.1+cu121，项目使用的 CUDA resize 已验证 |
| torchaudio | 2.3.1+cpu，项目不使用音频 |
| diffusers / huggingface-hub | 0.24.0 / 0.20.3 |
| editable 包 | isaacgym、rsl_rl、skrl、legged_gym |
| 依赖一致性 | `pip check` 已通过 |

GPU 占用会变化。交接时 GPU 0–3 已有约 4.6–4.8 GiB 占用和其他计算任务，GPU 4–7 各约 66 GiB 占用。现有验证默认在物理 GPU 0 运行，单场景、CPU 线程 2、nice 10。正式训练需重新核对实时负载，逐步扩大规模。

旧的 `/data/zyf/video2door/.repro/envs/b1z1` 是历史失败诊断环境，其官方预编译 PyTorch 2.4.1+cu121 不支持这台机器的 `sm_120`。**不要使用旧的 `.repro/activate.sh`，不要直接用 pip 升级覆盖当前定制 PyTorch。**

### Blackwell 兼容方案的留存

参考过用户提供的 [CSDN 方案](https://blog.csdn.net/m0_56706433/article/details/148902144)，保留 Python 3.8，使用私有 CUDA 12.8 编译 PyTorch。构建时修复了两位数计算能力识别、C++ 扩展架构识别与 cuDNN 库定位。

- 源码：`/data/zyf/video2door/.repro/blackwell/src/pytorch`。
- PyTorch 提交：`63d5e9221bedd1546b7d364b5ce4171547db12a9`。
- 已构建 wheel：`/data/zyf/video2door/.repro/blackwell/src/pytorch/dist/torch-2.3.1+sm120-cp38-cp38-linux_x86_64.whl`。
- wheel SHA256：`02e230c25b3b4a51cd52305c78270a3144e55e7051648e6dd4fbbc28ad4ff9d7`。
- 精确版本：`.repro/blackwell/pip-freeze.txt`、`.repro/blackwell/conda-explicit.txt`。
- 构建/补丁/日志：`.repro/blackwell/build_torch.sh`、`pytorch-sm120.patch`、`logs/`。

该构建保留 CUDA、cuDNN、自动求导、TorchScript，关闭分布式、多 GPU 通信、Flash Attention 等部分功能。先按单 GPU 训练设计，不要默认当前环境支持 DDP。torchvision 独立 CUDA 扩展如 NMS 尚未验证；已有推门流程不使用这些算子。

## 4. 必须使用的激活与设备映射

在服务器 Bash 中：

```bash
ssh ps1
source /data/zyf/video2door/.repro/blackwell/activate.sh
```

激活脚本会进入项目根目录，激活正确 conda prefix，配置私有 CUDA/cuDNN、扩展缓存、临时目录、CPU 线程，并设置 `WANDB_MODE=disabled`。

换卡应在激活前设置，例如以下命令中的索引应替换为当时确认可用的 GPU：

```bash
export VIDEO2DOOR_GPU=0
source /data/zyf/video2door/.repro/blackwell/activate.sh
```

**不要在激活后只覆盖 `CUDA_VISIBLE_DEVICES`。** 这台多卡服务器的 CUDA 与 Vulkan 枚举顺序不同：之前 CUDA 0 是物理 GPU 0，原始 Vulkan 0 却是物理 GPU 7，导致相机 `cudaImportExternalMemory error 999`。直接指定匹配的原始 Vulkan 4，又会触发旧 Gym 的 `invalid device ordinal`。

当前激活脚本按 GPU UUID 自动匹配原始 Vulkan 索引，再启用项目私有设备筛选层，使选定 GPU 在 CUDA 和 Vulkan 中均为编号 0。因此相机使用 `--graphics_device_id 0`。相关文件：

- `/data/zyf/video2door/.repro/blackwell/probe_vulkan`。
- `/data/zyf/video2door/.repro/blackwell/vulkan-layer/`。
- `/data/zyf/video2door/.repro/blackwell/vulkan_device_filter.c`。
- `/data/zyf/video2door/.repro/blackwell/build_vulkan_filter.sh`。

私有层只改变设备枚举，不修改系统配置或驱动。需要重新编译辅助层时才运行 `bash /data/zyf/video2door/.repro/blackwell/build_vulkan_filter.sh`；正常使用无需重建。

快速只读检查：

```bash
ssh ps1
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
cd /data/zyf/video2door
git rev-parse HEAD
git status --short
source /data/zyf/video2door/.repro/blackwell/activate.sh
python -m pip check
python -c 'from isaacgym import gymapi; import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print((torch.ones(4, device="cuda") * 2).cpu())'
```

使用 Gym 的 Python 进程需先导入 Isaac Gym，再导入 torch。服务器没有安装 `rg`，远程检索可使用 `grep` 或在本地源码用 `rg`。

## 5. 已验证的推门启动命令与证据

推荐直接运行：

```bash
ssh ps1
bash /data/zyf/video2door/.repro/float-push/run.sh
```

这会自动激活环境，运行原始浮动底座演示的观测包装器，2850 步、单场景、240 秒超时上限，不训练。每次结果独立保存：

```text
/data/zyf/video2door/.repro/float-push/runs/<时间戳>-<进程号>/
  run.log
  summary.json
  <步数>-overview.png
  <步数>-wrist.png / <步数>-front.png
  <步数>-wrist-handle.png / <步数>-front-handle.png
  <步数>-wrist-depth-seg.npz / <步数>-front-depth-seg.npz
```

无需相机时：

```bash
bash /data/zyf/video2door/.repro/float-push/run.sh --no_show_camera_images
```

仅运行原项目、不使用观测包装器的已验证命令：

```bash
source /data/zyf/video2door/.repro/blackwell/activate.sh
python -u high-level/float_ik/isaacgym_float_ik_b1z1_basearn_push_door.py \
  --asset_root /data/zyf/video2door/low-level/resources/robots/b1z1 \
  --door_name 99650089960001 \
  --headless --nographics --no_show_camera_images \
  --steps 2850 --num_threads 2 --log_interval 60
```

当前成功记录：`/data/zyf/video2door/.repro/float-push/runs/20261010-235824-420064/`。

| 检查 | 实测结果 |
| --- | --- |
| 流程步数 / 用时 | 2850 步，带相机约 17.50 秒；无相机约 15.53 秒 |
| 初始门角度 | 0° |
| 最大把手角度 | 约 45°，解锁成功 |
| 最大 / 最终推门角度 | 约 90° / 89.98025° |
| 最终底座 xyz | `[1.30000, -0.02462, 0.60000]` 米 |
| 底座尾端过门距离 | 约 0.55000 米，按脚本尾端偏移 0.65 米计算 |
| 双相机 | 各 24 帧，深度/分割 54×96 |
| 有效把手深度像素最大值 | 手腕 464，前置 106 |
| 自动开门扭矩 | 0 |
| 结果 | `summary.json` 中 `passed=true`，进程退出码 0 |

本地证据：`/Users/yu/Data/Video2Door/reproduction/ps1/float-push/evidence/`，含 `summary.json`、`run.log` 和 0000/1560/2160/2760 步概览截图。**这些 NPZ 是相机快照，不是完整的专家状态/动作训练数据集；当前尚无 MP4 视频。**

## 6. 浮动底座专家入口、资产和参数

实际专家入口：

`/data/zyf/video2door/high-level/float_ik/isaacgym_float_ik_b1z1_basearn_push_door.py`

共享加载器与 IK：

`/data/zyf/video2door/high-level/float_ik/isaacgym_visualize_b1z1_basearn.py`

辅助文件：

- `/data/zyf/video2door/.repro/float-push/run.sh`：有界验证启动器，**不是训练启动器**。
- `/data/zyf/video2door/.repro/float-push/verify_push.py`：包装观测、结果检查、无窗口相机保存，不替换原项目的轨迹、IK 或物理函数。
- `/data/zyf/video2door/.repro/float-push/REPORT.txt`：本阶段报告。

机器人资产路径：`/data/zyf/video2door/low-level/resources/robots/b1z1`，URDF 为 `urdf/b1z1_basearn.urdf`。共享加载器默认路径仍指向原作者 `/home/sivan/...`，因此必须显式传入 `--asset_root`。临时拆分的基座/机械臂资产自动写入 `.repro/tmp` 并清理，不要复制作者目录结构来掩盖路径问题。

门配置：`/data/zyf/video2door/high-level/data/cfg/b1z1_opendoor.yaml`。

门资产根目录：`/data/zyf/video2door/high-level/data/asset/door_set`。现有完整目录为 `99650089960001`、`99650089960006`、`99655039960001`、`99655039960006`；只对第一组验证了完整推门。配置共列出 667 组，缺少 663 组；当前范围无需补全，也不要直接启用全量加载。

成功基线的关键参数与机制：

- 门缩放 1.2；门位置 `(2.5, 0)`；机器人初始 $x=4.1$、$z=0.60$、$\mathrm{yaw}=\pi$，y 根据把手位置与 `robot_y=-0.06` 合成。
- 仿真 $dt=1/60\,\mathrm{s}$；GPU PhysX，但该演示为了 CPU API/IK 使用 CPU tensor pipeline。日志 `GPU Pipeline: disabled` 是设计行为，不是 GPU 仿真失败。
- 7 个机械臂/夹爪 DOF，IK 控制 joint1–joint6，末端 `ee_gripper_link`。
- 推门角度原始符号为负，`door_motion_sign=-1`；对外报告正的 signed push angle。
- 把手解锁比例 `40/45`；锁定期间脚本会保持门铰链关闭，这是原任务锁机制，需要保留并清楚记录。
- `door_auto_open_force=0`；不能在策略评估中打开辅助扭矩或直接设置开门角度来制造成功。
- 默认过门开启，尾端偏移 0.65 米，目标尾端净过门距离 0.55 米。
- IK 阻尼 0.08，最大关节增量 0.06；控制增益/接触参数应从解析后的实际配置保存，不要只抄函数声明。
- Gym 的 `gymutil.parse_arguments` 对部分 `store_true` 参数的 `default=True` 不保留，原脚本仅对部分显示参数做了修正。以已通过的 `run.sh` 命令和运行时参数为基线，修改参数后重新验证。

实际各阶段步数：walk 260、initial_hold 150、grasp 150、grasp_hold 100、close_gripper 120、rotate_handle 300、push_door 1080、return_home 360、hold_home 330。最后 330 来自总运行步数 2850，不要把参数中的 `hold_steps=300` 误当作已执行终止逻辑。

## 7. 下一阶段最关键的适配边界

### 7.1 不能直接沿用已有训练命令

本次源码核对确认：

1. `/data/zyf/video2door/high-level/dp/record_door_dp_dataset.py` 的 push 模式调用 `high-level/play_b1z1_push_with_door_asset_camera.py`，不是 `float_ik` 入口。
2. `/data/zyf/video2door/high-level/dp/play_door_dp_policy.py` 同样调用上述带低层行走策略的脚本。
3. `/data/zyf/video2door/high-level/door_rl/door_asset_rl_env.py` 创建 `low_env`、加载 `low_policy` 并每步执行该策略，因此目前也不是可直接使用的浮动底座训练环境。

需要先把已成功的浮动底座仿真适配为可 reset/step、可读状态/图像、可施加策略动作、可判定终止的接口，再接入专家录制与训练。不要以两轮低层测试模型替代行走预训练模型，绕回本阶段已排除的低层任务。

### 7.2 建议优先评估专家 → DP 的路线

用户关注专家轨迹、训练收敛、推理与视频；仓库已经有专家录制→数据转换→Diffusion Policy 的管线，适合作为优先复用候选。**这是下一阶段建议，不代表训练适配已经完成。** 如选择 PPO/DAgger 或其他方案，应依据源码说明为什么更合适，保持浮动底座约束，避免不必要地重做所有算法。

相关原始路径：

| 功能 | 服务器源码路径 |
| --- | --- |
| DP 使用文档 | `/data/zyf/video2door/high-level/dp/docs/door_dp_usage.md` |
| 原专家录制 | `/data/zyf/video2door/high-level/dp/record_door_dp_dataset.py` |
| 原数据/状态/动作/推理工具 | `/data/zyf/video2door/high-level/dp/door_dp_common.py` |
| LeRobot 转换 | `/data/zyf/video2door/high-level/dp/convert_door_raw_to_lerobot.py` |
| DP 模型 | `/data/zyf/video2door/high-level/dp/models/door_diffusion_policy.py` |
| DP 训练 | `/data/zyf/video2door/high-level/dp/train_door_dp.py` |
| 原 DP 播放包装器 | `/data/zyf/video2door/high-level/dp/play_door_dp_policy.py` |
| DP 训练依赖清单 | `/data/zyf/video2door/high-level/dp/requirements_dp.txt` |
| PPO/DAgger 指标参考 | `/data/zyf/video2door/high-level/door_rl/train_door_rl.py` |
| 奖励/成功条件参考 | `/data/zyf/video2door/high-level/door_rl/door_asset_rl_env.py` |
| RL 文档 | `/data/zyf/video2door/high-level/door_rl/docs/door_rl_usage.md` |

本地源码对应把 `/data/zyf/video2door` 替换为 `/Users/yu/Data/Video2Door/code`。

原 DP 使用 10D 动作 `[vx, yaw, ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, gripper]`；原 RL 使用 9D 增量动作，二者不可混用。浮动底座适配必须明确坐标系、速度/位置/增量语义、控制频率、四元数归一化和动作约束。原 DP 状态包含全身 DOF、低层动作、足接触；浮动底座状态不一定有这些字段，不要无说明地填充虚假数据。优先复用字段语义，必要时调整维度并保存 schema 与映射。

原数据字段包含 `state`、`action`、`wrist_handle_mask`、`wrist_masked_depth`、`front_handle_mask`、`front_masked_depth`、`subtask_index`、`task`、`fps` 与特征名。`subtask_index` 只用于调试/可视化，原文档明确不将其作为策略观测；不要在评估时用专家阶段标签泄漏答案。GT 把手/门状态若用于专家或评估统计，应与策略实际可见观测分开。

### 7.3 Python 环境分工

Isaac Gym 仿真、专家录制和策略闭环播放保留在已验证的 Python 3.8 环境。仓库 LeRobot 转换/训练要求 Python ≥3.10，**尚未创建这部分环境**。如采用该管线，建议另建例如 `/data/zyf/video2door/.repro/envs/door-dp-train` 的 conda prefix，固定依赖版本并保存清单，不要向 Gym 环境直接安装整个 `requirements_dp.txt`。

新训练环境必须选择真实支持 Blackwell 的 PyTorch 构建，并执行 CUDA 运算与模型训练检查。不要假定官方旧 cu121 wheel 能用，也不要为了新环境重复破坏已成功的旧环境。跨 Python 环境需实际验证 checkpoint 配置、state_dict、归一化统计及 diffusers 推理兼容性。

已有 `probe_project_model.py` 只证明默认 DP 网络在合成输入下 GPU 前向、反向、优化器和 scheduler 可运行；它不是专家数据训练结果，也不是新环境已就绪的证据。

## 8. 优先复用的指标与需要核实的语义

建议为每个指标保存“原仓库来源、公式/实现、单位、统计对象、阈值、与浮动底座适配的差异”，既保留仓库名字，也避免名字与实现不一致。

| 原有指标/字段 | 来源与建议用途 |
| --- | --- |
| `Success / open_80deg` | `door_rl` 的 `success_rate`；评估开门成功率，核实实际阈值，见下方说明 |
| `Success / passed_door` | 同上；保留过门成功作为更完整的任务指标，说明门角度与基座距离条件 |
| `Success / reach`、`Success / grasp`、`Success / phase` | 分阶段统计，帮助定位靠近、抓取、转把手、推门的失败点 |
| `Door metrics / signed_angle_deg`、`signed_open_ratio` | 门角度/开度轨迹、每回合最大/最终开度 |
| `Door metrics / handle_open_ratio` | 把手角度归一化与解锁情况 |
| `Door metrics / ee_to_handle_m`、`ee_goal_to_handle_m` | 实际末端与目标末端到把手的误差，分别统计 |
| `Door metrics / arm_base_to_handle_m`、`action_l2` | 基座接近程度、动作幅度；浮动底座字段含义要明确 |
| `Reward terms / reach`、`handle`、`open`、`pass_align`、`pass_progress` 等 | 如果采用 RL 或需要评估奖励分解，复用对应定义并记录适配；DP 不要虚构训练奖励 |
| `Phase / ...` | 阶段 ID、reach/grasp/phase success rate，用于专家质量和评估诊断 |
| `train/step`、`train/loss`、`train/lr`、`train/elapsed_sec` | `dp/train_door_dp.py` 原有 DP 指标；loss 是预测扩散噪声的 MSE，不是直接动作误差 |
| `Loss / Policy loss`、`Value loss`、`Entropy loss` | 采用 PPO 才记录；不要求 DP 伪造这些指标 |
| `Loss / DAgger loss`、`Online teacher-student MSE` | 采用 DAgger 才记录 |
| `dp_action_raw`、`applied_action`、`ee.pos_error`、`robot_command`、`base`、`gripper`、`door.dof` | `door_dp_common.make_door_dp_log_record` 的推理日志字段，适合对比网络输出与实际下发动作 |

**指标陷阱：** 当前 `door_rl` 虽把 key 命名为 `Success / open_80deg`，实际 `open_bonus` 使用可配置的 `open_success_angle_deg`，其 CLI 默认是 **20°**；`pass_bonus` 还使用 `pass_success_x > 0.7`。不能仅看名字就声称达到 80°。原 DP 录制文档则使用默认 80° 加“开始 pass 阶段”筛选成功演示。本阶段浮动底座验证检查开度 ≥80°，并按基座尾端净距离约 0.55 米检查过门；这些是不同口径，下一阶段应显式对齐或分别报告。

建议同时保存原指标与明确阈值的 episode-level 结果。成功率分母使用所有评估尝试，不能只统计保存的成功专家轨迹。训练日志中的跨环境瞬时均值与独立评估的回合成功率也应区分。

为了判断收敛，按固定评估集/随机种子比较多个 checkpoint，至少保留损失、闭环开门成功率、过门成功率、门角度轨迹和失败类型。验证 loss、动作误差、推理时延、显存/吞吐等原仓库未完整提供的指标可以补充，但需标为新增指标。不要用训练 loss 下降替代闭环成功，也不要预先承诺一定收敛。

## 9. 数据、评估、视频与实验留存建议

下一阶段应完成以下链路，并逐项保留证据：

1. **专家基线与轨迹设置**：保存完整解析后参数、实际阶段时长、dt、资产、控制接口、随机种子；说明是否随机化初始位姿和门参数。记录所有采样尝试数、成功/失败数，再筛选训练轨迹。
2. **训练数据**：逐步保存同步的状态、动作、图像、门/把手状态和诊断字段；正确记录绝对目标与实际施加动作。按 episode 拆分 train/val/test，禁止相邻帧随机拆分导致泄漏。单门单一确定性轨迹可以用于验证拟合，但不能据此声称泛化。
3. **训练**：先短训练与 checkpoint 加载验证，再有界地扩大数据和训练规模；保留每阶段配置、日志、损失、评估结果、可恢复 checkpoint。把训练策略、脚本专家和随机/未训练策略作为可区分的基线。
4. **闭环推理**：重置真实仿真，由训练模型控制约定的动作；记录任何残留脚本/阶段/GT 辅助。若底座仍按固定脚本移动，只能声称学习了机械臂/夹爪部分，要清楚说明，不能称为完整策略自主推门过门。
5. **图表**：输出 PNG/PDF 和原始 CSV/JSONL，包括损失曲线、评估成功率随训练进度变化、门/把手角度、EE 误差与动作轨迹；坐标单位、横轴步数/回合/墙钟时间、平滑方式和随机种子清楚标注。至少保存未平滑数据。
6. **视频**：录制专家与训练模型的实际 Isaac Gym rollout；概览视角为主，必要时并列手腕/前置视角和角度指标；输出可播放 MP4，明确模型、回合、成功/失败、fps、仿真时间与是否加速。保存代表性失败和全量统计，不只挑成功片段。
7. **交付**：输出数据清单、模型文件、评估明细、图表、视频、完整命令、环境清单与最终报告。未达到目标则报告实测性能、失败证据和后续建议，不将脚本成功冒充训练成功。

时间采样需要特别注意：原 DP 文档默认 fps=50，而浮动底座演示实际 $dt=1/60\,\mathrm{s}$；当前相机验证每 120 步只保存一帧，只适合截图证据。训练数据和最终视频需重新按实际控制/采样频率同步采集，不能把旧稀疏截图直接标成实时视频。

建议新实验按独立 run_id 保存，例如 `/data/zyf/video2door/experiments/float-push/<run_id>/` 下放 `config/`、`data/`、`checkpoints/`、`metrics/`、`plots/`、`videos/`、`logs/`。这是建议目录，当前尚未创建。本地最终报告与必要图表/视频可同步回 `/Users/yu/Data/Video2Door`。

当前 DP 训练源码还有两点应先核实：

- README 的 checkpoint 路径写为 `high-level/logs/door-dp/...`，但当前 `train_door_dp.py` 实现实际写入 `/data/zyf/video2door/high-level/dp/logs/door-dp/<run_name>/checkpoints/model_latest.pt`。以执行输出和实际文件为准。
- 默认反复覆盖 `model_latest.pt`，且没有完整独立验证/评估管线；下一阶段要保留按训练步数区分的 checkpoint 和 CSV/JSONL，才能回看收敛并选择模型。小数据 smoke 时 `drop_last=True` 还要求数据序列数至少达到 batch_size，避免空 DataLoader 导致无训练步进。

## 10. 历史测试与不要误用的文件

上一阶段兼容性验证通过：原生 sm_120 CUDA 内核、torch CUDA 运算、gymtorch 扩展、GPU PhysX 状态张量、反向传播、cuDNN 卷积、GPU 深度相机、DP 网络合成输入测试。

曾运行过低层入口两轮 debug 训练（约 6144 transitions），只为验证环境链路：

- `/data/zyf/video2door/low-level/logs/repro-smoke/blackwell_smoke/model_0.pt`。
- `/data/zyf/video2door/low-level/logs/repro-smoke/blackwell_smoke/model_2.pt`。

这两个模型不具备充分训练的行走能力，更不是浮动底座推门模型。不要用于证明推门收敛。

`/data/zyf/video2door/.repro/blackwell/verify.sh` 包含上述低层两轮训练并覆盖同名历史测试输出。**新的 agent 为了解当前环境无需重跑整个 verify.sh**；先用本交接的只读检查和 `float-push/run.sh`。

旧 `.repro/REPORT.txt` 和 `.repro/blackwell/REPORT.txt` 下文保留了全 README 资产缺失的历史报告。当前阶段的准确信息优先看本文、`.repro/float-push/REPORT.txt` 和成功 run 的 `summary.json`。


## 11. 交接后最短阅读顺序

1. 本文及 `/Users/yu/Data/Video2Door/reproduction/ps1/float-push/REPORT.txt`。
2. `/Users/yu/Data/Video2Door/reproduction/ps1/float-push/evidence/summary.json` 与 `run.log`。
3. 服务器 `.repro/blackwell/activate.sh` 和 `.repro/float-push/run.sh`。
4. 原始浮动底座推门脚本及共享 IK 加载器。
5. `high-level/dp/docs/door_dp_usage.md`、DP 状态/动作工具、训练模型与现有录制/播放包装器。
6. `high-level/door_rl/door_asset_rl_env.py` 中的实际指标与成功条件，实现浮动底座适配时对照引用。

读完这些内容即可开始下一阶段；不需要先补齐全 README 的行走控制、全部资产和其他任务。
