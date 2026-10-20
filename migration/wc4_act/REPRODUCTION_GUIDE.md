# wc4 Joint9 ACT：从作者仓库到可用开门策略的复现手册

更新：2026-10-20。适用对象：在本项目上执行复现的后续 agent。

**本轮模型训练已由用户确认完成，当前没有待执行的续训、补采或确认评测任务。** 本文总结已成功采用的技术方案，供以后明确要求“重新复现”时使用。阅读、维护本文本身不启动实验。

本路线为：

```text
固定作者源码及 wc4 资产
→ A2W 浮动底座 + 作者 wc4 skill 专家
→ 50 条成功 Joint9 NPZ
→ 双 depth LeRobot 数据
→ ACT baseline，训练量 30k 以上，可扩大至 40k～50k
→ 导出 Door checkpoint
→ Isaac Gym 中 64 回合闭环评测
```

已经取得的结果：50 条专家轨迹、27,500 帧训练数据；50k 模型在固定 64 回合上成功 48 次（75%）。同一固定集上的 36k 模型为 52/64。成功仅指最大绝对门铰链角达到 80°，不代表穿越、碰撞合理性和停稳已经完成验收。重新训练应复现流程、数据契约和有效策略行为，不能要求模型 SHA 或成功次数逐位相同。

## 1. 先确定使用哪份代码

### 1.1 固定版本

| 用途 | 仓库 / 提交 |
|---|---|
| 原作者仓库 | <https://github.com/Sivan666666/doorgym> |
| 作者分支，仅用于定位 | `train_lerobot_merged` |
| 本路线固定作者源 | `096938c44ecc1c7602304d42502f96658e46eb25` |
| 项目 fork | <https://github.com/YuFei-yuu/doorgym> |
| 本项目分支 | `main` |
| 实际训练和录像支持版本 | `5ca33bf00bf7f0bd4e0484db8cbeb48c84cb7f7f` |
| 复现支持 | 本项目当前 `main`，运行前记录实际提交 |

**不要直接使用作者分支的最新 HEAD。** 本项目没有把旧 DP 的 `forward_v3`、停车尾段、旧统计或旧模型迁入 ACT。

推荐取得作者固定源码，再使用项目 fork 中已验证的外围支持层。项目改动包括环境包装、命令生成、RGB 整体录像和验收工具；作者网络、skill、资产、物理和数据转换核心未改。

### 1.2 在现有工作站 / pro5000 上

| 内容 | 位置 |
|---|---|
| 本地新代码 | `/Users/yu/Data/Video2Door/code-wc4-act` |
| 本地交付 | `/Users/yu/Data/Video2Door/reproduction/wc4-act` |
| 服务器 | `ssh pro5000` |
| 服务器新代码 | `/data/zyf/video2door-wc4-act` |
| Gym / Blackwell 支持 | `/data/zyf/video2door/.repro/blackwell` |
| Gym 环境 | `/data/zyf/video2door/.repro/envs/b1z1-blackwell` |
| ACT 环境 | `/data/zyf/video2door-wc4-act/.repro/envs/door-act` |
| 新实验输出 | 新代码下 `.repro/runs/<唯一run-id>/` |

已有工作区先检查 `git status`、HEAD 和运行进程，不覆盖或重置已有文件。原始数据、模型、环境和视频不全部存放在 Git 中。

### 1.3 新工作区从仓库取得代码

以下为 Linux Bash 示例。`WC4_REPO` 必须是一个尚不存在的新目录；在现有服务器原路径已有代码时，跳过克隆。

```bash
export WC4_REPO=/data/zyf/video2door-wc4-act
test ! -e "$WC4_REPO"
git clone --filter=blob:none --no-checkout \
  https://github.com/Sivan666666/doorgym.git "$WC4_REPO"
cd "$WC4_REPO"
git remote rename origin upstream
git fetch upstream 096938c44ecc1c7602304d42502f96658e46eb25
git remote add origin https://github.com/YuFei-yuu/doorgym.git
git fetch origin main
git switch -c reproduce/wc4-act origin/main
git merge-base --is-ancestor 096938c44ecc1c7602304d42502f96658e46eb25 HEAD
git diff --diff-filter=AM --exit-code 096938c44ecc1c7602304d42502f96658e46eb25 HEAD -- \
  high-level low-level third_party
```

完整检出最不容易遗漏依赖。需要稀疏检出时，至少保留：

- `high-level/dp/`、`high-level/float_ik/`；
- `high-level/lerobot/`，包含 `src/`、`setup.py`、`pyproject.toml`、`README.md`、`LICENSE`、`MANIFEST.in`；
- `high-level/data/cfg/` 和 `high-level/data/asset/{a2wz1,z1,door_set/wc4}/`；
- 新建 Gym 环境时需要的 `third_party/`、`low-level/`；
- 项目的 `migration/wc4_act/`、根 `AGENTS.md`。

当前main已包含手册和相邻的 [reproduction_support](reproduction_support/README.md) 附件。上面的diff只检查保留源码的新增/修改；主线有意删除了其他实验，不要求与作者完整仓库文件列表相同。

仅用作者源也可以复现算法，但必须自行实现本文列明的环境、路径、录像与证据编排。无需重做旧低层训练，也无需先准备旧实验缺失的玻璃门资产：此路线用作者 A2W 浮动实现及 wc4 资产。

## 2. 环境：Gym 与 ACT 必须分开

所有仿真、训练和数值检查在 pro5000 的项目专用环境执行。本地 macOS 用于阅读、编排、文档和交付检查。

### 2.1 已验证组合

| 项目 | Gym 进程 | ACT / 转换 / 导出进程 |
|---|---|---|
| Python | 3.8.20 | 3.10 |
| Torch | 本项目 `2.3.1+sm120` | `2.7.1+cu128` |
| torchvision | 0.18.1 | `0.22.1+cu128` |
| NumPy | 1.23.5 | 2.2.6 |
| LeRobot | 不安装到 Gym | 仓库内定制实现，包版本0.4.4 |
| 视频 | 原专家相机、整体录像 | PyAV15.1.0、TorchCodec0.4.0、FFmpeg6.1.1 |
| 主要作用 | Isaac Gym、物理与策略执行 | 双深度数据、ACT训练、模型推理 worker |

服务器为 RTX PRO 5000 72GB Blackwell，计算能力12.0。普通 Python3.8 的旧 Torch wheel 即使报告 `cuda.is_available=True`，也可能不能执行这张卡的 CUDA kernel。本项目使用私有 CUDA12.8 工具链编译 Gym 所需 Torch；ACT使用官方cu128轮子。

### 2.2 优先复用现有环境

在新的 Bash 子进程中，按阶段激活：

```bash
cd /data/zyf/video2door-wc4-act
# ACT、转换和导出；不需要 GPU 的检查可不设置 VIDEO2DOOR_GPU。
source migration/wc4_act/activate.sh act
python -m pip check
python -c 'import lerobot; print(lerobot.__file__)'
```

LeRobot 路径必须指向本工作区 `high-level/lerobot/src/lerobot`。不能仅看“安装了lerobot”或版本号。

Gym 激活前先选择并锁定 GPU，见第4节；随后：

```bash
source /data/zyf/video2door-wc4-act/migration/wc4_act/activate.sh gym
```

包装器完成 CUDA/Vulkan UUID 对齐，然后返回新代码目录；设置 `DOOR_DP_LEROBOT_PYTHON` 指向 ACT Python。作者 `door_policy_backend.py` / `door_policy_worker.py` 已提供子进程桥接，不要自建 RPC 或将 Python3.10 包注入 Gym Python3.8。

### 2.3 从零创建 ACT 环境

环境附件保留了成功采集时的 Conda 显式包清单及 pip 版本。其 ACT Conda 清单已经包含 FFmpeg6.1.1。不要再额外求解一套“最新版”环境。

```bash
export WC4_REPO=/data/zyf/video2door-wc4-act
export WC4_ACT="$WC4_REPO/.repro/envs/door-act"
export WC4_SUPPORT="$WC4_REPO/migration/wc4_act/reproduction_support"
test ! -e "$WC4_ACT"
/data/miniconda3/bin/conda create -y --prefix "$WC4_ACT" \
  --file "$WC4_SUPPORT/act/act-conda-explicit.txt"
source /data/miniconda3/etc/profile.d/conda.sh
conda activate "$WC4_ACT"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
python -m pip install \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$WC4_SUPPORT/act/requirements-pinned.txt"
python -m pip install --no-deps -e "$WC4_REPO/high-level/lerobot"
python -m pip check
```

`requirements-pinned.txt`删除了原始freeze中自动生成的editable Git URL，改为最后一行显式安装当前本地源码。原freeze曾把本项目提交写成作者仓库URL；直接照抄该URL可能找不到提交。

在已有ACT环境补缺库时，使用附件 `act/ffmpeg-additions-explicit.txt` 作为历史42项新增包记录，先审查差异。不要把 Conda revision 回滚直接作用于已有混合pip环境；本轮曾发现它会覆盖pip管理的packaging/setuptools。新建环境按完整显式清单构建即可。

### 2.4 从零准备 Gym / Blackwell 支持

这部分只用于没有既有Gym环境的新机器。**现有 pro5000 无需重建。**

附件 `blackwell/`是实际构建支持文件的副本；包含补丁、构建脚本、Vulkan设备筛选层源码和依赖清单，不包含环境二进制、CUDA下载包或Torch源码。为兼容现有包装器，下面保留 `/data/zyf/video2door/.repro` 支持路径。新机器只需建立这棵支持目录，不需要重做旧DP实验。

先确保Linux主机有兼容Blackwell的NVIDIA驱动、`gcc/g++`、`git`、Vulkan开发头文件/链接库及NVIDIA Vulkan ICD。实际支持脚本使用 `/data/miniconda3`、`/usr/bin/gcc`、`/usr/bin/g++` 和 `/etc/vulkan/icd.d/nvidia_icd.json`。其他路径需一致调整**环境支持副本**，不要调整作者算法。

```bash
export WC4_REPO=/data/zyf/video2door-wc4-act
export WC4_SUPPORT="$WC4_REPO/migration/wc4_act/reproduction_support"
export WC4_GYM=/data/zyf/video2door/.repro/envs/b1z1-blackwell
export WC4_BW=/data/zyf/video2door/.repro/blackwell
test ! -e "$WC4_GYM"
test ! -e "$WC4_BW"
mkdir -p "$WC4_BW"
cp -a "$WC4_SUPPORT/blackwell/." "$WC4_BW/"
/data/miniconda3/bin/conda create -y --prefix "$WC4_GYM" \
  --file "$WC4_BW/conda-explicit.txt"
source /data/miniconda3/etc/profile.d/conda.sh
conda activate "$WC4_GYM"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
# Torch 尚未构建；按完整冻结列表安装其余依赖，不让依赖求解器偷装普通Torch。
python -m pip install --no-deps \
  -r "$WC4_BW/requirements-build-and-runtime.txt"
mkdir -p "$WC4_BW/src"
git clone --recursive --branch v2.3.1 \
  https://github.com/pytorch/pytorch.git "$WC4_BW/src/pytorch"
test "$(git -C "$WC4_BW/src/pytorch" rev-parse HEAD)" = \
  63d5e9221bedd1546b7d364b5ce4171547db12a9
python "$WC4_BW/prepare_toolchain.py"
bash "$WC4_BW/build_vulkan_filter.sh"
# 后续build_torch会调用activate，先按第4节选择并锁定GPU。
export VIDEO2DOOR_GPU=1
bash "$WC4_BW/build_torch.sh"
python -m pip install --no-deps "$WC4_BW"/src/pytorch/dist/torch-2.3.1+sm120-*.whl
python -m pip install --no-deps torchvision==0.18.1
python -m pip install --no-deps \
  'https://download.pytorch.org/whl/cpu/torchaudio-2.3.1%2Bcpu-cp38-cp38-linux_x86_64.whl'
python -m pip install --no-deps \
  -e "$WC4_REPO/third_party/isaacgym/python" \
  -e "$WC4_REPO/third_party/rsl_rl" \
  -e "$WC4_REPO/third_party/skrl" \
  -e "$WC4_REPO/low-level"
python -m pip check
```

`build_torch.sh`会应用 `patch_pytorch.py`，仅编译sm120，使用CUDA12.8/cuDNN9.7，关闭本项目不用的分布式和部分加速后端。不要同时再手工重复应用 `.patch`。附件中保存的既有wheel哈希仅用于核对该历史二进制；重新构建不保证产生相同字节。

私有Vulkan层解决多GPU机器上CUDA与Vulkan枚举不一致的问题。激活后仿真使用逻辑 `cuda:0` 和 `graphics_device_id=0`，不能把物理GPU编号直接填成graphics ID。导入Gym相关代码时先导入 `isaacgym`，再导入 `torch`。

旧支持目录中的 `verify.sh` 包含历史低层训练检查，不作为本手册默认步骤。不要为了复现ACT顺带启动旧低层训练。

### 2.5 环境门槛

先完成这些检查，再启动数据采集：

1. `check_static.py`：固定源、wc4/A2W/Z1资产及网格引用完整。
2. ACT环境的 `check_act.py`：实际定制LeRobot导入及转换/训练/导出参数解析。
3. ACT环境的 `check_video.py`：默认TorchCodec解码H.264；应输出 `[2,3,480,640]`，不能仅以PyAV编码成功或`pip check`替代。
4. Gym环境的 `check_gym_bridge.py`：无仿真解析及真实worker协议。该检查不加载模型，不能声称验证了模型推理。
5. 已选GPU上的最小Torch CUDA运算，以及首批真实采集的GPU相机、资产和数值检查。首批有效结果计入50条，不另跑大量“烟雾实验”。

前四项在各自环境下运行，证据写新目录；CPU检查隐藏GPU：

```bash
CUDA_VISIBLE_DEVICES='' python -B migration/wc4_act/check_static.py
CUDA_VISIBLE_DEVICES='' python -B migration/wc4_act/check_act.py
CUDA_VISIBLE_DEVICES='' python -B migration/wc4_act/check_video.py
# 切换到Gym环境后：
CUDA_VISIBLE_DEVICES='' timeout 90 python -B migration/wc4_act/check_gym_bridge.py
```

主机迁移后的环境构建命令是根据保留的成功构建脚本整理的恢复步骤；本次写文档没有在空白机器重新构建，也没有重跑训练。上述门槛是新机器的实际准入依据。

## 3. 冻结实验合同，避免混用不同 wc4 路线

| 环节 | 本路线固定设置 |
|---|---|
| 机器人 / 执行 | A2W+Z1浮动底座；`ikpush`；`gym_jacobian` |
| 门 / skill | `--door_name wc4`；原样 `wc4_smoke_skill.json` |
| 门物理 | 作者wc4配置、URDF及controller multipliers |
| 专家 | 原skill的接近、抓取、转把手、推门、穿越、收臂阶段 |
| 原始数据 | 50条成功NPZ，显式 `--state_action_mode joint_state9` |
| 并行 / 时间 | 最多16环境；1100仿真步；仿真dt约0.02秒 |
| 数据 / 相机 | 25Hz；每回合550帧、22秒 |
| 图像 | 前置与腕部masked depth，640×480；0.2～1.5m；无独立mask输入 |
| 相机扰动 | 位置0.02m、旋转5°；深度噪声/模糊关闭 |
| 相机内参 | `camera_intrinsics_mode=legacy`，不是real-K |
| 初态扰动 | 保留源分布，包含俯仰0～5°；采集与评测一致 |
| 几何检查 | 源skill建议的 `--no_collision_geom_check`；不改PhysX |
| 转换 | 直接Joint9、25fps、H.264、作者sanitizer |
| 训练 | seed1000、batch16、lr1e-5、ResNet18默认ImageNet预训练 |
| ACT | chunk100、训练n_action_steps50；全部50条用于训练 |
| 训练量 | 30k以上，按明确预算扩大到40k～50k；本文示例50k |
| 关闭项 | Plücker、Interaction、camera gating、handle aux、点云、end signal、keyframe加权/采样 |
| 闭环 | 64回合，16环境/批；25Hz；chunk100、实际执行25步 |
| 主成功判据 | `max(abs(door_hinge_angle)) × 180/π ≥ 80°` |

`legacy`这里只是原相机投影的枚举名称；专家始终使用显式wc4 skill。不要混入legacy专家、pull、real-K、Interaction、TRAC-IK转换或EE10数据。

Joint9不是任意9维向量：

- 状态：`[上一条底座vx指令, 上一条底座yaw-rate指令, 实测joint1..joint6, 实测jointGripper]`。
- 动作：`[当前vx指令, 当前yaw-rate指令, joint1..joint6位置目标, jointGripper位置目标]`。
- 单位：vx为m/s，yaw为rad/s，关节位置为rad。动作字段`yaw`不是绝对航向。
- 采集入口文件名虽然含`state10`，默认也确实是EE10，必须显式指定Joint9。
- 转换选`joint_state9`，不能用从EE数据派生关节动作的`tracik_joint_state9`。

## 4. 新 run、GPU 和命令执行规则

示例使用新run名；不要使用本文结果表里的历史run ID。下面变量需在每个新Bash会话重新设置，或保存为该新run的配置文件。

```bash
export WC4_REPO=/data/zyf/video2door-wc4-act
export WC4_RUN_ID=wc4-reproduce-20261020-r1
export WC4_RUN="$WC4_REPO/.repro/runs/$WC4_RUN_ID"
export WC4_TRAIN_STEPS=50000
export WC4_DATA_ID="local/wc4_joint9_skill_$WC4_RUN_ID"
export WC4_DATA="$WC4_RUN/lerobot/$WC4_DATA_ID"
cd "$WC4_REPO"
test ! -e "$WC4_RUN"
mkdir -p "$WC4_RUN/commands"
git rev-parse HEAD > "$WC4_RUN/source-head.txt"
git diff > "$WC4_RUN/source-worktree.patch"
```

**GPU编号不是固定配置。** 每次启动GPU阶段前检查实时显存、利用率和compute-app PID，选择空闲卡。与本项目其他任务共用 `/data/zyf/video2door/.repro/locks/<GPU-UUID>.lock`，不能另建一套互不相识的锁。下面是锁定示意，执行者还需复查资源快照：

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv
export VIDEO2DOOR_GPU=1  # 示例；必须根据上述实时结果选择
WC4_GPU_UUID=$(nvidia-smi -i "$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader)
mkdir -p /data/zyf/video2door/.repro/locks
exec 9>"/data/zyf/video2door/.repro/locks/$WC4_GPU_UUID.lock"
flock -n 9
```

持有锁的父Bash必须等待阶段子进程退出再释放FD9；退出这个Bash或`flock -u 9`释放锁。历史编排还检查了连续三次显存<1024MiB、利用率<10%、无compute-app占用，可参考`reference_tools/common.py` 的 `GPULease`。`nice10`和两线程限制不等于GPU租约。

`commands.py` **只生成命令，不执行**。它的`convert` shell前置校验是准备阶段版本，已知不适配缺省相机元数据；不要盲目将输出管给`bash`。使用下列方式保存真实argv，完成各阶段门槛后执行argv：

```bash
python - <<'PY'
import json, os, sys
from pathlib import Path
repo = Path(os.environ["WC4_REPO"])
run = Path(os.environ["WC4_RUN"])
sys.path.insert(0, str(repo / "migration/wc4_act"))
from commands import build
run_id = os.environ["WC4_RUN_ID"]
gpu = int(os.environ["VIDEO2DOOR_GPU"])
specs = [
    build("record", run_id, gpu=gpu, seed=62000, max_rollouts=100),
    build("convert", run_id),
    build("train", run_id, gpu=gpu, seed=1000,
          train_steps=int(os.environ["WC4_TRAIN_STEPS"])),
    build("export", run_id),
    build("evaluate", run_id, gpu=gpu, seed=615455575, trials=64),
]
for spec in specs:
    path = run / "commands" / (spec["stage"] + ".json")
    with path.open("x") as f:
        json.dump(spec, f, indent=2)
print("Generated five command specifications; nothing executed.")
PY
```

每个JSON的`argv`是实际参数，`cwd`是代码根目录。`shell`仅作为来源参考；原始数据门槛按下一节执行。所有阶段都应保存stdout/stderr、退出码、时间、环境和GPU绑定。不要为补日志重新运行成功阶段。

下面是通用执行示例。先激活所需环境、检查输出目录不存在、完成上游门槛，再用它执行**一个**stage。这不是自动串行跑全流程的脚本：

```bash
export WC4_STAGE=record  # 每次明确改为 record / convert / train / export / evaluate
python - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
run = Path(os.environ["WC4_RUN"])
stage = os.environ["WC4_STAGE"]
spec = json.loads((run / "commands" / (stage + ".json")).read_text())
targets = {"record": run/"raw", "convert": Path(spec["dataset"]),
           "train": run/"train", "export": run/"door-checkpoint",
           "evaluate": run/"evaluation"}
if targets[stage].exists():
    raise FileExistsError(f"Refusing implicit overwrite/resume: {targets[stage]}")
cmd = [sys.executable, *spec["argv"][1:]]
with (run / (stage + ".log")).open("x") as log:
    completed = subprocess.run(["nice", "-n", "10", *cmd],
                               cwd=spec["cwd"], stdout=log, stderr=subprocess.STDOUT)
(run / (stage + "-exit.json")).write_text(json.dumps(
    {"argv": cmd, "returncode": completed.returncode}, indent=2) + "\n")
completed.check_returncode()
PY
```

受控运行的有限值守卫、W&B监控和恢复前核验可参考 [历史工程实现](reference_tools/guarded_train.py)，复制到**新run**后按本次版本、路径和预算适配。它们检查状态，不应改变源模型计算。不要直接执行归档脚本：其中故意保留了历史HEAD、模型SHA、绝对路径和run ID。

## 5. 采集专家轨迹

在Gym环境执行record规范，基seed62000，最多100个批次，目标50条成功轨迹；作者入口会按剩余数量收缩最后一批。已经完成过的参考采集为16+16+16+2，seed62000～62003，全部成功。新复现不保证也只需四批，应保存每次失败及原因。

当前record入口已接入整体录像，不能为了省事退回裸作者入口而漏录。若自定义分批编排，必须接入 `overview.install_live`，并为每批设置独立录像/日志目录。作者`run_one`、原始成功筛选和数据记录逻辑保持不变。

数据门槛：

1. 50份成功 `episode_*.npz`，连续编号；每份550帧，总27,500帧。
2. 状态/动作均`[550,9]`、有限；动作字段准确；状态关节值对应记录的实测关节，上一底座指令与前一帧动作一致。
3. 两路depth均`uint8 [550,480,640]`，帧数对应，不能整条恒定或冻结。穿越后某些帧全白/全黑不能单凭颜色判作坏数据，应结合clip范围和场景检查。
4. `fps=25`、`record_sample_stride=2`，实际记录频率约25Hz；`sim_dt`按`float(np.float32(0.02))`核对，不能与Python十进制0.02要求逐位相等。
5. 原始门角按该门运行时方向符号达到80°；DoorTwin的通过、解锁等报告单独记录，不代替源NPZ保存条件。
6. 未启用`record_camera_pose`时，源NPZ可能没有`camera_intrinsics_mode`；从已保存的实际采集命令及源配置确认`legacy`。**不要为迎合检查脚本往原始NPZ补字段。**
7. 每条数据绑定batch seed、env ID和源保存日志。不要以DoorTwin导出的轨迹列表顺序猜测NPZ对应环境。

按上述实际契约检查，不使用准备阶段 `validate_data.py raw` 中强制相机元数据的旧假设。最小检查可直接运行：

```bash
# 在ACT环境，隐藏GPU；只读取当前新run的数据。
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import json, os
from pathlib import Path
import numpy as np
run = Path(os.environ["WC4_RUN"])
names = ["vx","yaw","joint1","joint2","joint3","joint4","joint5","joint6","jointGripper"]
cmd = json.loads((run/"commands/record.json").read_text())["argv"]
assert cmd[cmd.index("--camera_intrinsics_mode")+1] == "legacy"
files = sorted((run/"raw").glob("episode_*.npz"))
assert [p.name for p in files] == [f"episode_{i:06d}.npz" for i in range(50)]
for p in files:
    # 本run自己生成的可信NPZ含object类型的字段名。
    with np.load(p, allow_pickle=True) as d:
        s, a = d["state"], d["action"]
        assert s.shape == a.shape == (550,9)
        assert np.isfinite(s).all() and np.isfinite(a).all()
        assert list(d["action_names"]) == names
        assert d["state_format"].item() == "a2w_last_command_joint_state9"
        assert d["action_format"].item() == "a2w_joint_action9"
        assert d["door_asset_name"].item() == "wc4"
        assert int(d["fps"].item()) == 25
        assert float(d["sim_dt"].item()) == float(np.float32(0.02))
        assert int(d["record_sample_stride"].item()) == 2
        assert np.max(np.abs(s[:,2:] - d["replay_dof_pos"][:,12:19])) < 1e-6
        assert np.max(np.abs(s[1:,:2] - a[:-1,:2])) < 1e-6
        assert float(np.max(float(d["door_motion_sign"].item())
                            * np.rad2deg(d["replay_door_dof_pos"][:,0]))) >= 80
        for view in ["front","wrist"]:
            im = d[view+"_masked_depth"]
            assert im.shape == (550,480,640) and im.dtype == np.uint8
            assert np.any(im.max(axis=(1,2)) > im.min(axis=(1,2)))
            assert np.any(np.diff(im[:,::8,::8].astype(np.int16), axis=0))
print("PASS: 50 episodes / 27500 frames / direct Joint9 / dual depth")
PY
```

最小检查不替代随机化核对、失败账本与人工视频审查。首批零成功、系统性无效图像或错误动作语义时停止，不通过改成功门槛凑50条。源采集器可能在已保存数据后的Gym清理阶段退出异常，须查看已保存数据和原始退出日志，不能因异常立即重采并覆盖。

## 6. 转换并验证实际训练数据

在ACT环境执行convert规范。关键参数为：

```text
convert_door_raw_to_lerobot.py
--state_action_mode joint_state9 --depth_only --fps 25
--image_storage video --video_codec h264 --num_workers 4
```

输入是刚通过检查的raw；输出为 `$WC4_DATA`，禁止覆盖已有数据集。

作者转换器保留物理量，并进行其已有的state/action sanitizer，例如近零角速度处理。应比较“原始NPZ经过同一个作者sanitizer后的值”与Parquet，不能误把该处理当作迁移误差。H.264是有损编码，图像不能要求与原NPZ逐像素相等。

转换门槛：

- `meta/info.json`为50回合、27,500帧、25fps；状态/动作特征均9维，前置与腕部depth键存在。
- `door_dp_feature_names.json`是数据契约的一部分，随数据保留；状态/动作名称、`joint_command`、视觉模式和预处理配置必须一致。
- 全量状态/动作经源sanitizer后逐值比较；每回合timestamp为0、0.04、…。
- 使用**真正训练用的** `lerobot.datasets.factory.make_dataset`、`use_imagenet_stats=False`、ACT chunk100，读取每条轨迹开头/中间/末尾，共150个窗口。
- 检查`action [100,9]`、`action_is_pad [100]`，末尾只按作者规则在该episode内padding，不能跨episode取动作。
- 图像经TorchCodec读为`[3,480,640]`，与对应raw帧/视角匹配；多进程DataLoader实际取batch。
- 两路视频全量可解码，各27,500帧；编码H.264、25fps。不要假设每episode对应一个文件：本次LeRobot将每个视角合并到一个chunk视频。
- 生成数据全文件SHA清单；训练前后核对，不在训练期间重算统计或修改数据。

完整检查行为已列在上述门槛中，不依赖已清理的历史检查脚本。仓库 `validate_data.py lerobot` 只能做较浅的结构检查，不能替代实际窗口读取。

## 7. ACT训练：30k以上，可扩大至40k～50k

在ACT环境使用独立输出和新的W&B run，执行train规范。本文以50,000更新为示例；若新任务选择40,000，设置 `WC4_TRAIN_STEPS=40000` 后重新生成命令。这里的“步”是optimizer更新次数，不是epoch，也不是仿真步。

实际调用作者 `high-level/lerobot/src/lerobot/scripts/lerobot_train.py`，保留：

```text
--dataset.use_imagenet_stats=false
--policy.type=act --policy.device=cuda --policy.push_to_hub=false
--policy.chunk_size=100 --policy.n_action_steps=50
--policy.vision_backbone=resnet18 --policy.optimizer_lr=1e-5
--policy.use_action_loss_weight=false --keyframe_sampling_ratio=0
--policy.plucker_conditioning=false --policy.interaction_state_conditioning=false
--policy.camera_input_gating=false --policy.handle_latent_aux=false
--policy.point_cloud_conditioning=false --policy.end_signal_prediction=false
--seed=1000 --batch_size=16 --num_workers=3
--save_freq=10000 --log_freq=100
--wandb.enable=true --wandb.mode=online --wandb.disable_artifact=true
```

注意：

- ResNet18保留作者默认ImageNet预训练权重，不要误改成整个网络完全随机初始化，也不加载本项目旧DP或ACT任务模型。
- 首次构建模型可能下载官方骨干权重；提前准备缓存可以避免开训时断网。缓存位置由激活后的`XDG_CACHE_HOME`决定。
- 视觉、状态和动作归一化由作者数据factory和processor生成；`use_imagenet_stats=false`不等于关闭ResNet预训练。
- 不启用旧DP的min/max、四元数符号处理、额外停车监督、历史窗口或动作裁剪规则。
- 全部50条参与训练；没有独立验证集loss。训练loss只用于监控数值和趋势，不能替代闭环成功率。
- 首次更新检查有效batch形状、有限loss与梯度；保存有效cfg、数据哈希、模型版本和W&B链接。
- 时间估算用于资源安排，不因“预计时间到了”截断正常训练。到明确更新预算后结束。
- 不需要每2k闭环评测才能复现；必要checkpoint用于恢复，正式终点做一次64回合即可。中期检查需有明确预算。

W&B使用`fi-yu-individual/door-dp-stop-tail`时需要该账号权限；迁移到别的账号只调整实体/项目，不更改算法参数。不要把API key写入仓库、文档或命令证据。旧Blackwell激活脚本会禁用W&B，新包装器最后重新设为online；直接调用旧激活脚本会漏掉这一步。

### 7.1 可选：从已有checkpoint继续训练

恢复是运行能力，不是必须重走的历史阶段。比如一次训练已保存40k、任务明确允许到50k，可以从40k恢复，`--steps=50000`表示**累计终点**，不是再加50k。

需要保留完整目录：

```text
checkpoints/040000/
  pretrained_model/     模型、train_config、pre/postprocessor及统计
  training_state/       optimizer、param groups、RNG、training_step
```

只给`--policy.pretrained_path`相当于加载模型，不足以证明完整续训。使用作者`--config_path=<checkpoint>/pretrained_model/train_config.json --resume=true`。

为不续接历史W&B run，先新建一个在线run并保存其ID，再让作者恢复入口以该ID执行；作者恢复时使用`resume="must"`，不能给一个尚未创建的随机ID：

```bash
# ACT环境；WC4_RESUME_RUN必须是新的独立输出目录。
export WC4_PARENT_CHECKPOINT="$WC4_RUN/train/checkpoints/040000"
export WC4_RESUME_RUN="$WC4_REPO/.repro/runs/${WC4_RUN_ID}-resume50k"
test ! -e "$WC4_RESUME_RUN"
mkdir -p "$WC4_RESUME_RUN"
python - <<'PY'
import json, os
from pathlib import Path
import wandb
out = Path(os.environ["WC4_RESUME_RUN"])
with wandb.init(entity=os.environ["WANDB_ENTITY"], project=os.environ["WANDB_PROJECT"],
                mode="online", name=out.name, group=out.name, job_type="train-resume",
                dir=str(out), config={"parent_checkpoint": os.environ["WC4_PARENT_CHECKPOINT"]}) as run:
    receipt = {"id": run.id, "url": run.url}
(out/"wandb-bootstrap.json").write_text(json.dumps(receipt, indent=2)+"\n")
PY
WC4_NEW_WANDB_ID=$(python -c \
 'import json,os; print(json.load(open(os.environ["WC4_RESUME_RUN"]+"/wandb-bootstrap.json"))["id"])')
python high-level/lerobot/src/lerobot/scripts/lerobot_train.py \
  --config_path="$WC4_PARENT_CHECKPOINT/pretrained_model/train_config.json" \
  --resume=true --steps=50000 --save_freq=2000 \
  --output_dir="$WC4_RESUME_RUN/train" --job_name="${WC4_RUN_ID}-resume50k" \
  --wandb.run_id="$WC4_NEW_WANDB_ID" --wandb.enable=true --wandb.mode=online
```

执行前仍需GPU租约和日志监督。恢复前核对模型权重、optimizer步数、RNG及归一化统计；恢复后仅计新更新。该作者实现会重建DataLoader迭代器，不能声称恢复与不中断训练逐位等价。冻结统计和模型来源，不能把恢复当成重新初始化优化器的“另一次独立训练”。

## 8. 导出 Door checkpoint 与实际模型桥接

在ACT环境执行export规范。生成器默认取该run的`train/checkpoints/last`；为避免歧义，正式交付应解析它并冻结为具体数值目录，例如`050000`。如果训练用了可选恢复run，显式替换checkpoint路径，同时继续引用**原数据集**：

```bash
export WC4_FINAL_CHECKPOINT="$WC4_RUN/train/checkpoints/$(printf '%06d' "$WC4_TRAIN_STEPS")"
# 若使用可选恢复run，改为该run中明确的数值checkpoint路径。
test -d "$WC4_FINAL_CHECKPOINT/pretrained_model"
test ! -e "$WC4_RUN/door-checkpoint"
python high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
  --official_checkpoint "$WC4_FINAL_CHECKPOINT" \
  --root "$WC4_RUN/lerobot" --repo_id "$WC4_DATA_ID" \
  --out_dir "$WC4_RUN/door-checkpoint" --depth_only --device cpu \
  --action_horizon 25
```

`model_latest.pt`是Door入口manifest，旁边的模型目录、统计和metadata必须一起保留，不是只拷一个`.pt`就完成模型搬迁。官方训练checkpoint和实际数据sidecar也要保留。

正式闭环前，按`reference_tools/export_and_check.py`、`reference_tools/check_gpu_bridge.py` 的方法，检查：

1. 官方与导出权重逐张量一致，使用同一训练统计；实际state/action为9维。
2. 相同真实数据输入经官方与导出预处理后的最大差异≤1e-6；物理动作输出差异≤1e-5。
3. 实际Gym Python→ACT GPU子进程相对CPU参考的动作一致性，原门槛为`atol=rtol=1e-3`。这是设备数值容差，不能随失败放宽。
4. checkpoint配置chunk100，控制器实际执行25，输入状态和图像均有限，输出为Joint9。

这一步需要构造模型和推理，不能用此前“Unknown worker command / close正常”协议测试代替。历史`export-reference`用了5个真实窗口，输出形状`[5,100,9]`，并未新开仿真。

源导出metadata可能仍带未使用的旧EE10顶层描述。实际裁决依据是policy config、sidecar、动作名、controller动作模式和运行日志。保留原产物并附说明，不通过编辑原权重伪造元数据一致。

## 9. 闭环评测与整体录像

在Gym环境、持有空闲GPU租约后执行evaluate规范。生成器调用 `evaluate_with_overview.py`，其内部仍使用作者评测器与控制器。

关键设置：

```text
--mode ikpush --robot_body a2wz1
--num_envs 16 --total_trials 64 --parallel_batches 1
--steps 1100 --base_seed 615455575
--pass_open_angle_deg 80 --success_metric abs
--dp_fps 25 --dp_action_horizon 25 --dp_log_interval 1
--depth_only --headless --graphics_device_id 0
```

其余门、资产、skill、俯仰、相机、depth clip等参数必须与采集一致，由`commands.py`共同参数生成。单个64回合run应使用615455575～615455578四个batch seed；seed相同不自动证明初态相同，跨模型比较还需核对实际随机化和初始状态。

验收：

- 每环境完整1100条仿真日志，`phase_name=dp_policy`、`action_mode=a2w_joint_action9`，9维状态/动作有限；无专家回退或人为覆盖。
- 从原始日志独立计算每回合最大绝对门角；汇总与作者`summary.json`一致。日志50Hz与策略/视频25Hz不要混淆。
- 本次一个模型的分母是64；失败不重跑替换。基础设施异常单列，不悄悄删除后重算。
- 主指标只看开门80°。穿越、解锁标志、碰撞和停稳可作诊断，不能临时加进或移出成功条件。
- 第三视角RGB整体录像默认覆盖每环境，1920×640双侧视角，25fps、550帧、22秒。视频包含全身、门、接近、操作、通过及结束阶段。
- 若新任务明确允许减量，按本轮已采用方式：在既定64次内同步录制候选，保留最早成功和最早失败各一段，两类齐备后可关闭后续批次录像。缺类如实记录，不补跑。只处理本run新生成的未选临时片，保留其哈希/receipt和全部日志。
- 状态回放标注`RECORDED STATE REPLAY`；它只能补充可视化，不能冒充原始闭环录像或新增试验。

`eval_door_dp_on_expert_obs.py`只是离线动作比较，不能作为此处成功率评测入口。原作者一些失败启发式含旧10维下标，不把它的自动根因描述当成Joint9事实。

## 10. 已取得结果及正确解释

最终50k模型在固定64次闭环中成功48次（75%）。模型SHA：

```text
3f1f32f96eb3a78abd215f6affa6786d7b2cf76b94def6b8268bdeefe3bfbdb9
```

完整产物定位和有效配置见[BASELINE.json](BASELINE.json)。训练量30k以上已取得有效行为，可按任务扩大到40k～50k；训练loss下降不保证闭环成功率单调提高，也不要求新训练产生同一模型SHA。

本次失败视频显示末端未能成功对准门把手，16次失败均未见解锁标志。正接触不等于正确抓握，当前结果没有因果区分感知、姿态、动作时序或接触维持问题。开门80°是本任务唯一主判据，穿越、碰撞合理性和停车未被纳入验收。

**用户已接受当前成果，模型训练任务完成。** 旧试验、失败准备记录和中间checkpoint已清理，不是后续复现的前置条件。

## 11. 必须记住的实现陷阱

| 现象 | 正确处理 |
|---|---|
| 作者分支更新 | 固定commit，不追latest |
| sparse checkout安装失败 | `setup.py`和打包文件必须存在，不能只取LeRobot `src/` |
| 普通Python3.8 Torch不能运行Blackwell | 使用本项目sm120构建；不能仅看CUDA可见 |
| CUDA/Vulkan看到不同物理卡 | 按UUID对齐并使用私有层；逻辑graphics ID为0 |
| TorchCodec缺`libavutil`等 | 新ACT环境补匹配FFmpeg共享库；保持默认解码器 |
| W&B静默disabled | 使用新激活包装器，核对真正online run；恢复用新ID |
| NPZ没有相机内参字段 | 按实际源保存条件验证命令，不往原数据补字段 |
| `dt == 0.02`断言失败 | 按Isaac Gym的float32表示核对，不改仿真dt |
| 原raw准备校验器误拒绝数据 | 用真实schema门槛；不要盲目执行生成器shell guard |
| ACTConfig直接加载含`type`的JSON报错 | 用作者`PreTrainedConfig`分派，不改checkpoint JSON |
| 检查模型时batch包含task字符串 | 仅传源模型需要的state、图像等输入特征 |
| optimizer step审计找不到键 | safetensors实际键为`state/<id>/step`，不是点分隔 |
| Joint9日志`ee.pos_error`接近零 | 它不等于独立把手目标误差，不能据此证明准确对准 |
| 固定专家动作回放失败 | 不等于反馈专家失败；状态回放、动作回放、真实闭环须分清 |
| 出现失败回合 | 原样计入分母；保留证据，不重新挑seed刷成功率 |
| 归档脚本硬编码历史HEAD/路径 | 仅作参考；新run复制适配外围配置，不改历史归档 |
| 50k之后loss仍下降 | 不构成自动续训授权，也不证明闭环改善 |

## 12. 交付清单和证据导航

每次真正重新复现至少交付：

1. 作者源及执行提交、工作区差异、环境导出、实际模块路径和GPU租约。
2. 完整命令、有效参数、attempt/batch账本、失败和修复记录。
3. NPZ、数据sidecar、转换数据、SHA清单、实际训练窗口及视频检查。
4. 每个保存checkpoint的模型、optimizer、RNG、统计和更新数；W&B在线链接及本地进度对应。
5. Door导出目录、官方/导出一致性、实际GPU worker检查。
6. 每回合原始日志、重算成功率、与model/seed/env绑定的原始视频。
7. 最终停止状态；分别说明工程通过范围和策略效果，不以loss替代成功率。

不要求在Git中保存大模型、原始视频和环境缓存。原始证据只封存，不改写；本次文档编写也没有重新执行成功实验。

### 当前保留的成功产物

服务器新代码的 `.repro/runs/` 下仅保留两个成功基线目录：

- `wc4-joint9-skill50-s62000-20261015`：50条原始数据、LeRobot数据、哈希与数据核验、采集命令和双深度示例视频。
- `wc4-act-resume40k-50k-s1000-20261015`：50k最终模型/恢复状态、Door导出、最终64次原始日志、有效配置、训练进度和代表视频。

本地 `/Users/yu/Data/Video2Door/reproduction/wc4-act/` 保存上述成功结果的轻量镜像；完整模型和50份NPZ以服务器为准。`BASELINE.json`给出绝对位置。最终评测中的失败回合日志和代表视频属于完整64次结果，保留用于准确解释75%成功率。

服务器 `/data/zyf/video2door` 已精简为Gym环境、Blackwell支持、已安装的editable依赖源码及共享GPU锁目录，不再保留旧Git工作区或旧DP实验。这个支持目录不可整棵删除，否则会破坏当前环境。其路径与第2节环境重建布局一致。

Git内入口：

- [固定工程参数](config.json)、[命令生成器](commands.py)、[环境包装器](activate.sh)。
- [录像实现](overview.py)、[项目约定](../../AGENTS.md)。
- [最终成功基线](README.md)、[产物索引](BASELINE.json)。
- [检查实现参考](reference_tools/README.md)、[环境附件](reproduction_support/README.md)。

W&B最终训练：`xobd7ef0`；最终评测：`lj85duyv`，项目为`fi-yu-individual/door-dp-stop-tail`。新复现创建新run，不续接这些历史ID。
