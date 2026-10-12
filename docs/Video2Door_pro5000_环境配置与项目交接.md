> 历史环境交接，保留当时部署背景；当前状态见 [最新入口](FLOAT_PUSH_TRAINING_NEXT.md)。服务器连接和本地引用已更新，模型训练与评估以 [正式报告](FLOAT_PUSH_LONG_TRAINING_REPORT.md) 为准。

# Video2Door：pro5000 环境配置

更新日期：2026-10-11，汇总本次对话的部署过程和已验证状态，供新对话接手使用

## 1. 当前结论

**用户当前要求的阶段已经完成：在 pro5000 的独立 conda 环境中，使用现有门资产和浮动底座 IK 入口，完整执行成功推门，无需正式训练。**

| 项目 | 当前状态 |
| --- | --- |
| 服务器代码 | `/data/zyf/video2door`，已完成克隆 |
| 推荐环境 | `/data/zyf/video2door/.repro/envs/b1z1-blackwell` |
| Blackwell 与旧 Isaac Gym/PyTorch 兼容 | 已解决并验证 |
| 浮动底座推门 | 2850 步完整通过，最终开门约 89.98° |
| 穿过门框 | 通过，按脚本尾端偏移计算，尾端越过门平面约 0.55 米 |
| 相机 | 手腕、前置相机的 RGB、有效深度、把手分割采集通过 |
| 正式训练 | 未启动；当前用户要求无需启动 |
| 项目源码和 YAML | 未为本次复现修改，服务器 Git 工作树干净 |
| 本项目后台任务 | 验证进程已退出；整理文档时未发现仍在运行的推门验证进程 |

启动命令：

```bash
ssh pro5000
bash /data/zyf/video2door/.repro/float-push/run.sh
```

该脚本自动激活正确环境和 CUDA/Vulkan 设备映射，进行一次有限步数的脚本化推门，保存证据并检查成功条件，**不会执行训练**。

## 2. 用户目标、范围变化及约束

本次对话按以下顺序推进：

1. 最初要求把本地 `/Users/yu/Data/Video2Door/code` 项目部署到 SSH 服务器 `pro5000`，克隆到 `/data/zyf/video2door`，以跑通 README 各部分的环境为目标。
2. 发现 Blackwell 显卡与旧 Python/PyTorch/Isaac Gym 栈的版本断层后，用户要求参考提供的 CSDN 方法，尝试在当前服务器上解决。
3. 用户随后明确缩小目标：**不需要补齐缺少的 663 组门资产，不需要完整低层运动控制模型；采用浮动底座入口，满足成功推门的环境即可，继续配置环境但无需启动正式训练。**

后续工作保留的约束：

- 使用 conda 隔离。新增克隆、下载和环境放在 `/data/zyf` 下，优先继续放在项目的 `/data/zyf/video2door/.repro` 内。
- 尽量不要动服务器其他环境、代码、进程；本次未更改系统驱动、系统 CUDA、base、其他 conda 环境或 shell 启动配置。
- 服务器有其他用户/项目的 GPU 工作负载，不要停止这些任务；运行前重新查看 GPU 占用，不要把历史占用当作当前空闲情况。
- 不需要完成完整 README 中补齐大数据集的门资产或低层权重。


## 3. 服务器、代码来源与硬件

| 项目 | 信息 |
| --- | --- |
| SSH 别名 | `pro5000` |
| 本地 SSH 配置 | `/Users/yu/.ssh/config` |
| SSH 用户/地址 | `ubuntu@172.81.242.38`，以现有 SSH 配置为准 |
| 服务器主机名 | `VM-32-10-ubuntu` |
| 操作系统/架构 | Ubuntu 22.04.4 LTS，Linux x86_64 |
| GPU | 8 张 NVIDIA RTX PRO 5000 72GB Blackwell |
| GPU 计算能力 | `12.0`，架构 `sm_120` |
| 驱动 | `580.126.20` |
| 系统 CUDA | 13.0，历史核对系统 nvcc 为 13.0.88；未修改 |
| conda 可执行文件 | `/data/miniconda3/bin/conda` |
| 本地项目仓库 | `/Users/yu/Data/Video2Door/code` |
| 服务器项目仓库 | `/data/zyf/video2door` |
| 实际项目仓库 | [YuFei-yuu/doorgym](https://github.com/YuFei-yuu/doorgym) |
| 本地 origin | `git@github.com:YuFei-yuu/doorgym.git` |
| 服务器 origin | `https://github.com/YuFei-yuu/doorgym.git` |
| 已部署提交 | `0acc2f050479fb54a0670ccb90bc72b98b52a54a` |

服务器直连 GitHub 下载缓慢，部署时从本地仓库生成完整 Git bundle 后在服务器克隆，保留 Git 历史并设置官方 origin。根 README 仍有原始 `visual_whole_body` 项目的介绍和克隆示例；当前接手时应以已部署的 `doorgym` 仓库及上述提交为准。

整理文档时：服务器工作树干净；辅助脚本存放在服务器 `.repro` 中，已通过 `.git/info/exclude` 排除，不混入主项目源码提交。

## 4. 推荐环境配置清单

环境是 **prefix 环境**，不是依赖全局同名环境：

```text
/data/zyf/video2door/.repro/envs/b1z1-blackwell
```

| 组件 | 已安装/实测版本或设置 |
| --- | --- |
| Python | 3.8.20 |
| Isaac Gym | 仓库自带版本，editable 安装；`setup.py` 标记 `1.0.preview3`，实际使用 `gym_38.so` |
| PyTorch | `2.3.1+sm120`，从源码定制编译 |
| `torch.version.cuda` | `12.8` |
| `torch.cuda.get_arch_list()` | `['sm_120']` |
| CUDA 工具链 | conda 私有 CUDA 12.8.1 系列，nvcc 12.8.93 |
| cuDNN | `nvidia-cudnn-cu12==9.7.1.26`，运行时版本 `90701` |
| torchvision | 运行时 `0.18.1+cu121`，其 CUDA 编译版本为 12.1 |
| torchaudio | `2.3.1+cpu` |
| NumPy / SciPy | 1.23.5 / 1.10.1 |
| PyYAML / OpenCV | 6.0.2 / `opencv-python==4.10.0.84` |
| diffusers / huggingface-hub | 0.24.0 / 0.20.3 |
| Gym / Gymnasium | 0.26.2 / 0.29.1 |
| Pillow / imageio / imageio-ffmpeg | 10.4.0 / 2.35.1 / 0.5.1 |
| Matplotlib / TensorBoard / wandb | 3.7.5 / 2.14.0 / 0.17.9 |
| ninja / pydelatin | 1.11.1.3 / 0.2.8 |
| 本地 editable 包 | `isaacgym`、`rsl_rl`、`skrl`、`legged_gym` |
| 编译器 | `/usr/bin/gcc`、`/usr/bin/g++` |
| C++ ABI | `_GLIBCXX_USE_CXX11_ABI=0`，运行时实测为 false |
| 默认计算/渲染卡 | 物理 GPU 0，经私有映射后均为逻辑设备 0 |
| 运行 CPU 线程 | OMP/MKL/OpenBLAS 均为 2 |
| 编译并发 | `MAX_JOBS=16`，构建使用 nice 10 |
| wandb | `WANDB_MODE=disabled` |
| 依赖检查 | 整理文档时 `python -m pip check` 再次通过 |

环境还包含 pip 的 `nvidia-cublas-cu12==12.9.2.10`、`nvidia-cuda-nvrtc-cu12==12.9.86` 和 `nvidia-nvtx-cu12==12.1.105`。它们与 conda 中的 CUDA 开发包并存；不要把所有 NVIDIA 包笼统改为同一版本。当前组合以已验证环境及精确导出为准。

完整版本清单：

- 服务器：`/data/zyf/video2door/.repro/blackwell/pip-freeze.txt`、`/data/zyf/video2door/.repro/blackwell/conda-explicit.txt`。
- 本地：[pip-freeze.txt](/Users/yu/Data/Video2Door/reproduction/pro5000/blackwell/pip-freeze.txt)、[conda-explicit.txt](/Users/yu/Data/Video2Door/reproduction/pro5000/blackwell/conda-explicit.txt)。
- 安装依赖清单：`/data/zyf/video2door/.repro/blackwell/requirements-runtime.txt`；约束文件：`/data/zyf/video2door/.repro/blackwell/constraints.txt`。

`pip-freeze.txt` 中 torchvision 元数据记作 `0.18.1`，运行时带 `+cu121`；torch 则记录为指向定制 wheel 的本地 URL。不要仅根据包名或 CUDA 后缀推断兼容性。

## 5. 环境启动与推门命令

### 5.1 推荐：一次完整推门并保存证据

在本地终端：

```bash
ssh pro5000
```

在服务器执行：

```bash
bash /data/zyf/video2door/.repro/float-push/run.sh
```

启动脚本会设置独立环境、选择物理 GPU 0、对齐 Vulkan、使用 2 个 CPU 线程和 nice 10，运行单场景 2850 步。超时上限 240 秒；退出码非 0 或结果检查失败时，查看本次输出目录的日志。

无需相机采集时：

```bash
bash /data/zyf/video2door/.repro/float-push/run.sh --no_show_camera_images
```

换卡时，先用 `nvidia-smi` 查看当前负载，确认目标卡后，在启动/激活前设置 `VIDEO2DOOR_GPU`。例如选择物理 GPU 0：

```bash
VIDEO2DOOR_GPU=0 bash /data/zyf/video2door/.repro/float-push/run.sh
```

`VIDEO2DOOR_GPU` 支持 nvidia-smi 能识别的索引或 UUID。不要在激活后只覆盖 `CUDA_VISIBLE_DEVICES`，否则可能再次让相机和 CUDA 落在不同物理卡上。

### 5.2 只激活正确环境

在 pro5000 的专用 Bash 会话中：

```bash
source /data/zyf/video2door/.repro/blackwell/activate.sh
which python
python -m pip check
```

激活脚本会切换工作目录到 `/data/zyf/video2door`，除了 conda 还配置 CUDA/cuDNN 动态库、编译器、缓存、线程限制和私有 Vulkan 层。因此仅执行 `conda activate` 不完全等价。

建议在专用 SSH/Bash 会话里运行，结束后退出该会话。`conda deactivate` 不保证撤销脚本手动导出的全部 CUDA/Vulkan 变量，但这些变量不会修改其他已有进程或系统配置。

### 5.3 直接运行原项目入口，不采集相机

这条命令已经完整通过，可用于区分原脚本与额外的检查包装层：

```bash
source /data/zyf/video2door/.repro/blackwell/activate.sh
python -u /data/zyf/video2door/high-level/float_ik/isaacgym_float_ik_b1z1_basearn_push_door.py \
  --asset_root /data/zyf/video2door/low-level/resources/robots/b1z1 \
  --door_name 99650089960001 \
  --headless --nographics --no_show_camera_images \
  --steps 2850 --num_threads 2 --log_interval 60
```

原脚本的机器人资产默认路径写死为作者目录，所以需要 `--asset_root` 覆盖。服务器没有桌面显示时使用推荐包装层采集相机，避免原脚本的 OpenCV 窗口显示。

**不要误用以下历史入口：**

- `/data/zyf/video2door/.repro/activate.sh` 指向旧的 `b1z1` 环境。
- `/data/zyf/video2door/.repro/verify.sh` 属于最早的失败环境诊断。
- `/data/zyf/video2door/.repro/blackwell/verify.sh` 是后来完整兼容性验证，包含两轮低层 debug 训练，还会重写历史日志和测试模型。在无需训练的阶段不执行。

## 6. 成功推门的证据及结果定义

实际入口：

```text
/data/zyf/video2door/high-level/float_ik/isaacgym_float_ik_b1z1_basearn_push_door.py
```

使用机械臂 IK 和脚本给定的底座位姿轨迹，顺序执行靠近、预抓取、抓取、闭合夹爪、压把手、推门、回收机械臂、保持。未加载低层行走策略。

原入口采用 **GPU PhysX + CPU tensor/API pipeline** 进行 IK 和控制。日志中的 `GPU Pipeline: disabled` 是此入口的设计，不代表物理仿真回退到 CPU。独立的 GPU tensor pipeline 已在兼容性测试中另行通过。

| 指标 | 已保存结果 |
| --- | --- |
| 门资产 | `99650089960001` |
| 步数 | 2850 |
| 无相机运行耗时 | 约 15.53 秒，退出码 0 |
| 带相机运行耗时 | 约 17.50 秒，退出码 0；不包括全部启动开销 |
| 初始门角度 | 0° |
| 最大把手角度 | 约 45°，成功解锁 |
| 最大开门角度 | 约 90° |
| 最终开门角度 | 89.9802496° |
| 最终底座坐标 | `[1.30000, -0.02462, 0.60000]` 米 |
| 尾端越过门平面 | 按脚本 0.65 米尾端偏移计算，为 0.55000005 米 |
| 自动开门扭矩 | 参数保持 0 |
| 手腕/前置相机 | 各采集 24 帧，深度与分割尺寸为 `[54,96]` |
| 最大有效把手深度像素 | 手腕 464，前置 106 |
| 结果 | `passed=true`，`training_started=false` |

`verify_push.py` 包装原脚本收集观测和检查结果，并把相机窗口替换为文件保存；没有改写原来的资产加载、IK、轨迹或物理计算函数。成功结果不是通过直接设置开门角度获得的。

检查项包括：完整阶段、初始关门、把手解锁、最大及最终开门角度至少 80°、底座尾端通过门平面、自动开门扭矩为 0、最终状态有限值，以及启用相机时获得有效把手深度。

### 输出路径

每次新运行会创建：

```text
/data/zyf/video2door/.repro/float-push/runs/<时间戳>-<进程号>/
```

已通过的带相机运行目录：

```text
/data/zyf/video2door/.repro/float-push/runs/20261010-235824-420064/
```

其中 `summary.json` 为结构化结果，`run.log` 为执行日志；PNG 为相机 RGB、把手掩码和场景概览，NPZ 为原始深度/分割数组。原始 Gym 深度保留负距离及无穷远值约定，读取时应筛选有限负值。


## 7. 其他已经通过的兼容性验证

下表是前一阶段的历史验证，不是当前需要重新运行的任务：

| 检查 | 结果及范围 |
| --- | --- |
| 原生 CUDA | CUDA 12.8 nvcc 编译并执行 sm_120 内核通过 |
| PyTorch CUDA | 实际张量运算通过，原来的 no kernel image 错误消除 |
| gymtorch | C++ 扩展编译、导入通过 |
| GPU PhysX/GPU tensor pipeline | 4 个环境、40 步，状态张量 `[4,13]` 位于 `cuda:0` |
| 控制和优化器链路 | TorchScript、MLP、Adam 反向/更新、GPU 力张量传入仿真通过 |
| cuDNN | Conv2d 前向/反向通过，运行时 90701 |
| GPU 深度相机 | CUDA 张量 `[120,160]`，有效深度像素 15040 |
| 原项目低层训练入口 | 128 个环境，两轮 debug：DAgger 第 0 轮、PPO 第 1 轮，共 6144 transitions，保存模型成功 |
| DoorDiffusionPolicy | 合成输入 GPU 前向/反向/AdamW、DDPM scheduler step、torchvision CUDA resize 通过，输出 `[1,32,10]` |
| 依赖元数据 | pip check 通过，整理文档时再次确认 |

低层测试模型位于：

```text
/data/zyf/video2door/low-level/logs/repro-smoke/blackwell_smoke/model_0.pt
/data/zyf/video2door/low-level/logs/repro-smoke/blackwell_smoke/model_2.pt
```

它们仅用于证明训练入口能运行，不能替代 README 所需的充分训练低层策略。DP 测试使用合成输入，没有真实数据集或训练好的 checkpoint，不能代表 DP 策略效果。

## 8. 文件与缓存位置索引

### 服务器目录

| 路径 | 用途 |
| --- | --- |
| `/data/zyf/video2door` | 主项目 Git 仓库 |
| `/data/zyf/video2door/.repro/envs/b1z1-blackwell` | 当前推荐 conda 环境 |
| `/data/zyf/video2door/.repro/envs/b1z1` | 最初官方 PyTorch 环境，保留诊断，不推荐使用 |
| `/data/zyf/video2door/.repro/blackwell` | 兼容修复源码、工具链、构建脚本、导出清单、日志 |
| `/data/zyf/video2door/.repro/float-push` | 当前推门启动器、验证包装器、报告及运行证据 |
| `/data/zyf/video2door/.repro/blackwell/activate.sh` | 正确的完整环境激活入口 |
| `/data/zyf/video2door/.repro/float-push/run.sh` | 推荐的无训练推门验证入口 |
| `/data/zyf/video2door/.repro/float-push/verify_push.py` | 结果检查及离屏相机文件保存 |
| `/data/zyf/video2door/low-level/resources/robots/b1z1/urdf/b1z1_basearn.urdf` | 当前入口使用的机器人原始 URDF |
| `/data/zyf/video2door/high-level/data/cfg/b1z1_opendoor.yaml` | 门资产索引所在配置，仍保留完整 667 组定义 |
| `/data/zyf/video2door/high-level/data/asset/door_set` | 现有门资产 |
| `/data/zyf/video2door/.repro/blackwell/logs` | Blackwell 兼容性验证和构建日志 |
| `/data/zyf/video2door/.repro/float-push/REPORT.txt` | 当前阶段报告 |
| `/data/zyf/video2door/.repro/blackwell/REPORT.txt` | 兼容性修复及历史完整范围报告，顶部有最新说明 |
| `/data/zyf/video2door/.repro/REPORT.txt` | 最早报告，主体是旧失败记录，顶部有更新说明 |

### 本地目录

| 路径 | 用途 |
| --- | --- |
| `/Users/yu/Data/Video2Door/code` | 原本地项目仓库 |
| `/Users/yu/Data/Video2Door/reproduction/pro5000` | 部署辅助文件和历史诊断的本地留存 |
| `/Users/yu/Data/Video2Door/reproduction/pro5000/blackwell` | Blackwell 修复脚本、清单及精选日志副本 |
| `/Users/yu/Data/Video2Door/reproduction/pro5000/float-push` | 当前推门脚本、报告及证据副本 |
| `/Users/yu/Data/Video2Door/code/docs/Video2Door_pro5000_环境配置与项目交接.md` | 本交接文档 |

本地主要留存辅助源码和部分日志；完整 conda 环境、PyTorch 源码构建目录、wheel 及运行时 Vulkan 二进制在服务器，不能假设本地副本包含全部产物。

### 缓存隔离

激活脚本配置的项目私有缓存：

| 变量 | 路径 |
| --- | --- |
| `CONDA_PKGS_DIRS` | `/data/zyf/video2door/.repro/conda-pkgs` |
| `PIP_CACHE_DIR` | `/data/zyf/video2door/.repro/cache/pip` |
| `XDG_CACHE_HOME` | `/data/zyf/video2door/.repro/blackwell/cache` |
| `TORCH_EXTENSIONS_DIR` | `/data/zyf/video2door/.repro/blackwell/cache/torch_extensions` |
| `CUDA_CACHE_PATH` | `/data/zyf/video2door/.repro/blackwell/cache/cuda` |
| `TMPDIR` | `/data/zyf/video2door/.repro/tmp` |

**例外**：Isaac Gym 自带 VHACD 几何缓存仍使用 `/home/ubuntu/.isaacgym/vhacd`，日志已有缓存命中。未清理该共享缓存目录；不要把“环境放在项目内”理解成 Gym 的所有内部缓存都已经重定向。

## 9. 未做的部分与当前不构成阻塞的事项

门配置共引用 667 组门，仓库只有以下 4 组完整资产，缺少 663 组、1989 个配置引用文件：

```text
/data/zyf/video2door/high-level/data/asset/door_set/99650089960001
/data/zyf/video2door/high-level/data/asset/door_set/99650089960006
/data/zyf/video2door/high-level/data/asset/door_set/99655039960001
/data/zyf/video2door/high-level/data/asset/door_set/99655039960006
```

当前只完整验证了第一组门。启动时明确指定 `--door_name 99650089960001`，不遍历缺失门，因此不需要修改主 YAML。其余 3 组不能直接声明已经逐组成功。

此前完整 README 的其他缺口包括：

- `b1z1_opendoor.yaml`、`b1z1_pickmulti.yaml` 中的低层权重路径仍指向作者目录 `/home/sivan/whole_body/visual_whole_body/low-level/logs/b1z1-low/b1z1_locomanip/model_45000.pt`，服务器没有该文件。
- README 给出的低层权重 Google Drive 下载端点曾在服务器超时，未取得并验证兼容权重。该网络情况是历史记录，不代表今后一定无法下载。
- 完整行走控制、teacher/student、完整门集合、DP 数据录制/转换/正式训练/播放、GUI 交互等没有完成端到端验证。
- 需要 Python 3.10 或更高版本的独立 LeRobot/DP 数据转换和训练环境尚未创建。
- torchvision 的独立 CUDA 扩展、定制 PyTorch 的分布式能力和其他 GPU 架构未在本任务内验证。

这些事项**不是当前浮动底座推门目标的剩余工作**。除非用户重新扩大范围，否则无需自动补齐、迁移到 Isaac Lab、升级系统栈或启动训练。

## 10. 新对话接手时的建议顺序

1. 先读取本文及本地推门 `summary.json`，确认用户最新目标；当前阶段已完成，没有待继续的正式训练任务。
2. 若用户要求再次运行，先 `ssh pro5000`，用 `nvidia-smi` 查看即时负载，并核对仓库状态。
3. 使用 `/data/zyf/video2door/.repro/float-push/run.sh`，避免旧环境激活器和带训练的 `blackwell/verify.sh`。
4. 以本次运行目录的 `run.log`、`summary.json` 及退出码判断结果，不能仅看 `torch.cuda.is_available()`。
5. 若出现 CUDA 架构错误，先检查是否误激活旧环境、定制 torch 是否被替换；若仅相机报错，先检查 UUID 映射和私有 Vulkan 层。
6. 只在具体故障证据指向环境损坏时考虑修复或重建；现有成功环境不需要默认重装。

不执行训练的基础诊断命令，在服务器 Bash 中运行：

```bash
source /data/zyf/video2door/.repro/blackwell/activate.sh
nvidia-smi
which python
python -m pip check
git -C /data/zyf/video2door status --short
python - <<'PY'
from isaacgym import gymapi  # 保持 Isaac Gym 在 torch 之前导入
import torch
print("torch:", torch.__version__)
print("compiled CUDA:", torch.version.cuda)
print("architectures:", torch.cuda.get_arch_list())
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA arithmetic:", (torch.ones(16, device="cuda:0") + 1).sum().item())
PY
```

预期关键输出为 `2.3.1+sm120`、`12.8`、`['sm_120']`、`90701` 和运算结果 `32.0`。仓库的 Isaac Gym 对导入顺序敏感，编写新诊断脚本时先导入 Isaac Gym 再导入 torch。
