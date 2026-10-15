# wc4 ACT Joint9：迁移准备

人工闭环审查视频统一采用第三视角RGB整体录像 `rollout.mp4`，成功和失败均保留；前置/腕部深度视频仅为补充。录制入口默认使用相反两侧观察相机拼接（1920×640、25fps），避免门墙遮挡通过后的机器人。`commands.py record/evaluate` 已接入录像包装器，作者控制器、策略输入、skill和物理配置未修改。
自定义编排必须使用 `overview.install_live` 或相同包装器。每个并行环境都有独立完整视频；帧数不足会保留部分录像并报错。录像标注 `LIVE EXECUTION`，包含帧号、时间、阶段和门角。
历史漏录可用 `replay_with_overview.py OUTPUT -- --raw_episode NPZ --mode ikpush --replay_mode state --headless --no_show_seg --no_real_time` 补录，标注 `RECORDED STATE REPLAY`，不能作为新增闭环成功结果。本次补录证据在本地 `reproduction/wc4-act/overview-review-20261015/` 和服务器 `/data/zyf/wc4-act-evidence/overview-review-20261015/`。

2026-10-15。仅准备环境、命令和无仿真检查，未采集、训练、仿真或验证策略效果。

**状态：迁移准备复核通过，等待独立验收。** 本轮仅补齐新 ACT 环境内的 FFmpeg 共享库及必要依赖；未采集、训练、仿真或策略推理。通过的是准备检查，真实数据和策略效果仍待后续授权验证。

上一轮 `preparation-20261015/video-codec-check.log` 的共享库缺失证据保留原样。按本轮任务书补库后，原 `check_video.py`（文件哈希不变）仍使用作者默认 TorchCodec，H264 两帧按0和0.04秒采样、25Hz，输出 `[2,3,480,640]`，均值约0.250980407和0.501960814；形状、时间容差1e-4及数值断言全部通过，未放宽检查。

新 ACT prefix 新增 defaults 渠道 `ffmpeg=6.1.1=hecf7045_5` 及41项依赖，没有移除或改变原有包记录。安装前冻结求解并审查，再按42个精确URL及校验值安装。FFmpeg7的首份冻结求解与现有库约束冲突，未安装；改用同一TorchCodec支持的FFmpeg6，不切换解码器。Torch2.7.1+cu128、TorchCodec0.4.0、PyAV15.1.0、NumPy2.2.6与定制LeRobot0.4.4保持不变。

实际解码加载新 ACT prefix 内 `libtorchcodec_decoder6.so`、`libavutil.so.58.29.100`、`libavcodec.so.60.31.102` 等库；PyAV原有带哈希文件名的vendored库用于编码，同进程加载路径、库版本和SHA256见本轮 `loaded-shared-libraries.json`。`pip check`、定制ACT类导入与Python3.8→作者worker启动/消息协议复核通过；未构造模型或加载checkpoint，未重复资产审计。

本轮证据：本地 `/Users/yu/Data/Video2Door/reproduction/wc4-act/ffmpeg-recheck-20261015/`；服务器 `/data/zyf/wc4-act-evidence/ffmpeg-recheck-20261015/`。包括安装前后快照、求解/安装日志、精确安装清单、原脚本输出、共享库映射、环境差异和恢复方案。第一次前后探测的用户级包可见性不同，原输出保留；同条件复核证明pip freeze、核心包元数据及旧环境分布完全一致。

恢复本次补库时使用本轮证据中的 `rollback-additions-only.sh`：仅移除安装前不存在的42个包。已dry-run验证只有这42项UNLINK、无LINK，尚未执行。不要使用早期 `rollback.sh` 的revision恢复：其dry-run会重装Conda记录中的packaging/setuptools，覆盖已有pip版本。恢复后视频依赖将回到上一轮阻塞状态。

## 版本与隔离

- 固定作者源：`Sivan666666/doorgym@096938c44ecc1c7602304d42502f96658e46eb25`。
- 新分支：`migration/wc4-act-joint9`。本地 `/Users/yu/Data/Video2Door/code-wc4-act`；服务器 `/data/zyf/video2door-wc4-act`。
- 旧线：`64b1bcc162c4997dbe39fd3f76b61500573d69a8`。本地 `/Users/yu/Data/Video2Door/code`；服务器 `/data/zyf/video2door`，保持原 HEAD、源码、环境和产物。
- 源码采用稀疏检出，保留作者目录结构。已取齐 `dp`、`float_ik`、定制 LeRobot 源码/打包文件、配置及 A2W/Z1/wc4 资产；未检出无关实机部署输出、测试大文件及第三方源码。Gym 使用旧线保留的项目专用安装。
- 作者算法、skill、URDF、物理配置、转换器、backend/worker 均未修改。新增内容仅为 `migration/wc4_act`（含本说明）、README 入口和 `.repro` 忽略规则。

历史索引（按本地/服务器路径读取，不从新基线自动续跑）：

- 旧线 `code/docs/FLOAT_PUSH_BASELINE_SWITCH_ARCHIVE.md` 及 `docs/evidence/baseline-switch-20261015/archive_summary.json`。
- 服务器完整历史：`/data/zyf/video2door-archives/20261015-baseline-switch-v1/`。
- 本地既有交付：`/Users/yu/Data/Video2Door/reproduction/pro5000/`。
- 本轮证据：本地 `reproduction/wc4-act/preparation-20261015/`；服务器 `/data/zyf/wc4-act-evidence/preparation-20261015/`。

## 已确定路线及来源

机器可读配置为 `migration/wc4_act/config.json`；命令构建器把所有路径映射到新工作区。

| 项目 | 有效设置 | 作者来源 |
|---|---|---|
| 专家 | A2W 浮动底座、ikpush、gym_jacobian | `float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py` |
| 轨迹 | 原样 `wc4_smoke_skill.json` | `float_ik/door_twin/examples/`；`door_twin/doc/wc4_skill_alignment_notes.md` |
| 门与物理 | 只选择 wc4；保留变换、运动符号和 controller multipliers | `data/cfg/b1z1_opendoor.yaml` 中 wc4 条目及 wc4 URDF |
| 俯仰 | 0–0.08726646259971647 rad，环境随机化开启；采集/评测相同 | `float_ik/config/a2w_float_ik_push_door_parallel.yaml` |
| 数据 | 50条成功 NPZ；state/action 均9维，vx/yaw、joint1..6、jointGripper | `dp/record/record_door_dp_dataset_a2w_state10.py --state_action_mode joint_state9` |
| 转换 | 显式直接 `joint_state9`，保持已记录关节数组 | `dp/convert_door_raw_to_lerobot.py`；不用 `tracik_joint_state9` |
| 深度 | front/wrist、640×480、0.2–1.5m；源单通道uint8 NPZ，经源转换器保存视频并按模型输入读取 | `dp/door_dp_common.py`、源转换器 |
| 频率 | 仿真源dt 0.02s；相机与记录/策略更新25Hz | skill文档及 Joint9 采集脚本/评测报告 |
| 相机 | 原投影内参，位置0.02m/旋转5°随机化；噪声和模糊关闭 | `dp/record/record_convert_wc4_joint9_realk500_ps1.sh` 的双深度公共设置；不采用其real-K或Interaction部分 |
| ACT | ResNet18，chunk100，训练n_action_steps50，bs16，lr1e-5，训练seed默认1000 | 作者 ACT variant 训练脚本与 `dp/result/A2W_WC4_200_ACT_EE_JOINT_FIRE_DOOR_EVAL_20260720.md` |
| baseline | 关闭keyframe加权/采样、camera gating、Plücker、Interaction、handle aux、end signal、点云 | 作者 baseline 分支及定制 ACTConfig |
| 闭环 | A2W ikpush、H25、25Hz、日志每步记录；最大绝对门铰链角≥80° | `dp/eval/eval_door_policy_success.py`；上述评测报告 |

`camera_intrinsics_mode=legacy` 是作者对原相机投影的枚举名称，**不表示采用 legacy 专家轨迹**。专家轨迹始终由显式 wc4 skill 驱动，不使用 real-K。

1100步取自作者 skill 对齐说明的调试建议（足够主任务及大部分回撤），是本轮命令准备值，不是已批准的执行预算。50条、Joint9、skill、0–5°俯仰是本轮确定的新组合；不能把作者其他历史配置的成功率当作它的预期结果。

采集仍使用源 recorder 的保存规则：`dp_record_success`、无已记录碰撞、可选接触筛选（本方案不启用）。skill tracker 另报开门、通过、解锁、相机等诊断；不能将 tracker 成功与 NPZ 保存规则混称一个指标。`--no_collision_geom_check` 来自作者 skill 说明，采集/评测一致；不修改 PhysX 或门参数。闭环主指标仅为日志最大绝对开门角≥80°，不验收穿越或停车。

## 环境

仅服务器执行。新 ACT prefix：`/data/zyf/video2door-wc4-act/.repro/envs/door-act`。
从旧训练环境复制后，安装本仓库 `high-level/lerobot` editable；导入路径和依赖导出见证据，不以包名/版本号代替源提交核验。

在新的 Bash 子进程中：

```bash
cd /data/zyf/video2door-wc4-act
source migration/wc4_act/activate.sh act
# Gym/闭环：先按实时资源快照选定空闲卡，再设置 VIDEO2DOOR_GPU 并 source ... gym。
```

Gym prefix 仍为旧线 `.repro/envs/b1z1-blackwell`，复用 UUID 对齐的 CUDA/Vulkan 支持。
激活包装器在旧支持完成后返回新工作区；设置新缓存和 `DOOR_DP_LEROBOT_PYTHON`，不向 Gym 的 Python3.8 注入 Python3.10 的 LeRobot。
原作者 backend 自动转入 `door_policy_worker.py` 子进程。无仿真检查验证了真实 worker 启动、消息反序列化、错误响应和正常关闭；没有加载 checkpoint 或验证预测动作。

两类激活最终均显式设置 `WANDB_MODE=online`，移除 `WANDB_DISABLED` 和离线 HF 设置。实体 `fi-yu-individual`、项目 `door-dp-stop-tail`、group `wc4-act-joint9-skill-baseline-20261015`。
在线准备检查 run：`93mii0ma`，job_type=preparation；已结束，无训练步、媒体或模型 artifact。
实际训练采用新 job_name/run，不复用准备检查 run ID，不 resume 旧实验。
官方 ResNet18 IMAGENET1K_V1 权重已下载到新 ACT 缓存并校验 SHA256；没有构造模型或复用旧任务 checkpoint，原始记录见 `resnet18-weights.log`。

## 阶段命令（本轮只生成、解析，未执行）

`commands.py` 只打印 shell 命令，绝不执行。所有产物位于新工作区 `.repro/runs/<run-id>/`：`raw/`、`lerobot/local/wc4_joint9_skill_<run-id>/`、`train/`、`door-checkpoint/`、`evaluation/`。
每个新实验使用唯一 run-id；禁止将旧目录链接进来。采集拒绝已有raw目录，后续阶段拒绝覆盖已有输出，转换/训练前只读核验数据维度与数量。

以下命令中的GPU、seed、尝试次数、训练步数、评测次数由下一轮任务书明确；生成示例不构成执行授权：

```bash
python migration/wc4_act/commands.py record --run-id RUN_ID --gpu GPU --seed SEED --max-rollouts MAX_BATCHES
python migration/wc4_act/commands.py convert --run-id RUN_ID
python migration/wc4_act/commands.py train --run-id RUN_ID --gpu GPU --seed SEED --train-steps UPDATES
python migration/wc4_act/commands.py export --run-id RUN_ID
python migration/wc4_act/commands.py evaluate --run-id RUN_ID --gpu GPU --seed BASE_SEED --trials TRIALS
```

可加 `--json` 查看准确 argv 和路径。默认seed1000仅是作者训练种子参考；采集/评测应显式给出种子。
执行前重新检查 GPU 利用率、显存、compute-app PID 及本项目锁；不得抢占他人作业。命令设置nice10、两线程，单卡执行；命令生成器本身不承担GPU锁定或超时控制，下一轮受控执行须在持有项目锁并设置批准的超时预算后运行，不直接排队自动启动。

工程验证先于效果验证。后续采集成功率报告和专家视频应先交人工审查；不得仅以环境检查通过而启动后续训练或调整研究变量。

## 无仿真检查与回滚

```bash
python migration/wc4_act/check_static.py
# ACT环境，隐藏GPU：
CUDA_VISIBLE_DEVICES='' python migration/wc4_act/check_act.py
CUDA_VISIBLE_DEVICES='' python migration/wc4_act/check_video.py
# Gym环境，隐藏GPU；外层timeout保护IPC检查：
CUDA_VISIBLE_DEVICES='' timeout 90 python migration/wc4_act/check_gym_bridge.py
```

`check_act.py` 验证定制ACT、转换及导出命令解析；`check_gym_bridge.py` 验证record→专家、evaluate→play→专家的完整参数展开。
`check_video.py` 额外用临时合成灰度帧检查H264编码和作者默认视频解码器，结束后移除临时文件；这不是专家数据采集或LeRobot转换。
Gym检查把 `gymapi.acquire_gym` 替换为抛错函数，禁止创建模拟器；运行资产拆分辅助函数但不加载 PhysX 资产。
worker probe 故意发送未知命令，预期返回结构化错误，再发送close取得ok响应；该错误是协议测试预期值。

回滚只需停止使用新工作区并回到旧工作区的 `64b1bcc`；旧环境与完整归档未移动。不要修改远程main。新工作区及环境本轮保留供审查，不永久删除任何资产。
稀疏检出规则在各工作区 `.git/info/sparse-checkout`，证据中保留副本；扩大源码范围应从同一固定提交补齐。

限制：无真实数据/模型，不能验证50条成功采集、转换产物、训练数值稳定性、完整模型桥接、仿真资产加载或策略成功率。环境和命令检查不构成这些项目的验收结果。
