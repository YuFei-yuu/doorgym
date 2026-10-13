# 新对话交接：浮动底座推门 DP

更新：2026-10-13。当前采用用户已确认的“开门后直接向前穿越”方案。

## 当前状态

- 专家已在9组位置、18个预定回合中全部达到正向80°并完成通过，80°后反向行程均为0。
- 当前 forward 方案还没有训练过DP。保留的5000更新模型来自旧版通过流程，6个完整评估回合通过率为0/6，只作为参考。
- 下一步是用当前 forward 专家建立新的训练/验证数据集，再训练与评估DP。采集规模、集合划分和更新数由下一轮实验确定。

## 下一轮训练要求

1. 使用当前已确认的浮动底座forward专家，重新采集演示，从随机初始化开始训练DP。其余环节尽可能直接复用固定源仓库的网络、数据处理、10D动作、IK/伺服、训练及推理方案。最终目标是模型自主推门达到正向80°并直接向前穿越。
2. 训练记录和图表参照源仓库实际关注的指标：更新数、loss、学习率、耗时，以及闭环开门/通过成功率、门角、底座路径和动作执行。记录数据划分、配置、checkpoint及视频，具体字段见[指标说明](FLOAT_PUSH_METRICS.md)。
3. 不确定的问题先查阅源码和现有记录；仍不确定的实现、参数或实验选择，向用户说明依据和选项，询问后再决定。已确认的方案继续沿用。
4. 时间预算用于预估与资源安排，不作为正常训练的硬截止，也不继承旧实验的固定预算。预计耗时有明显变化时更新说明。
5. 训练前汇报拟采集回合/帧数、初态覆盖、训练/验证/测试划分、batch、更新计划、预计耗时及首次视频检查节点，并与用户确认。训练中汇报实际规模与进度；达到检查节点后冻结阶段checkpoint，录制模型完整回合及同初态专家对比视频，报告任务成功率，供用户人工检查并商定下一阶段。

## 工作位置

| 项目 | 位置 |
|---|---|
| 本地仓库 | /Users/yu/Data/Video2Door/code |
| 服务器 | ssh pro5000 |
| 服务器仓库 | /data/zyf/video2door |
| 关联Git仓库 | YuFei-yuu/doorgym |
| 固定源提交 | Sivan666666/doorgym@0acc2f050479fb54a0670ccb90bc72b98b52a54a |
| 当前任务入口 | high-level/float_ik/learning/wholebody_float.py |
| Gym环境 | .repro/envs/b1z1-blackwell |
| 训练环境 | .repro/envs/door-dp-train |

源码和资产在仓库内；环境只在服务器项目目录中。环境用法见[环境配置](环境配置.md)。

## 运行一个专家回合

在服务器 Bash 中使用一个新的输出目录：

```bash
cd /data/zyf/video2door
task_run=experiments/float-push/new-forward-expert
.repro/envs/b1z1-blackwell/bin/python high-level/float_ik/learning/run_tools/launch.py \
  --root "$task_run" --stage expert --timeout 600 --total-seconds 600 -- \
  python high-level/float_ik/learning/wholebody_float.py \
  --headless --no_show_seg --camera_seg --camera_depth \
  --float_execution source --float_pass_mode forward \
  --robot_x 4.1 --robot_y 0 --robot_z 0.50 --stop_distance 0.25 \
  --push_base_vx 0.25 --lever_step_size 0.06 \
  --steps 2500 --float_seed 15000 --float_video \
  --float_output "$task_run/expert"
```

输出包含 summary.json、trajectory.npz、execution.jsonl、raw/、源码快照和 rollout.mp4。任务成功看正向80°及 pass_done。

## 训练入口

| 工作 | 代码 |
|---|---|
| 专家采集 | wholebody_float.py；collect_v3.py提供批量模板 |
| raw转LeRobot | high-level/dp/convert_door_raw_to_lerobot.py |
| 原DP训练循环 | high-level/dp/train_door_dp.py |
| 数据与checkpoint检查 | learning/checks.py、dataset_contract.py |
| DP闭环执行 | wholebody_float.py --dp_policy_checkpoint <模型路径> |
| 离线分析与绘图 | learning/offline_eval.py、analyze_v3.py |

collect_v3.py、finish_v3.py、resume_eval_v3.py保留了旧实验的种子和编排，作为参考；新实验需配置新的集合与目录。训练沿用原LeRobot逐帧读取，不启用帧缓存。

## 数据与证据

- 当前18回合：服务器 experiments/float-push/20261013-forward-expert-grid18/；本地 reproduction/pro5000/forward-expert-grid18/。完整轨迹、raw、相机数据、源码及视频均保留。14500–14517已用于验证。
- 同初态前后对比：服务器 experiments/float-push/20261013-forward-pass-review/；本地 reproduction/pro5000/forward-pass-review-20261013/。
- 旧5000更新参考：服务器 experiments/float-push/20261013-upstream-float-v3/；本地 reproduction/pro5000/float-push-v3/。保留原始训练/验证数据、最终模型和完整闭环结果；旧LeRobot转换目录已清理，可从raw重建。

需要实现细节时读[当前方案](FLOAT_PUSH_REPRODUCTION_PLAN.md)；需要指标时读[指标说明](FLOAT_PUSH_METRICS.md)。专家结果见[18回合验证](FLOAT_PUSH_FORWARD_EXPERT_VALIDATION.md)。
