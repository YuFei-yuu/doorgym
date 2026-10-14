# 新对话交接：浮动底座推门 DP

更新：2026-10-14。当前采用用户已确认的“开门后直接向前穿越”方案。最新停车分析与上游差异见[停车审查](FLOAT_PUSH_STOP_REVIEW.md)；源main仍为固定提交0acc2f0，未发现额外DP终止实现。下一轮采集/训练尚未授权。

## 当前状态

- **小训练停车问题已完成只读分析**：1000更新两个通过回合此后仍走8.8256m/8.3593m。专家每回合只有最后1帧停车，现有32步未来窗口中只出现在offset31，实际执行的前16步停车标签为0；DP还覆盖了脚本停车命令并继续运行50秒。优先修复终止语义和真实停车尾段/窗口覆盖，不能归因于单纯训练不足。本轮无新增训练或仿真，详见[独立分析与下一步提案](/Users/yu/Data/Video2Door/reproduction/pro5000/small3-stop-analysis-v1/REPORT.md)。系统终止停车与DP自主停车的选择及后续采集/训练仍待授权。
- **小数据训练验证已完成，未扩大训练**：仅已有3个标称专家回合16040–16042，forward_v3随机初始化，batch64、workers=0，严格停在1000更新；250/500/1000各3个完整闭环回合，自主联合分别0/3、0/3、2/3。最终失败回合seed16041最大79.698458°。run为`20261014-forward-v3-small3`，最终模型SHA256为`ea20f964c96099115c3ee326475210eae5f4f48a8ac3f1601d8116575b2c0f06`。训练/仿真已退出、GPU释放，全部模型/视频保留；仅训练初态验证，未使用最终留出测试。见[本轮结果](FLOAT_PUSH_SMALL_FORWARD_V3.md)。
- **前期审查意见已落实**：4项工程修复完成，新增`forward_v3`共享数据契约，统一min/max、起始历史补齐和四元数连续符号。训练环境23项/Gym环境6项测试通过，真实数据验收通过，旧模型输出逐位一致，旧证据保持不变。见[修复与验收](FLOAT_PUSH_REVIEW_FIXES.md)。小数据检查随后已获授权完成，后续扩大训练仍需确认。
- **2026-10-13无训练诊断**：`experiments/float-push/20261013-forward-dp-diagnostics-v1`。三种专家回放各3次：恢复冻结标志3/3通过、纯10D动作3/3通过、受裁剪动作0/3通过；405个离线采样窗口和2025个固定噪声窗口完成。动作范围冲突及起步/穿越拟合问题已确认，纯10D动作在所测3个初态可完成任务。详见[诊断记录](FLOAT_PUSH_DIAGNOSTICS.md)及本地诊断索引。首轮min/max v2候选仅作静态验证，当前v3接入见上述修复记录；原run与监控保持停止/暂停。
- **本轮已按用户要求在3000更新结束。** 状态 `stopped_by_user`，训练、仿真与监督进程已退出，GPU已释放；未训练4000/5000更新或运行留出测试。不得绕过 `blocked_by_user` 标记自动恢复。
- 3000更新的9回合验证：80°开门0/9、通过0/9、自主联合0/9，标称0/1，最大门角40.9926°；9回合均完整运行2500步，无专家回退/无效图像/进程错误。最终任务目标未通过验收。完整视频、同初态专家对比、模型与图表见本地 `/Users/yu/Data/Video2Door/reproduction/pro5000/forward-dp-v1/analysis/INDEX_3000.md`。
- 本地448个交付文件SHA256全部与服务器一致，9个完整DP视频和专家对比视频均完整解码通过；`pro5000-forward-dp`监控已暂停。最新模型SHA256为 `0d7dbe50eb5468d9f4ac9877d78ead681ec49bb39a904c5efa43ac62b4e0a0a6`。
- 本轮实现与运行：`experiments/float-push/20261013-forward-dp-v1/`，执行方案与入口见 [本轮训练记录](FLOAT_PUSH_FORWARD_DP_TRAINING.md)。108/108专家采集成功，训练90回合116340帧、验证18回合23269帧；转换及数据检查通过，已进入从随机初始化的正式训练，第一checkpoint为1000更新。实际进度以该run的 `pipeline_state.json`、`manifest.json`、`training/progress.json` 为准。
- 1000更新验证：正确方向80°开门0/9、通过0/9、自主联合成功0/9，标称0/1；全部9回合无进程错误/专家回退/无效图像，最大门角30.1567°。checkpoint检查通过，已可靠恢复训练至2000更新。
- 2000更新验证：80°开门0/9、通过0/9、自主联合0/9，标称0/1，最大门角50.4804°；checkpoint检查通过。
- 专家已在9组位置、18个预定回合中全部达到正向80°并完成通过，80°后反向行程均为0。
- 本轮forward DP从随机初始化训练；保留的旧5000更新模型来自旧版通过流程，6个完整评估回合通过率为0/6，只作为参考。
- 本轮使用90训练＋18验证演示，每1000更新评测；用户要求在3000更新提前结束，原5000视频检查及最终18回合网格测试/10回合标称验收未执行。

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
