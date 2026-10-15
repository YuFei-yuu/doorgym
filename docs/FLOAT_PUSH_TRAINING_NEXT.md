# 新对话交接：浮动底座推门 DP

更新：2026-10-15。最新学习率配对续训任务已完成：C/L各从同一30k完整状态新增10000更新，6个完整闭环后停止；未自主停车。后续新实验未授权。以下各条按新到旧保留历史状态。

## 当前状态

- **学习率配对完整续训已完成并停止**：C=1e-4、L=1e-5，同一30k模型/AdamW/采样器/RNG和固定统计，指定B单回合1446帧/1415窗口，batch8/workers0，各新增10000、累计40000；0/+2k/+5k/+10k恢复/采样器/RNG配对通过，9项CPU受控恢复测试通过。全部6回合2500步/50秒，开门/原通过4/6，停车0/4可评、联合0/6。40k C38.92s通过后仍走1.355m，L22.68s通过后仍走2.262m；真实保持观察充分、无停稳候选。前16步总体vx两组略降至约0.168，但保持vx由30k0.151恶化到C0.183/L0.175，32步vx均恶化；L位置条件预测较C好，底座路径和完整自身跟踪却更差，不能判L稳定占优或完整模仿。W&B [C/sk6n50eh](https://wandb.ai/fi-yu-individual/door-dp-stop-tail/runs/sk6n50eh)、[L/49viem9o](https://wandb.ai/fi-yu-individual/door-dp-stop-tail/runs/49viem9o)各1000原始loss记录云端回读逐项一致、同步异常0。准备NPZ读取效率修复及3次只读SSH断连留痕，无重复成功训练/仿真，无核心/执行器变更、重采/扩维/留出测试/追加或提交推送；见[完整交付](FLOAT_PUSH_LR_PAIR.md)。底座条件信息、监督、训练目标及闭环覆盖的因果作用尚未区分，下一步留给方案审查，不自动开展。

- **单回合B、随机初始化、batch8、30k完整任务已完成并停止**：指定1446帧/1415窗口、新训练数据统计，未加载旧权重/状态/统计；实际随机0与审计0权重/统计/空优化器/采样器/全部RNG一致，6节点完整状态续训，130参数优化器均30000步，全部6次2500步闭环，执行完整性通过。开门5/6、原通过2/6、停车0/2可评、联合0/6（6个不同模型节点，不是最终模型重复成功率）。30k开门82.56686°，46.62s原通过，后3.36s实测速度始终0.03019–0.11709m/s、额外路径0.22374m，无合格停稳。固定前16步位置RMSE从1k的0.50247降到30k的0.09487m，但转换/保持vx仍0.30322/0.15098m/s；真实时间实际末端轨迹RMSE0.83654m、通过晚20.72s，完整自身姿态跟踪101.19°，不能以loss/过门声称完整模仿。W&B [hn4uaqxa](https://wandb.ai/fi-yu-individual/door-dp-stop-tail/runs/hn4uaqxa)全部3000条native指标回读一致并结束；21项CPU回归通过，核心未变、原证据保持，未追加/使用留出测试/提交推送。见[完整交付](FLOAT_PUSH_SMALLBATCH30K.md)。

- **batch8任务准备/1k历史记录**：独立run真实准备、训练数据统计、随机零更新与RNG比较及新W&B开训前链接交付均通过；1k普通失败后继续既定30k，不按时间估计提前结束。最终结果见首条及[配置与完整记录](FLOAT_PUSH_SMALLBATCH30K.md)。

- **16040观测来源A/B分支v3已完成，两组各新增2000后停止**：同一5000权重/固定统计、优化器/采样器/RNG重置；真实A/B输入、全部窗口和各节点配对通过。1000两组均未达80°；2000两组开门/原通过，但停车及联合均失败（总4回合联合0/4，全部2500步）。A通过后仍走9.3055m；B49.58s才通过，只余0.40s且末vx0.0812，不能把0.0412m短路径当作停稳改善。B较A改善最终若干末端跟踪/速度/路径指标，仍有明显绝对误差与23.68s通过延迟；A任务事件通过但完整自身跟踪RMSE5.8386m。完整10D交叉诊断显示位置部分改善、vx/停车转换仍差，不能用loss或过门认定模仿合格。W&B两组云端2000/离线/闭环回读通过，15项CPU回归通过，所有进程退出，原证据保留；计算后两项纯分析修订留痕，无重复训练/推理/仿真。见[完整交付](FLOAT_PUSH_OBSERVATION_BRANCH16040.md)。未追加、未用留出测试、未提交推送，最终验收交方案对话。

- **v3准备阶段历史记录**：真实A/B输入、原转换器、两组全部1446帧/1415实际窗口、实际零更新初始化及Python/NumPy/Torch/CUDA RNG配对检查均通过。源5000模型仅提供权重及固定统计，两组重新初始化优化器/采样器/RNG，不能称作原5000完整续训。用户授权可恢复准备/后处理工程错误最小修复后继续，成功训练/仿真不重复；v1/v2失败证据保持。最终结果见本节首条。

- **16040分支v2在raw字段审计异常后停止，仍未开训**：用户已授权v1路径修正；独立`20261015-observation-branch16040-v2`错误要求raw含后端独有next_state，抛出KeyError。只读确认raw共有字段与trajectory完全一致、trajectory/执行日志next_state完全一致。代码已作最小schema审计修正，新增3项CPU负例回归通过，未自动重启。没有新模型、W&B、仿真或留出测试，见[分支失败记录](FLOAT_PUSH_OBSERVATION_BRANCH16040.md)。v1与v2证据均保留，未提交推送。

- **16040观测来源A/B分支在输入准备异常后停止，未开训**：`experiments/float-push/20261015-observation-branch16040-v1`误把精简`pilot/reference`当作含raw的完整输入，抛出FileNotFoundError。4项CPU分支入口测试通过；没有训练、仿真、W&B或留出测试。只读确认完整`reference/seed_16040`与精简副本关键记录SHA256相同，代码已作最小路径修正，但未自动重试，真实初始化与窗口门槛仍未完成。见[失败与待审查修正](FLOAT_PUSH_OBSERVATION_BRANCH16040.md)。重启需另行确认，未提交或推送。

- **停车小数据完整续训5000已完成，闭环仍未停稳**：新run `experiments/float-push/20261014-stop-tail-resume5000-v1`从旧1000模型完整恢复模型/优化器/采样游标/RNG，新增4000。原3回合4321帧/4228窗口、单卡batch64/workers0、lr1e-4、forward_v3和16/32/16均未改。2000/3000开门/通过均0/3、停车不可评估；5000开门3/3、通过2/3、停车0/2、联合0/3。9回合完整2500步/视频，无执行异常、专家回退或停车覆盖。5000通过后仍走2.928/1.974m；专家保持vx RMSE由0.283降至0.073，但转换vx约0.259未改善。尚无充分依据直接扩大数据。W&B在线同run已验证到5000：[在线监控](https://wandb.ai/fi-yu-individual/door-dp-stop-tail/runs/87d17369)，未上传模型或原始数据。用户授权源码修订2仅规范化DDPM声明列表顺序比较及修复SDK summary缓存回读，全部旧快照保留，训练未中断。最终模型SHA256 `2c0b9634465c37d15ff9d461d79f8e5268b1956076d1ec72ba1792b887338026`。详见[续训完整交付](FLOAT_PUSH_STOP_TAIL_RESUME5000.md)；未超5000、未用留出测试、未提交推送，不得自动继续。
- **停车回放→三回合尾段→1000更新小训练已完成；模型停车仍失败**：A/B各一次1446步单变量回放通过（统一10D四元数处理，仅冻结恢复不同）；只补采16041/16042，复用16040，三回合4321帧/4228个实际窗口通过。随机seed42、单卡batch64/workers0训练严格1000更新，250/500/1000开门分别3/3、2/3、3/3，通过0/3、2/3、3/3，停车和联合均0/3。全部9回合完整2500步，无执行异常/脚本回退/无效图像/停车覆盖，视频解码通过。最终三个通过回合仍走5.796/0.690/1.387m；不是剩余队列或记录不足。模型SHA256 `b7c49cce0c350f743bc08fe01a6732830951bf1eb765c6f2d188cc04dd3f79ec`。进程退出/GPU释放，未训练1000之后或用留出测试；全部新旧证据保留。详见[本轮完整交付](FLOAT_PUSH_STOP_TAIL_PILOT.md)，不得自动继续、扩大或恢复旧run，未提交推送。
- **16040单回合离线审计已通过，三回合任务仍未完成**：`experiments/float-push/20261014-stop-tail-offline-16040-v1`只读使用v2/16040，19项含动作/视频负例的回归检查通过。原转换器输出1446帧，实际forward_v3加载器逐帧及逐个读取全部1415个窗口核对通过，保持观测覆盖2.36秒，停车标签全32步4336个/前16步2040个。原停稳阈值未改；v1/v2失败结论及输入哈希保持不变。停车动作前16帧四路图像全零，与155个行进历史图像相同；已有角速度/伺服微小变化仅是候选信号，未证明提前可辨识性，冻结差异的动态影响仍未验证。见[单回合离线交付](FLOAT_PUSH_STOP_TAIL_OFFLINE_16040.md)。本轮止于交付，未补采、仿真、训练、留出测试或提交推送。
- **停车尾段v2首回合审计失败后停止，未完成3回合数据验收**：`experiments/float-push/20261014-forward-stop-tail3s-v2`已修复目录哈希及初始化失败落盘，24项针对性测试通过。16040采集1446帧（含首次通过后150步），正确方向86.17093°、原目标距离0.112451m、通过后实测平移/漂移/额外路径均0。新增审计误把原5位小数日志与float32 raw要求逐位相等，检查失败后立即停止；16041/16042未尝试，未转换或枚举窗口、未训练。只读诊断确认按原日志序列化规则全部一致，旧专家前1296帧逐位一致。另发现末16帧行进与停车尾段四路策略图像全零、冻结手臂目标差异最大0.0311173rad，须独立审查。详见[v2失败交付](FLOAT_PUSH_STOP_TAIL_V2.md)，不得自动修改判定、继续采集或训练。
- **3秒停车尾段数据任务在初始化失败后停止，尚未采集**：`experiments/float-push/20261014-forward-stop-tail3s-v1`将旧`user_stop_3000`目录误按文件计算SHA256，抛出`IsADirectoryError`。本轮仅有未提交的记录时机/验收代码和失败证据，测试、仿真、采集、转换、窗口验收均未开始，未训练、未使用留出测试。不得把该run当作可训练数据或自动重试。错误原因及下一步待授权调整见[尾段任务失败交付](FLOAT_PUSH_STOP_TAIL_DATA.md)。
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
