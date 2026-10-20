# wc4 ACT 50k基线终点交付报告

本轮从已验收40k磁盘checkpoint恢复，新增恰好10000次更新，严格结束于50k；只评测最终50k一次64回合。未补评30k、重跑已有模型或追加分析仿真。旧数据、模型与证据均未覆盖。

## 结果与最后10k收益

| 模型 | 成功/64 | 成功率 |
|---|---:|---:|
| 36k（旧记录） | 52/64 | 81.25% |
| 40k（旧记录） | 46/64 | 71.875% |
| 50k（本轮唯一新评测） | 48/64 | 75% |

| 对比 | 共同成功 | 共同失败 | 改善：旧失败→50k成功 | 退步：旧成功→50k失败 | 成功数净变化 |
|---|---:|---:|---:|---:|---:|
| 36k→50k | 45 | 9 | 3 | 7 | -4 |
| 40k→50k | 43 | 13 | 5 | 3 | +2 |

相对40k净增加2/64成功（+3.125个百分点），是在固定集上的小幅观察收益；同时仍有3个原40k成功回合退步。相对36k少4/64（−6.25个百分点），没有超过既有较高观察值。训练loss下降不等于已获得稳定或独立泛化收益。

每行四类之和均为64。详细trial清单及逐回合门角见[配对比较JSON](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/paired-comparison.json)、[配对比较CSV](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/paired-comparison.csv)和[50k全部64条结果](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/per-trial-results.csv)。工程链通过不代表策略达到稳定成功；收益应按上述配对结果判断，不能仅以训练loss下降认定。

旧36k→40k为共同成功44、共同失败10、改善2、退步8（净−6）；“7个持续失败”是32/34/36/38/40k五模型失败交集，不等于36k/40k的10个共同失败。所有比较使用固定四批seed615455575..615455578、每批16环境，仍是反复使用的固定评测集，不是独立泛化或未使用初态确认。后者尚未执行。

## 恢复、训练、环境及版本

40k来源：`/data/zyf/video2door-wc4-act/.repro/runs/wc4-act-resume30k-40k-s1000-20261015/train/checkpoints/040000`。原模型SHA256为`88fa37325c84e070d8f122bf098b006896fe75ec38e9cba3883edf202be3bcf5`。

恢复前模型逐张量相等；153个optimizer step均为40000。Python/NumPy/Torch/CUDA共8项序列化RNG张量与保存值相等；预处理/后处理各110个统计张量在首个新增更新前核对相等。按作者恢复流程重建DataLoader，不声称与不中断训练逐位一致。

新更新严格为40001..50000，共10000条。loss和梯度均有限，lr固定1e-5；前1000次新增更新均值0.02919770，最后1000次0.02629858。W&B有100个唯一进度点（40100..50000），原始110行保留，重复项要求值完全一致；对应本地均值最大差异0.0。

恢复节点42k/44k/46k/48k/50k均保存模型、optimizer、RNG、统计；optimizer步数与各节点一致。中间节点没有导出评测，实际闭环checkpoint列表仅[50000]。完整模型/状态哈希见[验收JSON](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/acceptance.json)。

50k模型SHA256：`3f1f32f96eb3a78abd215f6affa6786d7b2cf76b94def6b8268bdeefe3bfbdb9`。服务器官方checkpoint：`/data/zyf/video2door-wc4-act/.repro/runs/wc4-act-resume40k-50k-s1000-20261015/train/checkpoints/050000`；Door manifest：`/data/zyf/video2door-wc4-act/.repro/runs/wc4-act-resume40k-50k-s1000-20261015/checkpoints/050000/door-checkpoint/model_latest.pt`。

两端源码仍为`5ca33bf00bf7f0bd4e0484db8cbeb48c84cb7f7f`，原作者源固定`096938c44ecc1c7602304d42502f96658e46eb25`；未改作者核心、skill、物理或算法，也无新Git提交。运行编排与检查在`tools/`，差异见`runtime-adaptation.diff`。ACT环境为`/data/zyf/video2door-wc4-act/.repro/envs/door-act`，实际为仓库定制LeRobot；Gym复用`/data/zyf/video2door/.repro/envs/b1z1-blackwell`和Blackwell支持。无依赖/系统/旧环境改动。

保持原50条/27500帧数据、seed1000、batch16、lr1e-5、ResNet18预训练方式、chunk100、训练n_action_steps50和全部预处理。数据8文件与40k原checkpoint文件哈希均未变；有效cfg差异只允许恢复路径、训练终点、独立目录/job/W&B ID等，不改研究配置。精确差异见[training-config-diff.json](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/training-config-diff.json)。

## 导出、实际推理与原始录像

官方/导出权重与统计相等；真实数据窗口的预处理及物理动作对照、实际Gym→ACT GPU子进程对CPU参考均通过原检查阈值，未放宽断言。见`checkpoints/050000/export-consistency.json`和`gpu-bridge-check.json`。Joint9状态/动作语义保持作者定义，无专家回退；全部64回合为1100步`dp_policy`记录。仿真dt=0.02s，日志每仿真步记录（50Hz）；策略/视频25Hz。解锁步数是日志step，不是视频帧号。

继续wc4/A2W/作者wc4_smoke_skill/gym_jacobian/直接Joint9，0–5°俯仰与相机/物理随机化冻结；策略25Hz、预测100、执行25。主指标仍为最大绝对门铰链角≥80°，穿越/停稳不加入成功判据。128组与36k/40k对应seed/env的初始状态、随机化和双相机位姿配对检查通过，见[初态配对检查](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/paired-initial-state-check.json)。

对候选批次同步临时录制，在这64次内选取最早成功/失败各一段，两类齐备后关闭后续批次录像；仅移除本轮未选中的临时clip，保留其SHA及receipt和所有原始日志。不重跑、不用状态回放补原片。无缺类，成功与失败均存在。

- [成功原片](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/checkpoints/050000/videos/success/rollout.mp4)：seed=615455575，env=0，trial=0，最大门角=90.0002°；[模型/seed/帧数绑定](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/checkpoints/050000/videos/success/binding.json)。
- [失败原片](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/checkpoints/050000/videos/failure/rollout.mp4)：seed=615455575，env=3，trial=3，最大门角=0.0286°；[模型/seed/帧数绑定](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/checkpoints/050000/videos/failure/binding.json)。

所选原片均1920×640、550帧、25fps、22秒，未经剪辑、变速或合成。所有已保留选片的SHA在复制后复核。全部64次仍有逐步日志和逐回合结果，录像分母不影响成功率。

## 已有日志的失败阶段分析

分析在50k评测前固定了17个目标初态，原始32k–40k的320个回合日志均校验源SHA，分析产物在`analysis-existing/`单独封存，没有新增推理或仿真：

- 7个持续失败：3、24、35、43、52、57、61。
- 8个36k→40k退步：7、13、17、30、33、41、47、50。
- 2个36k→40k改善：26、27。

阶段标签以作者控制器记录的`replay_door_open_stage`为依据：未见解锁；或已见解锁但最大门角未达80°。成功仍只依据80°主判据，不额外要求解锁标签。所有实际控制phase都为dp_policy，不能当成专家skill阶段名。

旧日志事实：32k/34k/36k/38k/40k分别有13/17/12/16/18个失败未见解锁，另有1/1/0/1/0个已解锁未达80°。7持续失败在五个checkpoint中均未见解锁。8个退步在36k已记录解锁并成功，在40k均有正接触记录但未见解锁；2个改善方向相反。40k的18个失败全部存在正接触记录，不能简化为“没碰到把手”。

8个退步回合的七关节目标—实测RMSE在40k均高于36k（仅为550个新鲜25Hz状态样本的原始角度差统计）。这是同一采样时刻的新目标与当前实测之差，未校正执行延迟，不是稳态跟踪误差；也是伴随不同动作/接触轨迹的观察，不是因果分解；不能仅凭它判断底层控制故障。

7个原持续失败在50k仍全部失败；8个原退步回合恢复3个（7、17、50）；2个原改善只保持26，27再次失败。50k全部16个失败均未见解锁。

50k全部64回合的阶段统计：

| 观察结果 | 回合数 |
|---|---:|
| 达到80° | 48 |
| 失败且未见解锁 | 16 |
| 失败但已见解锁 | 0 |

预定义清单的50k结果：

| 旧清单 | 50k成功/该组数量 | 50k失败且未见解锁 | 50k已解锁但未达80° |
|---|---:|---:|---:|
| 原持续失败7个 | 0/7 | 7 | 0 |
| 原退步8个 | 3/8 | 5 | 0 |
| 原改善2个 | 1/2 | 1 | 0 |

以下只是在预先固定清单后追加50k结果，没有根据50k结果重选目标。

| 分组 | trial | seed/env | 36k最大门角°/阶段 | 40k最大门角°/阶段 | 50k最大门角°/阶段 | 50k正接触记录行数 |
|---|---:|---|---|---|---|---:|
| 原持续失败7个 | 3 | 615455575/3 | 0.031 / 未见解锁 | 0.026 / 未见解锁 | 0.029 / 未见解锁 | 719 |
| 原持续失败7个 | 24 | 615455576/8 | 1.794 / 未见解锁 | 1.969 / 未见解锁 | 3.042 / 未见解锁 | 370 |
| 原持续失败7个 | 35 | 615455577/3 | 2.006 / 未见解锁 | 0.338 / 未见解锁 | 0.385 / 未见解锁 | 622 |
| 原持续失败7个 | 43 | 615455577/11 | 1.784 / 未见解锁 | 0.215 / 未见解锁 | 1.893 / 未见解锁 | 361 |
| 原持续失败7个 | 52 | 615455578/4 | 1.800 / 未见解锁 | 1.717 / 未见解锁 | 2.190 / 未见解锁 | 347 |
| 原持续失败7个 | 57 | 615455578/9 | 2.390 / 未见解锁 | 2.037 / 未见解锁 | 2.282 / 未见解锁 | 407 |
| 原持续失败7个 | 61 | 615455578/13 | 1.697 / 未见解锁 | 1.687 / 未见解锁 | 1.776 / 未见解锁 | 505 |
| 原退步8个 | 7 | 615455575/7 | 90.000 / 成功 | 0.033 / 未见解锁 | 90.000 / 成功 | 497 |
| 原退步8个 | 13 | 615455575/13 | 90.000 / 成功 | 1.438 / 未见解锁 | 0.379 / 未见解锁 | 232 |
| 原退步8个 | 17 | 615455576/1 | 90.000 / 成功 | 0.065 / 未见解锁 | 90.000 / 成功 | 564 |
| 原退步8个 | 30 | 615455576/14 | 90.000 / 成功 | 0.141 / 未见解锁 | 0.036 / 未见解锁 | 708 |
| 原退步8个 | 33 | 615455577/1 | 90.000 / 成功 | 1.819 / 未见解锁 | 1.800 / 未见解锁 | 397 |
| 原退步8个 | 41 | 615455577/9 | 90.000 / 成功 | 1.183 / 未见解锁 | 1.762 / 未见解锁 | 534 |
| 原退步8个 | 47 | 615455577/15 | 90.000 / 成功 | 1.742 / 未见解锁 | 1.890 / 未见解锁 | 327 |
| 原退步8个 | 50 | 615455578/2 | 90.000 / 成功 | 1.952 / 未见解锁 | 90.000 / 成功 | 397 |
| 原改善2个 | 26 | 615455576/10 | 0.000 / 未见解锁 | 90.000 / 成功 | 90.000 / 成功 | 268 |
| 原改善2个 | 27 | 615455576/11 | 0.016 / 未见解锁 | 90.000 / 成功 | 0.015 / 未见解锁 | 596 |

观察与假设的边界：

1. 正接触只证明日志检测到配置指体与把手body的接触，不证明稳定抓握或有效施力；零接触可能是getter在缺body handle或API异常时返回零，日志无有效性标志，不能据零断言完全没接触。score还可能是接触数代理，不当作校准力值。
2. Joint9日志的EE target由当前EE观测填充，`ee.pos_error`不是独立的把手目标距离，本轮不据此判断接近/抓取误差。底座位移是命令轨迹，不证明无碰撞穿越。
3. “正接触后未解锁”可定位观察阶段，但根因可能涉及抓握姿态、旋转目标/时序、接触维持或视觉感知；现有日志没有逐步深度图、有效接触健康标志和独立任务目标距离，不能在这些假设中作因果裁决。关节跟踪偏差可能是动作/接触的结果，而不是起因。
4. 这些17个回合不一定都保留原始RGB选片；本轮用完整日志/曲线分析，未为补画面重跑或重建状态仿真。

原始观测、关节统计、随机化参数和曲线见[旧日志85个重点回合统计](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/analysis-existing/focus-metrics.csv)、[旧日志全320回合](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/analysis-existing/metrics-all320.json)、[17个回合三模型追加对比](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/focus-36k-40k-50k.json)和[50k全部阶段统计](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/failure-stage-50k.json)。阶段分析不是新的验收标准，未修改skill/算法/数据/物理。

## 命令、在线记录、异常及停止状态

本轮实际服务器R=`/data/zyf/video2door-wc4-act/.repro/runs/wc4-act-resume40k-50k-s1000-20261015`；所有计算在pro5000项目环境。入口记录如下，不应为补证据再次执行训练/仿真；完整argv见`train-command.json`与`checkpoints/050000/eval-batches.json`。

```text
source migration/wc4_act/activate.sh act
R/tools/launch_train.py                 40k→50k，GPU1互斥租约
R/tools/analyze_existing.py             只读旧日志分析，无模型/仿真导入
source migration/wc4_act/activate.sh gym
R/tools/run_final_evaluation.py         等待50k，仅四批×16，GPU2互斥租约
R/tools/accept_results.py               64次结束后核验、配对统计与封存
```

训练：[W&B xobd7ef0](https://wandb.ai/fi-yu-individual/door-dp-stop-tail/runs/xobd7ef0)；评测：[W&B lj85duyv](https://wandb.ai/fi-yu-individual/door-dp-stop-tail/runs/lj85duyv)，均在线同步。[40k–50k训练曲线](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/training-40k-50k.png)。

原始日志、失败现场及warning均保留。沿用作者metadata中未使用的旧EE10顶层标签，但实际policy_config/sidecar/控制器及每步日志均为Joint9；不为标签歧义改写原权重。作者原有旧10维失败启发式不作为本轮根因结论。其余本轮异常/修复如有，见异常记录，不通过改原证据、放宽断言或追加回合通过检查。

模型与优化器终点、128组初态配对及在线记录均已核对，19次训练API监控和18次评测API监控无失联记录；所有项目GPU计算进程已退出。最终保存50k模型与优化器，并停止所有训练和评测；不自动延长，不先行确定候选模型或运行未使用初态测试。下一步由方案对话审查配对收益、预定义失败回合和原始视频后，再确定候选与独立确认评测任务。

交付索引：

- [17个预定义回合的36/40/50k阶段对照](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/FOCUS_CASES.csv)
- [4份完整原始日志及gzip对应](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/RAW_LOG_INDEX.csv)
- [恢复节点模型SHA256](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/model-hashes.csv)
- [实际定制LeRobot与环境版本](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/environment-receipt.json)
- [最终源码、进程及监控状态](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/final-state-check.json)
- [云端64次最终进度及成功数](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/wandb-evaluation-final-check.json)
- [服务器证据封存清单](/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/evidence-manifest.json)
