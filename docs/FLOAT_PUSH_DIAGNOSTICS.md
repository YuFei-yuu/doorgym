# Forward DP无训练诊断

2026-10-14：已完成审查修复及v3共享契约接入，见[修复与验收](FLOAT_PUSH_REVIEW_FIXES.md)。下文保留2026-10-13首轮诊断历史与原交付入口。

更新：2026-10-13。用户批准执行无训练诊断，正式训练仍需再次确认。

服务器run：`experiments/float-push/20261013-forward-dp-diagnostics-v1`。本地交付见[诊断索引](/Users/yu/Data/Video2Door/reproduction/pro5000/forward-dp-diagnostics-v1/analysis/INDEX.md)、[结论](/Users/yu/Data/Video2Door/reproduction/pro5000/forward-dp-diagnostics-v1/analysis/CONCLUSIONS.md)、[后续提案](/Users/yu/Data/Video2Door/reproduction/pro5000/forward-dp-diagnostics-v1/analysis/TRAINING_PROPOSAL.md)。

## 结果

- 原专家动作＋冻结标志回放3/3成功，记录与原专家逐帧一致。
- 纯10D动作回放3/3成功，底座路径一致；手臂轨迹和snap时刻不同。保留原自然snap机制，因此不是脱离任务事件的动作充分性证明。
- 受原z-score＋±1裁剪限制的专家动作3/3开门≥80°、0/3通过；只运行原专家记录长度，不据此断言更长回放必然失败。
- 405个采样窗口、2025个固定噪声窗口及810个历史消融窗口全部完成。3000模型起步/穿越末端误差约1.00/0.96m，抓取约0.008m，推门姿态误差约20.7°且qy几乎一直饱和。
- 108回合raw/LeRobot动作与状态完全一致，统计与checkpoint一致，72张图像抽查一致。
- min/max动作候选覆盖100%训练动作，往返误差1.11e-16，无非法四元数；没有接入旧模型或训练器。W&B诊断使用offline模式。

原3000模型仍为0/9自主联合成功。原run的`stopped_by_user`、停止标记、模型/数据/日志/视频和暂停监控不得被诊断流程自动恢复或改写。

## 实现与使用

代码位于`high-level/float_ik/learning`。`forward_diagnostics.py`只有initialize/replay/offline/verify阶段，没有训练入口；`diagnostic_replay.py`继承原浮动后端，独立实现三种专家动作回放，完整保留失败及正常耗尽结果。`diagnostic_contract.py`仅提供显式版本的契约和检查，拒绝混用旧权重与新动作统计。

`diagnostic_report.py`负责数据审计、首次轨迹分歧及图表；`diagnostic_history.py`负责控制历史消融；`diagnostic_runtime.py`在Gym环境CPU中核对原控制器；`finalize_diagnostics.py`负责完整视频核验、原run保留验证及文件哈希。

所有数值计算在pro5000项目Conda环境中进行，CPU两线程、nice10。回放通过现有launch.py串行选择并锁定单张空闲GPU，每回合600秒故障保护，无累计时间硬截止。离线评估使用CPU并缓存恒定观测编码，已与原模型forward核对差异为0；未修改训练的数据读取方式。

初始化必须选择新目录；已尝试回放不会被覆盖或静默重试。发生失败时保留尝试分母和日志，不通过删目录/资源记录恢复。新的训练必须另外批准并使用新run。
