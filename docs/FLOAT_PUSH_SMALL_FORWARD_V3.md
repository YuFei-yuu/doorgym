# 3个标称回合的forward_v3训练验证

2026-10-14，用户已明确授权：随机初始化、单卡batch64、workers=0，训练1000更新，250/500/1000冻结模型并录制闭环视频，结束后汇报，不扩大训练。

服务器run：`experiments/float-push/20261014-forward-v3-small3`。本地交付：`reproduction/pro5000/forward-v3-small3`。

## 固定配置

- 数据取自已保留的`20261013-forward-dp-v1`，仅seed16040、16041、16042，均为x=4.1、y参数=0的标称成功专家回合；1296＋1292＋1283＝3871帧，forward_v3有效窗口3778个。
- 使用`door_action_minmax_continuous_quat_v3`：动作min/max、状态z-score、四元数连续符号、起始历史重复首帧。统计仅来自这3回合。
- 从随机初始化开始，seed42；源网络和四路54×96图像，27D状态/10D动作；窗口16/32/16、100步DDPM；AdamW lr1e-4、weight decay1e-6、梯度裁剪1。单卡batch64、workers=0、逐帧LeRobot读取。
- 250→500→1000串行训练。第一阶段不加载权重，后两段只恢复本run的前一冻结checkpoint及完整随机/采样状态。仅产生250、500、1000三个step模型。
- 每节点训练进程退出后，按16040/16041/16042串行运行3个2500步完整DP闭环回合，保存全部日志、轨迹和25fps视频，失败和超时保留在分母中。
- 闭环使用训练初态，评估小数据拟合，不作为泛化或最终留出测试。正确方向≥80°、通过目标距离≤0.12m、DP-only且图像有效才算自主联合成功；原自然snap机制保留并记录。
- 所有计算在pro5000项目环境，nice10、数学/PhysX两线程；统一启动器逐阶段检查并锁定空闲单卡。正常训练无累计时间硬截止，单仿真回合保留600秒故障保护。
- 原生loss/lr/耗时每10更新记录；优化结束后导入W&B offline，避免日志初始化影响分段训练随机状态。没有线上上传。

## 执行与证据

入口为`high-level/float_ik/learning/small_forward_v3.py --root <新run>`。代码、原始数据及专家对照复制到独立run，并保存SHA256。后台监督进程与SSH连接分离，状态在`state.json`，训练进度在`training/progress.json`，阶段PID/GPU在`active_command.json`及`resources/`。

训练和推理分别使用`.repro/envs/door-dp-train`与`.repro/envs/b1z1-blackwell`；旧模型、旧run停止标记和暂停监控不被改变。每个checkpoint核验更新数、契约、仅3回合的统计及恢复来源。

检查模型位于`training/checkpoints/step_000250.pt`、`step_000500.pt`、`step_001000.pt`；视频位于`evaluation/step_*/seed_*/rollout.mp4`。汇总在`analysis/REPORT.md`、`analysis/checkpoint_success.csv`、`analysis/all_results.csv`。同初态专家对比和离线动作误差在`analysis/`补充，源码差异与初始观测一致性单独记录。

本轮最多1000更新，没有1000之后的训练入口。后续是否扩大数据或训练量，须由用户根据本轮结果另行决定。

## 当前记录

**已在1000更新结束，未扩大训练。** 全部9回合均录制2500步/50秒，动作来自DP、无专家回退/无效图像/进程错误，视频完整解码通过。训练优化耗时1820.11秒，含转换、分段初始化、闭环检查和收尾的监督流程约49.51分钟。

| 更新 | 末次日志loss | ≥80°开门 | 原pass_done通过 | 自主联合成功 |
|---:|---:|---:|---:|---:|
| 250 | 0.074934 | 3/3 | 0/3 | 0/3 |
| 500 | 0.031011 | 3/3 | 0/3 | 0/3 |
| 1000 | 0.019632 | 2/3 | 2/3 | 2/3 |

1000模型的seed16040/16042分别在13.44s/18.26s达到80°，16.60s/20.36s触发通过事件，事件距离0.1129m/0.1131m；seed16041最大79.698458°，未达80°。成功事件后仍按计划录满50秒，模型继续前行；没有将停稳作为本轮验收项。

3个模型及全部原始证据保留。最终模型SHA256：`ea20f964c96099115c3ee326475210eae5f4f48a8ac3f1601d8116575b2c0f06`。

本轮说明新版链路在小数据上可学出有效闭环，但不是泛化或稳定性验收。原实验的模型/停止记录、专家记录及592518个文件清单保持不变；训练/仿真进程退出，GPU已释放，其他原有GPU4–7任务保持运行。

本地入口：[完整交付索引](/Users/yu/Data/Video2Door/reproduction/pro5000/forward-v3-small3/analysis/INDEX.md)、[结果与限制](/Users/yu/Data/Video2Door/reproduction/pro5000/forward-v3-small3/analysis/CONCLUSIONS.md)。下一轮须由用户另行决定，未自动采集、训练或运行留出测试。
