# 停车问题、上游对照与下一步

更新：2026-10-14。只读核查和CPU回归测试；未启动实验训练、仿真或采集，未实施停车控制改动。

## 上游版本核实

通过GitHub及`git ls-remote`核实，源仓库`Sivan666666/doorgym`的HEAD/main仍为`0acc2f050479fb54a0670ccb90bc72b98b52a54a`，与项目固定源提交一致。以下比较针对该不可变提交，不依赖网页缓存的更新时间。

源仓库有两条不同执行路线，不能把脚本停车等同于DP已经学会停车：

1. **独立浮动IK脚本**：`isaacgym_float_ik_b1z1_basearn_push_door.py`按阶段插值底座位置；`return_home`之后进入`hold_home`，每步把底座位置/yaw固定为`base_push/yaw_push`。这是脚本指定终点保持，没有调用Door DP控制器，也没有证明训练后的DP能自主停车。[源代码1093–1119行](https://github.com/Sivan666666/doorgym/blob/0acc2f050479fb54a0670ccb90bc72b98b52a54a/high-level/float_ik/isaacgym_float_ik_b1z1_basearn_push_door.py#L1093-L1119)
2. **Door DP训练/推理所用的推门入口**：`play_b1z1_push_with_door_asset_camera.py`到达通过目标后锁存`pass_done`，脚本设置vx/yaw为零；但是后面的DP调用会再次写入模型的vx/yaw，没有在DP之后执行停车覆盖，也没有把完成事件传给动作队列。[脚本与DP写入顺序2250–2308行](https://github.com/Sivan666666/doorgym/blob/0acc2f050479fb54a0670ccb90bc72b98b52a54a/high-level/play_b1z1_push_with_door_asset_camera.py#L2250-L2308)、[DP动作队列及应用接口](https://github.com/Sivan666666/doorgym/blob/0acc2f050479fb54a0670ccb90bc72b98b52a54a/high-level/dp/door_dp_common.py#L514-L548)

## 当前项目与源仓库的差异

| 环节 | 固定源提交 | 当前项目 |
|---|---|---|
| 浮动底座终点保持 | 独立IK示例脚本硬编码`hold_home`位置保持 | 训练使用Door DP入口＋`wholebody_float`速度积分后端，不直接运行该完整脚本的终点阶段 |
| 专家通过路线 | Door DP推门脚本含门前预对齐，再前往通过目标 | `forward`模式跳过再次预对齐，保持起始侧定义，直接向前穿越；这是已确认的任务改动 |
| 脚本完成后速度 | `pass_done`分支vx/yaw置零；无DP时可持续保持，直到循环结束或环境重置 | 同样置零，但浮动专家在首个完成控制步后直接结束回合 |
| DP完成后速度 | DP覆盖先前脚本命令，无显式终止队列或完成后保持 | 继承该行为；small3固定录制2500步/50秒，所以暴露了通过后持续行走 |
| 专家数据尾部 | 记录器添加完成帧后就关闭，不采集多秒停车保持 | 自有浮动记录器也在首个完成步后结束；本次3回合各仅1帧终点停车 |
| 训练窗口尾部 | 只接受完整32步未来动作，没有尾部填充 | 同样没有尾部填充；v3只修复起始历史补齐，并未修复末尾覆盖 |
| 归一化与工程功能 | 动作z-score、采样±1裁剪，简化checkpoint记录 | v3动作min/max、四元数连续符号、起始补齐、完整续训状态、不可变checkpoint和独立审计；旧模型语义保留 |

源数据关闭位置：[2355–2357行](https://github.com/Sivan666666/doorgym/blob/0acc2f050479fb54a0670ccb90bc72b98b52a54a/high-level/play_b1z1_push_with_door_asset_camera.py#L2355-L2357)。源窗口边界：[138–164行](https://github.com/Sivan666666/doorgym/blob/0acc2f050479fb54a0670ccb90bc72b98b52a54a/high-level/dp/train_door_dp.py#L138-L164)。当前主要差异见本地`high-level/play_b1z1_push_with_door_asset_camera.py:1936,1955,2479`及`high-level/dp/train_door_dp.py:190`。

**因此，DP无显式停车和训练尾部覆盖不足并不是forward_v3新引入的问题。源独立浮动脚本确实会保持终点，但不能直接作为源Door DP模型能够停车的证据。**

## 本次数据证据

完整机器可读结果：[停车统计](evidence/small3-stop-analysis.json)、[源码与模型核验](evidence/small3-stop-source-verification.json)。统计在pro5000项目训练环境执行，禁用CUDA、限制两线程、nice10；输入15个文件的SHA256随结果保留。

| 1000更新seed | 首次通过 | 通过后路径 | 通过后平均vx | 通过后重新预测次数 |
|---|---:|---:|---:|---:|
| 16040 | 16.60s | 8.8256m | 0.26424m/s | 105 |
| 16042 | 20.36s | 8.3593m | 0.28203m/s | 93 |

两个回合通过后均没有一帧`abs(vx)<0.01m/s`，去除一个完整16帧动作队列之后仍继续行走8.6465m和8.1606m。实际底座位移落实了这些命令，不是惯性或队列剩余动作造成的短暂延迟。

三个专家3871帧、3778个窗口中，各仅最后一帧为完成停车，vx/yaw均0、实测该步底座位移均0。停车标签只出现于每回合最后一个32步窗口的offset31，前16步停车标签数量为0；终点观测也没有成为窗口当前输入。原始终点停车帧比例0.07750%，枚举窗口中的终点停车目标比例0.002481%。其他阶段的静止标签不能替代终点停车条件。

对比视频专家结束后会保持最后一张图并标注`EPISODE ENDED - FINAL FRAME HELD`；它不是实际记录的多秒停稳段。原summary的`end_reason=pass_done`表示事件发生过，并非DP真的当场结束。

small3在250/500/1000更新的开门＋通过成绩分别为0/3、0/3、2/3。1000更新失败seed16041最大79.698458°，仍判失败；两个通过回合没有停稳。三者都是训练初态，只能证明新版链路初步学出有效闭环，不能认定泛化或稳定性达标。loss下降与各阶段动作误差并不单调对应，训练量不足可能同时存在，但不能消除上述数据/终止缺口。

复算入口（在服务器项目环境、独立的新输出目录运行）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
nice -n 10 .repro/envs/door-dp-train/bin/python high-level/float_ik/learning/analyze_stop.py \
  experiments/float-push/20261014-forward-v3-small3 experiments/float-push/new-stop-analysis
```

## 下一步建议与确认边界

1. **先区分系统完成停车与DP自主停车。** 若采用与脚本类似的系统终止方式，在完成事件后清队列并施加零底座速度，明确标记`terminal_hold`；不能算作DP自主停车。建议将DP自主停车列为最终模型验收项，但这一要求需用户确认。
2. **自主停车优先补真实示范尾段。** 在原3个标称初态采集通过后3–5秒零速保持的真实状态和图像；保证停车转换及保持观测成为训练窗口输入，停车标签覆盖实际执行的前16步。末尾不足32步时使用显式有效性掩码，只监督真实采集的目标；不复制静态图像伪造动力学。新增数据/窗口契约版本，保留原v3模型解释。
3. **保持小规模对照。** 随机初始化、单卡batch64、workers0、1000更新及250/500/1000视频节点保持不变，先隔离数据/窗口修复收益。增加固定终点转换与保持阶段的离线误差，单独定位79.698°失败回合的接触和末端偏差。若仍未拟合，再评估追加更新；暂不同时改变DDP、batch和读取方式。
4. **分开验收开门、通过、停止。** 建议通过后0.5秒内速度<0.01m/s、|yaw-rate|<0.02rad/s，随后2秒漂移≤0.02m，完成事件后的停车额外行程≤0.10m。上述新增阈值待确认，不改写旧成绩。小数据先3/3，再做标称10个固定推理随机种子的开发验证；候选冻结后才运行独立留出测试。

目前仅完成分析和代码归档，未实现上述停车控制、重新采集或追加训练。27项CPU单元测试及进程超时/互斥脚本检查通过，日志保留在`reproduction/pro5000/commit-review-20261014/tests.log`，服务器为`.repro/commit-review-20261014/tests.log`。
