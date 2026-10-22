# R15启动停止交付：新增更新0、评测0

2026-10-22。R15已按任务书启动，但执行方在监控适配中遗漏R13停止模块的导入路径，监控进程启动报`ModuleNotFoundError: No module named 'stopping'`。监督器请求可控停止；真实训练在首个新增optimizer更新之前停止，保存了已提交的030000一致状态。**学习趋势结论：证据不足。未达到040000/050000，未生成其模型或评测结果。**

## 实际执行和预算

执行提交`04b65d5`，分支`execute/wc4-fulltask-act-r15-20261022`，基于任务开始时实际HEAD `295d392`。本地/服务器工作区起初干净且相同，main仍为`87c5a8665f78b0627c95ea9be6a0e756fc359b11`。R13执行、评分、锁闭、物理、模型/数据核心未改，仅新增R15编排适配。

R14真实源checkpoint为服务器R14 run的`continuation-train/train/checkpoints/030000`，模型SHA256为`fdd5ad12f657a78dd7d9b483e7b657d9757bff4c829af6671017682543768159`。启动核对全部11个源状态文件哈希、72条159,529帧数据及16个实际冻结初态/相机参数，按R15直接恢复，没有前置离线诊断实验、烟雾闭环或30k补评。

| 项目 | 计划 | 实际 |
| --- | ---: | ---: |
| 新增optimizer更新 | 20,000 | **0** |
| 累计更新终点 | 50,000 | **30,000** |
| 040000评测，62200–62215 | 16 | **0** |
| 050000评测，同16初态 | 16 | **0** |
| 不确定更新 | 0 | **0** |
| R12损耗 | 单列 | 仍为1,122–1,123 |

20,000新增更新与32回合预算均未消耗，没有替换失败、追加64回合或新采集。没有策略效果或人工代表视频可交付，不能把未执行评测当0成功率。

## 恢复与停止实证

`restoration-before-first-update.json`证明实际恢复后：模型234项、optimizer459项（153个step均30,000）、参数组、全部8项RNG（含CUDA）及pre/post归一化状态与R14源相同。实际有效配置保持seed1000、batch16、lr1e−5、ResNet18默认ImageNet预训练、chunk100、训练n_action_steps50及原输入网络；累计steps=50000，resume=true。新增W&B监控ID不改变模型恢复。

监控错误发生后，监督器请求停止，训练退出75、`STOPPED.json`报告完成边界30000、不确定更新0；`updates.jsonl`与`completed-updates.jsonl`均为空，evaluation-claims为0。原子保存点为`stopped-checkpoints/stop-030000`，11项COMMITTED哈希全部通过。

停止快照与源再次逐张量比较：模型234项、optimizer459项、pre/post归一化各110项相同。RNG八项中只有CPU `torch_rng_state`不同，系恢复之后进入DataLoader初始化/取批阶段已有随机数消耗；CUDA等其余RNG相同。快照保存的是实际停止时RNG，不能误称与源逐项相同；DataLoader sampler/prefetch游标仍未保存，不声称样本顺序连续。

## 资源、环境与监控

启动容量约140GiB，原始证据/模型预留约80GiB，沿用40GiB可控停止余量；未清理历史。GPU0初次复查发现他人`/data/xd/envs/sim/bin/python`进程1985352，空闲检查拦截，尚未创建训练进程；该记录保存在`launch-attempt-001/`。随后按任务许可改用空闲GPU1，重复空闲检查并持共享UUID锁：`GPU-c9b25165-aabc-b3ed-d0fc-fbbd2842b772`。

实际supervisor1997927、trainer1998101、monitor1998102均已退出，GPU1锁释放有receipt，未触碰他人进程。使用既有activate.sh、door-act ACT环境；本轮没有创建Gym场景或策略评测，也没有升级依赖或全局配置。配置、解释器及实际LeRobot路径见`startup-binding.json`，真实命令见`prepared-command.json`/`train-command.json`。

W&B：[5e15e3d7](https://wandb.ai/fi-yu-individual/door-dp-stop-tail/runs/5e15e3d7)。已记录真实停止原因、累计30000、新增0、评测0、一致快照可用及不自动恢复。没有新loss/吞吐可报告；这不构成“W&B持续失联”，是本地监控模块导入错误。

## 已完成的修复与验证范围

修复提交`300622f`仅给R15监控补充R13模块搜索路径，保持R15 common/runtime优先，复用既有`stopping`，未改研究变量。停止后才修改源码，没有热改运行中的依赖；原执行source-binding仍保留`04b65d5`的哈希，不重写为修复版本冒充已执行。

`STOP-DELIVERY/monitor-cpu-regression.json`与原始before/after日志证明实际监控模块的启动导入：旧版退出1并复现同一错误，修复版退出0。该CPU回归使用提供路径的common fixture和不存在的PID，在W&B网络轮询/训练之前退出；无模型前向、optimizer更新、仿真或评测场景。**只证明本次导入故障已修复，未声称重新验证真实训练—监控全过程。**

R15工程入口已配置40k/50k各16回合、与R14同seed30k记录配对，以及保留全部32段原始整体视频供交互/穿模复核；这些尚未在实际评测中执行，不能作为已通过证据。没有采用轨迹相似度、单条接触记录或新接触门槛替代R11评分。

## 冻结与待确认事项

服务器run：`/data/zyf/video2door-wc4-act/.repro/runs/wc4-fulltask-act-r15-s1000-20261022`。

本地交付：`/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-fulltask-act-r15-s1000-20261022`。包含启动/错误/停止原始日志、命令/配置/源码绑定、真实恢复检查、停止快照与比较、CPU修复回归、W&B记录、预算及源码差异。模型和数据历史保留。

任务书明确要求实现错误时可控停止，且“不自动…恢复或追加”，因此本轮**停止并冻结，不自行续训**。待用户/方案对话确认：是否在修复版本上从原R14真实030000完整checkpoint重新开始尚未消耗的20,000更新＋32回合，使用新的R15续接输出目录并保留本次现场。若授权，仍保持同数据、seed、模型/输入、物理、判据和horizon，不重跑30k、不增预算。
