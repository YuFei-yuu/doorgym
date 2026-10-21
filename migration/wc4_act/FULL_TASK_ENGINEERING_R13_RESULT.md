# R13 工程修复与开训准备：CPU工程回归通过，等待独立验收

2026-10-22。执行方结论：**本任务书规定的CPU工程范围通过**。冻结交回方案对话独立验收，本阶段不自动开训。正式ACT训练更新、真实策略推理、仿真场景和闭环评测均为 **0**；未新建W&B run、未安装升级依赖、未改变研究变量。

## 版本和保护

从R12冻结交付 `3c713de5a5206c503878152b108a418f7c02af8d` 开始，新分支 `execute/wc4-engineering-r13-20261022`。实际最终CPU回归代码提交为 `f1a8b9f`；开训说明和冻结工具随后单独提交，最终交付HEAD见run内 `delivery-version.json`。本地和服务器工作区均冻结为同一交付提交，无工作区差异，未push远端。main保持 `87c5a8665f78b0627c95ea9be6a0e756fc359b11`。

改动仅为跨进程协议的附加观察证据、独立R13执行/停止/诊断外围及CPU测试。原ACT网络与配方、数据及统计、R7物理/资产、R11评分、R10时间戳修复和R12源码文件均未改变。全72条159,529帧仍为训练数据，未采集或删帧。中期62300–62307与终期62200–62263的72份冻结初态/相机tuple与R12逐项完全相同，未重采样。

停止后对325个保护文件重算SHA256全部相同，覆盖已验收派生数据清单、R12历史证据及受保护源码。另以Git差异门槛确认 `high-level/lerobot`、`high-level/float_ik`、`high-level/data`、`low-level`、`third_party`、R11和R12目录完全未改。附源码绑定、完整patch及bundle。

## 跨进程动作证据：动态修复前后对照

历史对照在独立子进程中加载R12的实际backend/worker源码，只将模型backend替换为明确标注的CPU工程替身。真实worker完成一次实际出队后，其deque长度为24，父端代理仍报告0；原始对照日志保留。这不是依据源码字符串推断。

修复后，真实生产代理、pickle长度前缀协议、worker命令分发、控制器预处理、采样/入队/出队路径全部执行，只替换网络backend，不构造ACT/DP或读训练checkpoint。两种父端环境各验证：

| 路径 | 完整chunk | 实际出队动作 | 夹具应用/完成回执 |
| --- | ---: | ---: | ---: |
| ACT Python3.10 → ACT Python3.10 worker | 3 | 75 | 150 |
| Gym Python3.8 → ACT Python3.10 worker | 3 | 75 | 150 |

worker的实际backend调用事件发生在观测0、25、50；每个chunk动作索引0–24、真实出队后深度24–0、逐动作内容与实际选中的25行完全一致。worker回传PID、epoch、chunk ID、动作/观测序号、真实queue_before/after、新采样事件、所选动作及SHA。父端不再用常数长度或按时钟推算重推边界。reset、clear、损坏回执拒绝及末尾部分动作/chunk也做动态回归。

新的 `ActionJournal` 在实际评测入口中连接worker回复、目标下发和物理步完成三个事件；每个25Hz动作须有两个50Hz子步回执，记录raw/applied/delta，缺失或不连续即停止，不伪造完成。CPU测试中的应用/完成回执来自夹具；**本轮没有验证真实Gym物理完成**，不能将150次夹具回执称为150次仿真。

## 可控停止和一致保存

SIGINT/SIGTERM处理器显式覆盖后台继承的SIG_IGN；信号只设置请求，文件请求也可独立使用。完整作者 `update_policy` 返回后才完成计数/日志fsync，停止时不再打断在途optimizer。新更新前也检查停止请求。模型、optimizer、scheduler、RNG、配置及pre/post processors由作者原保存接口写入私有目录，文件/目录fsync与SHA完成后原子发布，再写STOPPED并退出75。恢复前校验COMMITTED、全部文件SHA及训练step，不接受部分或损坏快照。

CPU玩具夹具覆盖更新前、forward内、backward后、optimizer后、完成后及继承忽略SIGINT的真实子进程。完成边界分别为0或1，均为**零不确定更新**；保存与恢复后的模型、优化器、scheduler、RNG及处理器状态对应同一边界，恢复后下一次玩具更新逐位一致。另用真实作者 `update_policy`、Accelerator CPU及真实作者保存/加载器验证了接入链，而非仅测试替代保存逻辑。

监控故障现在写STOP_REQUEST，不直接杀trainer或整个子树；实际监督夹具验证“监控退出4 → 请求落盘 → 子进程一致保存 → 退出75”，无强杀。保存失败不发布最终checkpoint；optimizer内部异常明确写UNSAFE，不宣称干净恢复。外部SIGKILL、进程崩溃或磁盘不可写等不受控故障不承诺可保存，未掩盖其不确定性。

原作者恢复状态不包含多worker DataLoader sampler游标/预取队列；本轮未为此改变采样器、数据或配方，**不保证异常恢复后样本顺序逐位连续**。GPU RNG、真实ACT大模型序列化以及在线W&B终止同步尚未做实际运行验证。

## 接触诊断和训练—导出—评测衔接

R12使用恒定phase=act，旧 `off_target_closing_contact` 依赖phase=close，可能误报为未发生。R13改用真实body对、原始lambda与实测门角向关闭方向变化的同时记录，不依赖专家phase。八项CPU对照覆盖非末端/末端、零/非零lambda、关转/开转/保持、墙体/自身及body顺序互换。非零lambda不称为校准力，不宣称做功或致因，不增加成功/失败条件。R11所用原始非零接触别名语义不变；零lambda单独不证明接触或释放。

接入检查覆盖实际训练argv/配置解析、作者更新及保存恢复接口、原有export CLI的checkpoint路径和25步参数。用全72条真实统计建立官方ACT pre/post processors，保存/加载后与直接Door归一化接口逐项比较：state、action、两路图像归一化最大误差均 **0**，动作反归一化误差 **0**，保存加载结果相同且保持float32。输入为明确标注的合成张量，不构造网络或执行真实策略。

原R11九项判据回归全部通过。正式入口仍保持seed1000、ResNet18默认ImageNet预训练、batch16、lr1e−5、chunk100、训练n_action_steps50、推理前25个25Hz动作后重推；评测仍为8/8/64、每回合最多180秒。锁闭抽象和R11评分未改；真实ACT权重导出一致性和Gym现场行为留待后续授权预算。

## 命令、环境和证据

最终完整CPU回归为 `regression_004`，九个命令退出码均0：历史对照、ACT父端IPC、Gym父端IPC、停止、监督、接触、作者接口衔接、真实统计归一化、R11边界回归。每项实际argv、开始/结束时间、退出码及原始日志保留。工程过程中首轮IPC夹具漏填Joint9动作名，触发默认10维名称与9维输出不符；修正夹具metadata后通过，首次失败日志未覆盖。随后只因新增具体边界/衔接检查重跑CPU回归，无任何正式实验重试。

统一CPU入口为：

```bash
cd /data/zyf/video2door-wc4-act
source migration/wc4_act/activate.sh act
CUDA_VISIBLE_DEVICES='' python -B migration/wc4_act/r13/run_cpu.py
```

CPU测试用到的小型玩具网络/optimizer和伪backend均标注fixture_only；它们不是正式ACT训练更新或策略推理。ACT解释器为既有door-act Python3.10.22，LeRobot实际指向本仓库；Gym父端为b1z1-blackwell Python3.8.20。激活后显示的 `/blackwell/cuda-12.8/bin/python` 是符号链接，规范路径仍是 `b1z1-blackwell/bin/python3.8`，sys.prefix及Torch路径均已核验。两套环境的CUDA_VISIBLE_DEVICES为空、cuda_available=false；未混环境、未改依赖。原始环境绑定JSON随交付。

服务器：`/data/zyf/video2door-wc4-act/.repro/runs/wc4-engineering-r13-20261022`。

本地：`/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-engineering-r13-20261022`。

核心证据为 `readiness.json`、`protected-before.json`、`source-binding.json`、`frozen-files.json`、最终回归命令/日志、worker/backend事件和逐动作回执、玩具停止快照及COMMITTED、诊断前后JSON、归一化结果、patch与bundle、最终交付SHA清单。没有本轮真实模型、checkpoint、成功率或代表视频。

## 简明开训说明和剩余限制

具体操作见 `r13/README.md`。先交独立验收，取得下一份正式任务书并明确R12已耗1,122–1,123次更新如何计入预算；R12无可恢复模型，R13所有保存样本均为玩具，不能充当训练checkpoint。本阶段不创建TRAINING_ORDER，不启动新run，不自动恢复或重开30k。

未来执行方依据新任务建立唯一run及绑定，沿用本轮冻结的72个初态/相机tuple和R7 YAML引用；GPU空闲检查和共享UUID锁FD9持有后，经既有activate.sh调用R13启动入口。入口校验新任务授权、R12预算确认、源码/冻结清单和GPU UUID；真实训练、W&B新run及既定8/8/64评测只能在新任务范围内执行。

未验证项明确保留：真实ACT/GPU forward与权重导出等价；CUDA/Vulkan实际场景绑定；Gym真实动作/图像时序、锁写入合法性与末端推关/释放原片；GPU保存恢复与在线W&B停止同步；原作者DataLoader恢复的逐样本连续性。没有为补齐这些事项扩大本阶段0预算，也不将CPU通过当作真实闭环通过。

最终停止状态：**R13工程范围完成、冻结交付、等待方案对话独立验收；未开训。**
