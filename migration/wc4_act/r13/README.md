# R13 工程修复与开训准备

本阶段仅源码和CPU工程回归；正式ACT训练、真实策略推理、仿真和闭环评测均为0。工程入口不默认创建训练run或W&B run，不生成训练授权文件。全部CPU数值回归在pro5000既有项目环境执行。

## 动作执行证据

`door_policy_worker.py`使用独立观察器记录实际`sample_action_chunk()`调用和真实deque出队；不改网络、输入、归一化、动作或队列调度。每条回复包含worker PID、epoch、chunk ID、动作/观测序号、真实队列前后深度、所选25条动作及SHA256。父端代理长度通过RPC查询真实队列，不再返回常数0。

`rollout.py`采用`ActionJournal`连接worker回执与实际目标下发/物理完成回执。每个25Hz动作保持两个50Hz子步；开始新chunk前必须观察到上一chunk的25次连续出队。末回合提前结束时明确记录部分chunk/部分动作，不伪称完成全部动作。完整记录仍包括raw/applied/clipping delta、状态、图像无损缓冲和时间。CPU夹具证明的是实际跨解释器IPC及控制器队列/应用回执链，不是Gym物理执行证明。

## 可控停止与保存

`StopBoundary`显式安装SIGINT/SIGTERM处理器，覆盖后台进程继承的SIG_IGN；处理器只设置请求，不打断在途optimizer。也可用原子文件请求：

```bash
# 对未来已授权训练run使用；本命令本身不训练或恢复。
python -B migration/wc4_act/r13/request_stop.py /absolute/new/run --reason 'operator stop'
```

在完整作者update_policy返回、完成计数和日志fsync后保存。新更新开始前也检查请求。保存调用作者原有save_checkpoint：同一边界的模型、optimizer、scheduler（如有）、RNG、config和pre/post processors先写私有临时目录，文件/目录fsync和SHA完成后原子发布，最后写`STOPPED.json`并退出75。`COMMITTED.json`与实际step/hash不符的快照拒绝恢复。不会自动恢复或补跑。

监控失败只持久化停止请求，不再对trainer/进程树直接SIGTERM。监督进程等待一致保存；超时标明未确认、保持租约，不谎称干净停止或自动强杀。不可控崩溃、SIGKILL或optimizer内部异常不承诺一致保存：留下unsafe/未发布状态，不产生干净恢复结论。

恢复状态遵循作者的model/optimizer/scheduler/RNG/processors语义。作者原有DataLoader的sampler游标和预取队列不在快照中；不声称多worker恢复后样本顺序逐位连续。本轮没有改变shuffle/采样器/输入或训练配方来取得这种额外保证。

## 接触诊断

基于实际robot-door body对、原始SDK lambda、门角实测变化分类，不读取专家phase。分别记录零lambda、非末端、非零lambda和与实测关转同时出现的记录；不将非零lambda称为校准力，不宣称接触覆盖率、做功或致因，也不增加自动否决。R11所需`loaded_door_contact`保留原始非零lambda别名语义。最终成功仍由未改动的R11评分和独立原片/源码审核决定。

## CPU回归

在服务器独立分支、干净工作区内：

```bash
source migration/wc4_act/activate.sh act
CUDA_VISIBLE_DEVICES='' python -B migration/wc4_act/r13/run_cpu.py
```

`run_cpu.py`只运行工程夹具，每次新建编号日志目录，不覆盖失败日志。包括历史缺陷动态复现、ACT解释器及Gym Python3.8到ACT Python3.10的真实worker路径、75动作/150回执、可控停止及官方保存恢复、监督失败链、八类接触诊断、实际训练/导出CLI配置、全72条真实统计的归一化保存加载，以及原R11九项边界回归。玩具网络/优化器与伪后端只用于工程测试；未构造ACT/DP网络或调用真实策略。

## 后续开训说明

1. 本阶段交独立验收，先取得下一份正式任务书，明确R12已耗1,122–1,123次更新的计入方式。R12没有可恢复模型；R13快照全部是玩具夹具，不能当训练checkpoint。
2. 新任务建立唯一新run、执行源码绑定和容量证据；复制本轮ready冻结清单，保持72份初态/相机tuple及原R7 YAML引用字节不变。中期仍62300–62307，终期仍62200–62263。不要重采样。数据引用仍为R11的72条159,529帧。
3. `runtime.py`的`training_argv()`提供原配方。实际`TRAINING_ORDER.json`需由未来执行方依下一份用户任务填写，包含`authorized_by_new_user_task`、`r12_consumed_budget_acknowledged`、seed1000和steps30000；本阶段没有该文件。若未来明确授权恢复，必须指定经COMMITTED/hash验证的真实停止checkpoint，不能用夹具或R12现场。
4. GPU阶段按手册复查空闲状态，持有项目共享UUID锁FD9，`VIDEO2DOOR_GPU`与锁文件UUID一致；用既有activate.sh act运行。未来入口为`r13/launch_train.py`，通过`WC4_EXECUTION_RUN`显式绑定新run；父进程必须保留锁至trainer和子进程退出。W&B另建对应run并保存真实ID/链接。
5. 010000/020000/030000保存后使用R13独立导出/评测入口，固定8/8/64回合、每回合180秒；停止请求在导出后、创建下一场景前及当前物理步边界传播。原预算、成功判据、25动作执行horizon不变。真实ACT/GPU导出一致性、Gym时序和原片验收只能在下一任务明确预算内完成。

R13不执行以上开训步骤，也不因CPU通过而自行追加一次真实烟雾闭环。
