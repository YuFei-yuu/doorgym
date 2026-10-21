# R8完整任务专家试采与ACT数据接入结果

本轮已停止：8个场景、8回合，全部走到自动终态，无补跑、训练或ACT策略推理。执行证据支持末端推关、物理关闭后锁存、释放及共同终态保持。按用户明确决定，第5、8回合因开门保持段极小负向浮点命令排除并保留待审；候选集为其余6回合、13,301帧。独立验收尚未完成，不能将自动8/8写成独立验收成功率。

数据链路通过，但发现绕行进度的观测歧义：几乎相同的双深度和Joint9对应不同的未来chunk100动作。本轮不修改输入或网络，不宣称适合直接扩大采集或训练，也不宣称鲁棒性。

## 版本、环境与保护

- 起点：`664aa7c41893967f1f839f85273fa0cf73671177`。
- 分支：`expert/wc4-fulltask-pilot-r8-20261021`。
- 实际8回合执行提交：`8ebdcffddd0b74b1a2dc39856307fe45d5c20cc6`；执行时工作区干净。交付提交仅增本文和执行后的审计脚本，实际提交号见run内`delivery-version.json`。
- main保持`87c5a8665f78b0627c95ea9be6a0e756fc359b11`。原作者源固定`096938c44ecc1c7602304d42502f96658e46eb25`。
- GPU：物理2，UUID `GPU-2c8bfc35-6335-164d-bba2-5c97c3d9cfd0`，CUDA/Vulkan逻辑0。每个进程启动前共享UUID锁及3次空闲核对，记录在`resource-*`和`compute-*`。
- Gym复用`/data/zyf/video2door/.repro/envs/b1z1-blackwell`及Blackwell支持；转换和factory复用本项目`.repro/envs/door-act`。没有安装或升级环境。
- 原始目录：`/data/zyf/video2door-wc4-act/.repro/runs/wc4-fulltask-pilot-r8-20261021`；本地交付：`/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-fulltask-pilot-r8-20261021`。
- R1–R7既有SHA清单共710项全部核验通过；执行源码/资产与冻结清单全部一致。旧数据、模型、环境和`/data/zyf/video2door`支持目录未被清理或覆盖。

## 实现及无仿真检查

`fullcycle_scene.py`从实际门资产铰链、actor变换、实际底盘高度及俯仰建立场景映射，保持R7相对门的紧凑绕行及推关设计。全身通过面取门与侧墙平面在通过方向上更远的一面；全身几何使用实测root姿态。接近和推关起点保持90°设计，不根据实测门角重选。

冻结手册初态及相机分布，seed为62000–62007，第1回合名义、其余7回合随机。初态原始采样、运行有效值、相机局部姿态及逐轴扰动分别见`frozen-parameters.json`、每回合`frozen-trial.json`、`physical-camera-binding.json`和账本。俯仰范围0–5°，相机各轴位置±0.02m、旋转±5°；深度噪声和模糊关闭，没有缩小分布。

`frozen-r7-physics.yaml`明确替换旧`randomization_absolute_ranges`中的门/把手物理范围，随机幅度和门阻力选择均置为固定R7值。导入前把手摩擦/阻尼0.05、弹簧刚度0.5/阻尼0.1，经既有wc4倍率后分别0.025、0.025、0.25、0.05；门铰链摩擦/阻尼、阻力、自动开门/锁力均0。保留的是把手复位，未加入关门弹簧或辅助力矩。每回合核验有效采样及DOF属性，资产/home/碰撞过滤不变。

无场景检查：27项测试通过；8份预检及各31姿态静态IK检查通过。名义回合与R7逐步比较4513步，动作、底盘实测、门角及机械臂实测最大绝对差均为0，支持新增记录未改变名义执行。

## 尝试账本

编号按人类1–8编号；文件目录索引为0–7。所有回合180秒内结束，无系统性记录/坐标/数值错误，全部原始数据保留。

| 回合 | seed | 仿真秒 | 25Hz帧 | 自动终态 | 候选处理 |
|---|---|---:|---:|---|---|
| 1 | 62000 | 90.26 | 2257 | 通过 | 候选episode0 |
| 2 | 62001 | 88.90 | 2223 | 通过 | 候选episode1 |
| 3 | 62002 | 88.82 | 2221 | 通过 | 候选episode2 |
| 4 | 62003 | 89.24 | 2231 | 通过 | 候选episode3 |
| 5 | 62004 | 89.70 | 2243 | 通过 | 用户决定排除、待审 |
| 6 | 62005 | 86.48 | 2162 | 通过 | 候选episode4 |
| 7 | 62006 | 88.26 | 2207 | 通过 | 候选episode5 |
| 8 | 62007 | 89.08 | 2227 | 通过 | 用户决定排除、待审 |

第5、8回合旧开门`initial_hold`各有350个50Hz极小负向命令，最小分别−2.546053963e−7、−4.584171904e−7 m/s；对应实测速度和位移均0，通过后命令无负向，25Hz监督底盘动作也为0。原命令不改写，不以容差替用户解释无倒车。用户选择“先排除两条，保留待审”。这两回合不是已观测到倒退的物理失败，亦不计入严格候选成功。

每回合`review/review.json`给出全身通过步、首次新末端接触、对齐推关步、门运动、锁写入和独立时间重算。闭门段非零记录仅见末端gripperStator—门板，没有腕部/机身承力替代推关记录；对齐末端接触且门关转的步数分别853、653、906、676、670、988、984、928。原开门段存在腕部接触，按继承行为记录，不扩大为接触白名单。图像仅能辅助判断，不能认证全部cooked几何或力单位。

各回合顺序均为全身通过、旧接触分离、绕行、重新末端接触推关、物理关闭保持、实测角锁存、释放、收臂、远侧停稳及朝向恢复。锁前独立实测跨度均0.519999988秒，锁写入与前一实测门角差为0，无吸附到0；解锁至锁存之间没有直接写门角关门或新增辅助力矩。最终共同保持分别2.28、2.22、2.26、2.26、2.04、2.26、2.26、2.28秒，均按实测时间戳而非阶段计数重算，沿用全部原容差。

所有整体RGB及双深度均为同步原执行录像，25fps、全帧解码通过，时长为帧数/25；与仿真末端可能相差至一个记录周期。每回合`stop-state.json`、`stop-near-rgb.png`、完整轨迹及原片抽帧图保留，无状态回放替代。总原始记录17,771帧。原`result.json`中的`nominal_only:true`为旧字段，在随机回合不准确；不改原文件，实际分布以冻结参数和有效绑定为准。

## 记录、转换与训练窗口

控制50Hz，物理步dt为float32表示的0.02秒。每偶数物理步完成后，用作者`record_float_dp_frame`实时记录；帧f源物理步2f，物理后时间约0.02+0.04f秒。图像与实测关节来自同一物理后状态。Joint9状态前两维为上一条25Hz记录动作的底盘量，其余7维为当前实测关节；监督底盘由两次记录之间的实际积分位姿差重建，关节为当前已下发目标。首帧底盘量按作者定义为0。转换后timestamp从0以1/25增长，源绝对时间见`recording-clock.jsonl`，不能混称成50Hz日志隔行抽样。

两视图直接保存原始`uint8 [T,480,640]`，每帧哈希与在线图像缓冲区核对；state/action均`[T,9]`。阶段、门角、接触、相机位姿及评分只作审查元数据，未增加策略输入。所有动作段及末尾驻留均保留，回合可变长，不截为550帧。

候选NPZ逐字节复制原片，映射及SHA在`candidate-index.json`。原作者转换器及sanitizer未修改，使用`joint_state9 --depth_only --fps 25 --image_storage video --video_codec h264`。转换退出0、耗时235.96秒。LeRobot路径为`lerobot/local/wc4_fulltask_pilot_r8_20261021`。视频容器可合并，episode索引和动作窗口保持独立。

真实`lerobot.datasets.factory.make_dataset`和DataLoader检查6回合384窗口（每条64），覆盖开头、所有阶段衔接、关门及末尾。chunk100和padding通过，padding重复本回合末动作，没有跨回合。全部转换状态动作匹配作者sanitizer；策略特征解析确认只有state9和两幅深度，输出action9。`subtask_index`未进入策略，`use_action_loss_weight=False`，其他条件/辅助分支均关闭。两个转换视频均完整解码13,301帧/25fps；H264有损，抽样归一化像素MAE最大0.0036064，原NPZ原始uint8仍完整保留。

初次factory只读检查曾沿用ACTConfig默认权重开关True（数据权重全1）；已保留初次报告，按基线显式False重做最终检查。两次读取的全部状态、动作及诊断特征逐字节一致，无重新采样或仿真。

## 可学习性结果与限制

以Joint9尺度[0.01m/s,0.01rad/s,其余0.03rad]及双图归一化RMSE≤0.02作近邻诊断，检查跨阶段、跨回合及同回合相隔≥5秒；动作差尺度为底盘0.10m/s、转向0.15rad/s、关节0.15rad。共80,559个候选状态/未来动作对、3496个粗图近邻对，排名前12对全部经全分辨率原图确认。这是有限近邻诊断，不是完备可观测性证明。

最清晰例子为名义候选episode0的703与832帧（约28.14与33.30秒），均在绕行段：前置深度RMSE=0，腕部=0.0000693242，六轴及夹爪差约1e−6rad，状态转向差0.007913rad/s。当前监督动作很近（转向差0.004337rad/s），但未来100步chunk最大前进差0.1470m/s、转向差0.5134rad/s。双深度主要看地面/自身、缺少绕行绝对进度，可能使ACT未来chunk监督多义；不能把该结果写成当前瞬时动作冲突或已经证明不可学习。

另对明确开门交互段与推关段检查，最近Joint9归一化无穷范数距离45.686>1，本批未找到满足同一阈值的开/关混淆对；不等于双深度在所有初态都能区分开关门。详细数值和图在`dataset-check/learnability-diagnostic.json`、`learnability-example.png`及`opening-vs-closing-state-separation.json`。本轮不移除困难帧、不增加阶段/历史或改网络，研究路线交独立审查。

## 资源估计及后续协议

全部8条NPZ共277,029,548字节，6条候选205,760,586字节，LeRobot29,253,621字节；全原始日志/证据约5.11GB（近似口径）。候选约15,470原始压缩字节/帧、2,199转换字节/帧。新增N回合存储可按实测平均帧数和上述每帧量估算，包含全部成功失败证据另预算；此估计不授权扩大采集。

实际factory耗时0.151秒，3 workers、batch16、chunk100读取384个窗口耗时2.704秒，约142窗口/秒，为选定窗口暖缓存测量。主检查进程峰值RSS约2.08GiB，非全部worker内存；float32双图单batch约112.5MiB，6个预取batch图像约675MiB。新训练GPU内存和收敛均未测。

只读旧基线训练日志中10,000更新耗时1179.55秒，中位update0.1132秒。未来U次更新的粗略时间为`U * max(旧update中位数,16/本次窗口读取率)`，另加启动/保存；这是历史参考及IO下界，不是本轮训练benchmark，不自动沿用50k预算。

完整任务评测协议见`FULL_TASK_ACT_EVALUATION_PROTOCOL.md`：全部运动只能由策略25Hz Joint9产生并在50Hz保持，禁止专家接管开门、推关、释放或复位；共享实测监视器独立于专家阶段，只用于锁闭抽象和评分，不能作为策略输入。正式评测前还需核对旧策略入口夹爪软范围与专家物理范围差异。本轮没有训练、构造ACT模型或执行forward。

## 命令与证据导航

以下为已用命令结构，预算已耗尽，不能再次执行attempt：

```bash
cd /data/zyf/video2door-wc4-act
# 准备/冻结与seal均在任何场景创建前完成
python -B migration/wc4_act/fullcycle_pilot.py --prepare .repro/runs/wc4-fulltask-pilot-r8-20261021
python -B migration/wc4_act/fullcycle_pilot.py --seal .repro/runs/wc4-fulltask-pilot-r8-20261021
# 每个i=0..7各执行一次；preflight不创建场景
bash migration/wc4_act/fullcycle_r8_launch.sh preflight "$i" 2
bash migration/wc4_act/fullcycle_r8_launch.sh attempt "$i" 2
```

每条完整展开命令及执行提交：`preflight_XX-command.json`、`trial_XX-command.json`；转换：`convert-command.json`及`convert.log`；环境与GPU：各回合`effective-config.json`、日志及资源记录。准备期路径导入和mock测试异常均在无场景阶段修复，初次日志保留；没有以重跑替换实际失败。执行后审计脚本副本见本目录`r8_audit/`及run根目录。

每条`trial_XX/`包含连续`states.jsonl`、`contacts.jsonl`、`lock-writes.jsonl`、`trajectory.npz`、`recording-clock.jsonl`、原始NPZ、双深度、`overview/env_0000/rollout.mp4`、终止快照和审查图。`attempts.jsonl`为完整账本；`candidate-index.json`为6条入选映射；`resource-estimate.json`为资源实测；`dataset-check/`为真实factory和可学习性检查；`pilot-paths-and-lengths.png`为8回合实测路径与长度。源码bundle、起点到交付完整diff及SHA清单随run交付。

本轮在8回合预算结束及交付完成后停止，等待独立验收，不进入下一阶段采集或训练。
