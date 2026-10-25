# 第八轮：50条新专家数据转换与定版验收

2026-10-25。**50条完整来源、121,305帧已转换为独立LeRobot数据集，全部规定的数据验收通过。实际训练factory的3-worker DataLoader完整读取121,305个样本，各索引恰好一次，共7,582批，异常0。完成后冻结停止，供方案对话独立验收；未启动训练。**

## 版本、范围及有效位置

交接基点为第七轮最终提交`4d35e3226568d8dfc12e91372e84da68eb614235`。实际转换及五项流水线命令提交为`d3e663e59d6896f95147335c7032db088b180afc`；读后来源复核及封存工具提交为`84b818e39c1466671e38b026e269f5327e4a22be`。最终报告提交、两端及origin一致性见freeze-receipt.json与version-verification.json。

继续使用execute/wc4-newexpert-stage1-20261025。main保护值87c5a8665f78b0627c95ea9be6a0e756fc359b11。本轮只新增newexpert_stage8入口、检查器、封存工具及报告；作者转换器、sanitizer、数据读取实现、专家轨迹／几何、物理资产、Joint9和R11均未修改。

服务器run：`/data/zyf/video2door-wc4-act/.repro/runs/wc4-newexpert-stage8-20261025`。

最终有效数据：`/data/zyf/video2door-wc4-act/.repro/runs/wc4-newexpert-stage8-20261025/lerobot/local/wc4_newexpert_stage8_20261025`。

新repo_id：`local/wc4_newexpert_stage8_20261025`。

本地交付：`/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-newexpert-stage8-20261025/FINAL`。含完整新数据集的无损tar字节归档及小型审查证据；原始50条大NPZ／物理日志继续留在原服务器路径，未在本地展开。

## 来源与转换

输入为第七轮已验收候选索引、统一来源索引及审核决定。三份文件SHA与第七轮独立本地FINAL/server-SHA256.json逐项相符；本轮启动记录的SHA与当前值也相符。50条原NPZ在启动和全部检查结束后均校验原SHA，全部保持原字节。

| 来源 | Seed | 条数 | 帧数 | 处理 |
|---|---|---:|---:|---|
| 原第六轮 | 62500–62507 | 8 | 19,367 | 原路径／原SHA继承 |
| 第七轮新增 | 62700–62741 | 42 | 101,938 | 原路径／原SHA继承 |
| 合计 | 50个固定来源 | 50 | 121,305 | 全部完整转换，无替换、裁剪或拼接 |

raw-links/episode_000000.npz至000049.npz是指向原文件的有序符号链接，不复制或重写原始数据。新episode为0–49，累计index为0–121304；每条frame_index从0重建，timestamp按f/25生成。SOURCE_TABLE.csv及candidate-index.json给出50条原路径、SHA、完整长度及新累计帧范围。

使用原作者convert_door_raw_to_lerobot.py：joint_state9、depth_only、25Hz、video、h264、预载workers4。原sanitizer版本door_dp_sanitize_near_zero_rate_v1，eps=1e-5；状态／动作额外预处理均为identity。没有复用旧统计、episode边界或帧索引，也没有启用overwrite或恢复旧输出。

最终8个数据文件合计273,256,495字节，包括两路H.264视频、数据与episode/task Parquet、新info／stats及作者特征sidecar。视频每路121,305帧，25fps，完整时长各4852.2秒。完整命令、环境、实际提交、开始／结束及退出码见COMMANDS.md、commands/和command-ledger.json。

## 逐项验收

| 项目 | 结果与证据 |
|---|---|
| 来源／索引 | 50条长度、episode、局部frame_index、累计index及f/25时间对应；读后50个原NPZ SHA再次一致 |
| 全量数值 | 状态和动作各1,091,745个元素，与各原NPZ经同一sanitizer后逐元素精确相等 |
| 图像及视角 | 前置／腕部均完整解码；共242,610帧，每路每帧恰好对应一次，无缺失、重复或跨episode映射 |
| 阶段和尾部 | 3,188个阶段交界／首尾／关闭窗口，200个多worker批次通过；chunk100不在阶段交界截断，末端重复本条最后动作且padding精确 |
| 时间精度 | 容器episode偏移为Parquet float64，保留到实际查询；原生PTS／查询直接float64比较，最大误差3.662110430013854e-6秒，全部严格小于1e-4秒 |
| 精度回归 | 阈内、等于阈值、阈外、长时间查询／PTS、乱序及重复查询均按原规则通过／拒绝；未先转float32再提升，未扩大容差 |
| 新统计 | 状态、动作及两路图像的所有作者统计字段，与独立重新计算结果精确一致；数值count各121,305 |
| 全量真实读取 | 原make_dataset、TorchCodec、3 workers、batch16、shuffle seed1000、drop_last=false、prefetch2；121,305样本各一次，7,582批，最后9样本，异常0 |

PyAV用于独立容器全帧解码审计；实际训练factory、窗口检查和全量DataLoader始终使用原TorchCodec后端，没有更换后端绕过异常。全量读取逐批核对每个状态、100步动作与padding；两路图像均为有限float32[3,480,640]、范围[0,1]。遍历本身811.59秒，仅为CPU读取验收，不作GPU训练性能结论。

H.264为原作者有损格式，不要求与原图逐字节一致。窗口原图对照的最大归一化MAE约0.003819，保留为诊断，未新增像素误差通过阈值。视角对应结合原作者写入源码、每路统计复算、窗口对照和全量时序共同核验。

图像统计遵循作者编码前图像的确定性采样、下采样、get_feature_stats及逐episode aggregate_stats；每路stats.count=17,258是统计抽样数，不是删帧或训练覆盖数。直方图分位数近似语义保留。图像增强关闭，dataset.use_imagenet_stats=false；实际读取仍按原规则/255。策略特征仅为Joint9状态及双路masked-depth输入、Joint9动作输出，phase／门角／接触及相机位姿不进入策略。

每回合局部timestamp字段仍遵循作者float32存储；保留的float64修复针对容器偏移、绝对查询和原生PTS运算。状态／动作／图像和Torch默认类型仍为float32，没有全局dtype改变。

## 资源、异常与封存

全部转换和数值检查在pro5000既有door-act环境CPU执行，隐藏CUDA，限制8个CPU核，OMP／MKL／OpenBLAS各2线程。Python3.10.22、Torch2.7.1+cu128、NumPy2.2.6、PyAV15.1.0、TorchCodec0.4.0；实际LeRobot导入自项目high-level/lerobot/src。未安装、升级或重建依赖。

启动前确认没有项目数据读取／转换冲突。先记录SHA并删除8个可由既有Parquet重建的HF Arrow缓存，逻辑37,521,512字节；未删除原数据、历史证据、权重或/data/zyf/video2door支持环境。新run按32GiB输出／临时／缓存增量及40GiB余量估算，启动余107.49GiB；10秒容量采样最低98.50GiB，读后封存快照98.42GiB。后续最终快照见freeze-receipt；未触发容量停止。

五项主流水线退出码均0，没有数值或系统性数据异常、跳过样本、改容差、换解码后端或转换重试。外围scp／git网络沙箱限制及一次错误工作目录的只读查询均保留在COMMANDS.md处理记录，未改变数据或造成物理重跑。

dataset-SHA256.json逐文件绑定8个数据文件；其SHA为d46b4515122f6bc64f8da88a9bdee624aff1dbc4c212023a51090aa6cc0e92c7。新meta/stats.json SHA为d01787fe611e93cc1f96e9dd3a50f8d62c13403e9d8651c5df02369c0ca3b9d6。完整无损数据归档SHA为ba0c154e551ee3595cb7bc4a958755bfc28d365fff733c8d7c116c2c176ed6d9。服务器及本地另有全量SHA索引和归档完整性回执。

证据限制：本轮完整任务资格继承用户已验收的第七轮50条来源，没有新执行物理回合或重做R11录像审查；旧物理大日志未在本轮重新逐条评分。没有构造ACT模型／优化器、加载旧策略、推理或训练。采集、仿真、策略推理、训练更新均0。运行内CPU监控随流水线退出，未持有GPU锁；封存后停止，由用户转回方案对话独立验收，再制定ACT训练任务书。
