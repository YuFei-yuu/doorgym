# 第三轮：单场景只读属性检查交付

日期：2026-10-20（Asia/Shanghai）。**属性快照已完成；完整任务仍未获启动认证。** 本轮创建场景1次、环境1个，`simulate`、`fetch_results`、轨迹调用均0次；初始化封闭后setter为0，场景已销毁、GPU租约已释放。没有重建场景、执行轨迹、训练、推理或改动时间约束。

## 版本与边界

- 按用户指定，从 `7be9882c5c532cd5adf30f17a07be6b319ee757d` 建立独立分支 `expert/wc4-scene-properties-r3-20261020`。main仍为 `87c5a86`。
- 唯一场景执行提交：`b3951e5d12143885c2ced716b001f5dea8dae50f`；成功CPU分析提交：`a672ecf4cdc245465121889d93bdaf01a5c3c88d`。后续防重复启动修正没有再次创建场景。
- 新路线直接使用独立审查的 `geometry-compact.json`，SHA256：`872dd74761d51bd241caa89682ccff285321a70ded53d3bc0988cd80fd16be47`，未重算或执行路线。
- 服务器输出：`/data/zyf/video2door-wc4-act/.repro/runs/wc4-fullcycle-properties-r3-20261020`。
- 本地交付：`/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-fullcycle-properties-r3-20261020`。
- 使用pro5000原Gym专用环境，物理GPU2，UUID `GPU-2c8bfc35-6335-164d-bba2-5c97c3d9cfd0`。启动前三次检查空闲并持有原共享UUID锁；CUDA/Vulkan逻辑0。后处理隐藏GPU、仅CPU执行。
- 没有修改home、资产、物理、碰撞过滤、观测/动作语义、原路线控制或学习方法。前两轮封存结果及已验收产物保留。

场景构造复用了作者的资产加载、名义初始化和原属性设置。**构造阶段存在原初始化setter**：包括DOF属性、初始状态和初始静态目标，全部保存于 `initialization-writes.jsonl`；它们不是轨迹序列，未经过步进。随后代理封闭所有场景setter/创建器，读取属性，最后只调用销毁。不能把本轮描述成“从构造开始从未调用任何setter”。没有调用作者main、IK setup/求解、专家或策略循环。

## 直接读到的形状、索引与过滤

| Actor | 刚体数 | 实际形状数 |
|---|---:|---:|
| A2W base | 20 | 43 |
| Z1 arm | 13 | 15 |
| wc4 door | 3 | 56 |
| 左侧墙 | 1 | 0 |
| 右侧墙 | 1 | 0 |

114个actor形状槽位被各刚体的IndexRange恰好覆盖一次。侧墙没有碰撞形状，因此墙体的几何禁入检查不能冒充承力墙碰撞证据。

关键映射（形状范围是**actor内**索引）：

| 刚体 | 环境刚体索引 | actor形状start/count |
|---|---:|---:|
| link02 | 23 | 0 / 3 |
| link06 | 27 | 5 / 1 |
| gripperStator | 28 | 6 / 4 |
| fixed_frame | 33 | 0 / 22 |
| door_panel | 34 | 22 / 32 |
| lever_handle | 35 | 54 / 2 |

刚体handle、actor局部索引、环境索引、simulation索引均逐项保存。在本单环境中handle/env/sim的数值对应不能推广成多环境索引规则。首轮相关body ID与本次body名称/actor映射相符。

读回的全部形状filter均0；关键对按 `(filterA & filterB)==0` 规则允许碰撞。创建参数group/filter也是0/0。SDK属性接口未给出全部native articulation逐对排除项；但首轮该自接触对有非零记录，不能用“它已整体被过滤”解释该历史样本。

`door_panel`包含32个形状，不等于仅一个门板平面；`gripperStator`有4个形状。仅用body名出现过接触，还不能证明碰到了计划门板区域。

## 接触/静止偏移与接口限制

实际传给SDK创建器的全局参数：dt≈0.02s、substeps=2、`CC_ALL_SUBSTEPS`，contact_offset≈0.02m、rest_offset≈0.001m。原求解器和所有覆盖参数均保存在 `sim-parameters.json`，没有修改。

**不能宣称已读回每个形状的contact/rest offset。** 本绑定实际返回的RigidShapeProperties只有filter、friction、restitution、compliance、rolling/torsion friction、thickness；直接属性探测也确认没有contact_offset、rest_offset、geometry、type、local_pose。文档中列出的字段不等于本绑定已实现字段，`RigidContact.dtype`类属性同样未暴露。

读回的shape.thickness为0；导入AssetOptions.thickness分别有原设置值。这些层次不能互相替代，也不能把全局参数冒充逐shape覆盖后的实际值。参考的两体全局rest和约2mm、contact和约40mm，只是解释用参考量，尚非逐形状认证。

没有可用于本轮认证的刚体cooked顶点/每shape局部变换导出。`get_sim_triangles`等接口针对软体/可变形三角形，不能拿来冒充刚体碰撞网格。SDK列有USD导出入口，但本轮未验证其是否包含所需cooked碰撞形状及一一映射，因此既不宣称它可替代认证，也不宣称所有导出途径绝对不存在。

## home：目标、读回与实测须分开

下表都使用原源圆柱的128边形内/外包络，不是直接导出的cooked形状距离：

| 对象 | 内包络间隔 | 外包络间隔 |
|---|---:|---:|
| 六轴全0目标的源FK | −8.010mm | −8.030mm |
| 本次未步进SDK刚体位姿放置源圆柱 | +2.252mm | +2.232mm |
| 既有专家episode000000首帧实测q的源FK | +2.090mm | +2.070mm |
| 同一既有专家末帧实测q的源FK | +1.959mm | +1.940mm |

既有专家首、末六轴命令都是0，但实际q不同。末帧joint3约−0.025164rad，最大六轴目标误差0.025164rad；首帧最大误差0.031914rad。这个跟踪偏差足以改变该对源几何的间隙，所以“目标相交”和“实测分离”不矛盾。末帧误差小于0.03rad只是单帧事实，未因此重验或宣称通过完整任务的2秒保持、速度或碰撞条件。

首轮名义执行的原始接触日志中，link02—link06共有2567条记录，其中453条lambda非零（walk 450、initial_hold 3）；**这不是2567次碰撞事件**。示例第0步lambda≈8.805，积分后实测q对应源外包络仍分离约1.915mm。这说明实测分离可以与求解器接触记录同时存在；与静止间距约束及位置跟踪偏差相容，但本轮未标定lambda单位或证明唯一动力学因果。

该非零接触证据来自**首轮执行日志**；既有已验收专家样本仅用于读状态/动作，没有把首轮力记录移植到该样本。不能因此宣称已验收50条全部存在或不存在同样受力。

本次未步进的六轴DOF缓存仍是0，但SDK刚体位姿与该DOF的源FK最大相差14.885mm、0.027642rad。原因尚未唯一确定（初始化/读回同步或其他导入语义需验证）；不能称为已测得的稳态home，更不能写状态让其吻合。因此本轮**没有“必须修改home”的充分结论**，也没有给home无自接触认证。

这一差异是在场景销毁后的CPU分析中发现的，本次没有追加刷新张量后的同步读回探针；这是检查覆盖缺口，不能归咎为SDK必然不支持同步。场景额度已用尽，因此不再创建场景补采。

若未来把该对已知home接触列为允许项，需要明确审查其条件；本轮不添加默认豁免、不扩大自碰撞过滤，也不修改home或资产。

## 坐标映射与首轮接触点

SDK的COM是body局部Vec3，未导出完整COM/principal-frame姿态。body-state.pose与get_rigid_transform逐体一致；导出的body局部COM通常非零。派生世界质心位置应写成 `p_com_world = p_body + R_body * com_body`，不能把COM平移重复加到body原点上。

在首轮569步两条夹爪—门框记录上，使用历史arm body位姿和同名义门框固定姿态重建：

| 局部点解释 | 两条记录变换后点间距 |
|---|---|
| 两侧都按body局部 | 约0.260mm / 0.0023mm |
| 仅夹爪额外加COM | 约47.77mm / 47.71mm |
| 仅门框额外加COM | 约1.286m / 1.287m |
| 两侧都额外加COM | 约1.282m / 1.282m |

这不支持“简单补上COM平移就能修正”的解释。门框历史body状态未保存，使用了同名义固定actor初始化；所有假设和局限已在 `r1-contact-coordinate-hypotheses.json` 明示，没有伪造历史属性。

两条记录力值均0；局部点变换后接近重合不等于两源碰撞表面间隙为0。此前源几何分析约39.116mm，与40mm的全局接触生成距离参考相容。minDist/initialOverlap也不能直接当作几何距离或穿透深度。

更可能的候选解释包括公共接触点表示、子步生成时刻与积分后位姿不同、导入形状与源几何映射差异；**尚未唯一认证**。本次配置收集两个子步，而原记录没有shape ID、子步ID或逐记录时间戳。不能按数组顺序强定时刻，也不能把所有接触与20ms末帧位姿硬配。简单COM误解可排查，但力臂、法向方向/单位及实际穿透判定仍缺校准，继续保持未认证状态。

## 运行接入与停止结论

具体接入点、数据合同及替代验证办法见 `INTEGRATION.md`。当前`classify_contact`和`ClosingEvidence`仍未在实际循环完成认证接线；仅保存这些属性不会让门槛自动通过。完整任务入口的认证标志不得手工改为true。

后续需要：

1. 获取或验证每shape几何类型、局部变换、cooked几何和逐shape偏移；不能用源UF或全局值补作“实测”。
2. 验证DOF/body读回一致性以及contact字段、子步时间、force/impulse单位、法向和施力点定义。
3. 在明确授权的实际循环中接入区间收证，确认末端作用于门板、实际门运动和功/力矩方向；按原关闭保持门槛锁存实测角，释放后同时验证全部2秒条件。
4. 独立审查已知home自接触如何分类，不能默认忽略；若需改home、资产、物理或碰撞语义，另行报告确认。

本轮到此停止交独立审查。新路线的86.26秒只是先前审查的运动学参考，本轮没有专家运行时间事实，因此没有改90秒或其他阈值。用户允许后续基于实际超时情况调整时间，不等于本轮可越过只读范围或自动获得闭环回合。

## 证据与工程记录

- 原始：`actor-properties.json`、`asset-properties.json`、`asset-imports.json`、`actor-creation-arguments.json`、全局参数、有效args、初始化setter及API审计、一次创建凭据。
- 资产：`generated-assets/`、`imported-assets/`实际UF/网格/可用材料闭包、源路径/SHA、原VHACD缓存副本与provenance。cooked缓存未被猜测解码成认证几何。
- 分析：`analysis/`下规范化属性、索引、过滤、COM/FK、各home案例、首轮接触假设、自接触原始记录计数及示例。
- 代码：执行源快照、完整差异、Git bundle、实际命令、GPU绑定、环境及总SHA清单。

初次CPU分析有一处括号语法错误，失败日志保留于 `analysis.log`，修正后另存 `analysis-v2.log`，**没有重建场景**。原始pybind序列化包含冗余dtype描述；规范化视图另存，原始快照不改。类属性探测发现文档声明的dtype不存在，改为明确记录缺失，没有补造字段。

另发现源夹爪OBJ引用的 `gripper_fix_90d.mtl` 文件原本缺失，保留了缺失清单，没有新建材料或改资产。所有引用的几何文件已归档，不能据此声称可视材料依赖全部完备。本轮没有渲染或动态录像，属性快照不是执行证据。
