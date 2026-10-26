# 第三轮：单条候选的停车、部署与干涉修正

执行侧完成唯一一条完整候选及只读检查，冻结待独立审查。停车和接近衔接得到改善，但全程几何仍需修正／存在未证部分，不进入物理试运行。没有第二条完整路线。几何、连续性与合成跟踪分别报告，不把参考轨迹完成当作完整任务成功。

## 相对 c2 的修正

- 全身通过前保留旧实测前缀；之后沿原直线到同一配置终点(1.259999752,0.180000007)m，仅重排时间。用端点速度/加速度约束的五次位置函数，在7.16s内从原0.2000033855m/s减至0，无越点。
- 既有c2虚拟DLS关节序列按空间进度作单调PCHIP时间重参数化，再保留原3rad/s指令界；实际起点、前一真实指令和合成状态分开记录。该模型没有重新运行真实开门或物理跟踪。
- 接触半径0.88→0.905m，高度仍0.85m，初始转向仍270°；接触为原固定指box 2的+x/+y棱，法向在其实体外法向锥内变化。原门板三角形40包含该接触参考点。
- 4cm预接触退距局部求解失败后，保持原8mm/0.03rad误差门槛，选用1cm退距；以一条平滑关节弧线从home部署到已求解近接触姿态，随后短程法向接近，取消旧部署过程中同时大幅扭转的方式。
- 在推关前显式检查位置/法向误差、关节/底盘速度、远侧以及非接触部位/自碰撞源几何间隔。失败则跳过close/settle/release并标记，不按时间强行推关。本次该门通过。
- 底盘在停车终点保持至推关；仍在预设门角35°以下、朝向已回正后平移0.45m。本轮不减量。回合底盘z始终0.5m；接触高度不等于底盘高度。

## 分项结果

- **停车连续性**：终点位置误差 0m；无越点=True；首个离散速度与原速度之差 -1.55848e-06m/s，最后移动区间速度 1.55822e-06m/s，随后0；解析起/止加速度分别 0/0m/s²。离散区间平均速度不等于瞬时端点速度。
- **关节与底盘**：新后段指令最大关节速度 0.7rad/s，继承段 2.99994rad/s；侧向速度残差 3.25e-17m/s。新后段最大指令二阶差分 34.7055rad/s²，未声称全程关节速度C1或真实驱动平滑性。
- **合成跟踪**：接近门通过=True；实际计算到推关=True。参考总时长 172.480s，相比c2的180.400s缩短；90s推关段没有增加。累计推关底盘平移 0.45m，最大位移 0.45m。
- **几何**：检查 10747 个原始/插值姿态；外部/自碰撞/远侧未证区间分别 852/570/0。采样远侧返回数 0；包含实测锚点的最小远侧边界余量 0.000552296m。证明仅限所声明的源包络与分段线性运动，不代表物理轨迹。

|阶段|最大位置跟踪误差 m|最大法向误差 rad|最小关节命令范围余量 rad|最大指令关节速度 rad/s|
|---|---|---|---|---|
|retimed_traverse|0.582231|0|0|2.99994|
|endpoint_hold|0|0|0|0|
|withdraw|0|0|2.22045e-16|0.7|
|fold|0|0|0|0.653887|
|turn|0|0|0|0|
|deploy|0|0|2.22045e-16|0.551944|
|seek|0.00174117|0.012519|2.22045e-16|0.651133|
|close|2.92669e-05|0.000211068|4.44089e-16|0.18369|
|settle|0|0|0.0186359|0|
|release|0.00643047|0.0630015|2.22045e-16|0.7|
|retract|0|0|0|0.561789|
|restore|0|0|0|0|
|hold|0|0|0|0|

关节弧线阶段没有笛卡尔跟踪目标，表中0表示不适用，不代表端点残差自动为零；撤离端点解见withdraw-endpoint.json。原始测量锚点中跟踪误差与指令范围差异保留，不裁剪旧实测。

## 口径和几何疑点

远侧检查与运行时同样纳入全部collision及visual包围几何。初始预检查发现名义腿目标会使边界相差约2.829mm，因此在完整候选生成前，从旧实测刚体父子旋转恢复腿/轮关节姿态；起点复算与运行时max_x相差约7.7e-8m。此后腿姿态冻结仍属合成假设，未来需用实际腿关节状态检查。
自碰撞只排除同一固定组件及直接关节相邻组件；其余相关运动部件全部纳入，link02/link06的既有home重叠没有排除。外部源凸包、球外包盒和圆柱外包多边形重叠按实际部件逐项定位；封闭原网格用两种射线，源box/cylinder/sphere用原始实体体积核对。无内部采样点不证明无交叠。

`intended`仅标识预选接触基元，并不自动表示该时刻接触合法。部署/转向/收臂/撤离中的预选基元重叠单独保留；按阶段分别定位，不能让更早的继承接触掩盖新后段问题。

|类别|部件对|最坏源包络间隔 m|时间 s|原几何内部采样点|说明|
|---|---|---|---|---|---|
|external|wrist_down_mount:0 / door_panel:door_slab|-0.161969|17.765|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|wrist_down_mount:0 / door_panel:glass_panel|-0.0854412|17.760|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|wrist_down_mount:0 / lever_handle:long_grip|-0.0644534|17.700|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|wrist_down_mount:0 / handle_back_plate:back_plate|-0.0488755|17.672|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|wrist_down_mount:0 / door_panel:glass_gasket_left|-0.0470749|18.540|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|wrist_down_mount:0 / door_panel:glass_gasket_right|-0.0451724|19.150|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|gripperMover:2 / door_panel:door_slab|-0.0430378|19.030|125/125|source-geometry intersection supported at this synthetic pose|
|external|wrist_down_mount:0 / lever_handle:short_return|-0.0396361|19.420|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|wrist_down_mount:0 / handle_back_plate:handle_hub|-0.0388471|17.672|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|gripperMover:0 / lever_handle:long_grip|-0.0357646|18.790|0/125|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|gripperMover:1 / door_panel:door_slab|-0.0346017|19.300|95/125|source-geometry intersection supported at this synthetic pose|
|external|wrist_down_mount:0 / door_panel:door_vertical_grain_11|-0.0319615|17.682|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|wrist_down_mount:0 / door_panel:door_vertical_grain_10|-0.0308083|19.130|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|wrist_down_mount:0 / door_panel:latch_edge_plate|-0.0303011|19.860|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|gripperMover:1 / lever_handle:long_grip|-0.0296305|17.722|12/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:0 / lever_handle:short_return|-0.0249372|19.150|0/125|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|gripperStator:3 / lever_handle:long_grip|-0.0243206|19.010|1/128|source-geometry intersection supported at this synthetic pose|
|external|wrist_down_mount:0 / door_panel:door_vertical_grain_9|-0.0230972|18.220|0/7343|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|gripperMover:1 / handle_back_plate:back_plate|-0.0207996|19.110|16/125|source-geometry intersection supported at this synthetic pose|
|external|gripperStator:2 / lever_handle:long_grip|-0.020495|18.590|10/125|source-geometry intersection supported at this synthetic pose|
|external|gripperStator:2 / lever_handle:short_return|-0.0202522|18.540|10/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:2 / door_panel:glass_panel|-0.0201153|17.820|45/125|source-geometry intersection supported at this synthetic pose|
|external|gripperStator:3 / lever_handle:short_return|-0.0199505|19.100|0/128|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|gripperMover:2 / lever_handle:short_return|-0.0197361|17.720|12/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:1 / door_panel:glass_gasket_right|-0.0176778|18.700|10/125|source-geometry intersection supported at this synthetic pose|
|external|gripperStator:1 / lever_handle:long_grip|-0.016883|18.660|8/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:2 / door_panel:glass_gasket_right|-0.0167253|18.690|45/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:1 / lever_handle:short_return|-0.0161318|17.715|7/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:1 / door_panel:latch_edge_plate|-0.0152474|19.440|6/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:1 / door_panel:door_vertical_grain_10|-0.0150319|18.750|0/125|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|wrist_down_mount:0 / door_panel:door_slab|-0.0139191|70.520|441/7343|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:2 / door_panel:latch_edge_plate|-0.013189|19.440|11/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:1 / door_panel:door_vertical_grain_11|-0.0123823|19.050|0/125|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|gripperMover:2 / lever_handle:long_grip|-0.0120351|17.712|6/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:1 / handle_back_plate:handle_hub|-0.0108017|19.100|2/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:2 / door_panel:door_vertical_grain_10|-0.0106758|18.730|1/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:2 / handle_back_plate:handle_hub|-0.010365|17.727|7/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:0 / handle_back_plate:handle_hub|-0.010077|19.140|3/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:2 / handle_back_plate:back_plate|-0.00890381|17.732|6/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:1 / door_panel:glass_panel|-0.00875957|17.810|9/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:0 / handle_back_plate:back_plate|-0.00854433|19.200|8/125|source-geometry intersection supported at this synthetic pose|
|self|link06:0 / link02:0|-0.00832464|34.620|13/360|既有home部件对；未豁免|
|self|link06:0 / link02:0|-0.00832464|158.820|13/360|既有home部件对；未豁免|
|self|link06:0 / link02:0|-0.00832464|34.280|13/360|既有home部件对；未豁免|
|self|link06:0 / link02:0|-0.00832464|158.280|13/360|既有home部件对；未豁免|
|self|link06:0 / link02:0|-0.00832464|170.300|13/360|既有home部件对；未豁免|
|self|link06:0 / link02:0|-0.00832454|44.300|13/360|既有home部件对；未豁免|
|external|gripperStator:1 / lever_handle:short_return|-0.00820281|18.680|1/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:0 / door_panel:door_slab|-0.00647177|19.340|7/125|source-geometry intersection supported at this synthetic pose|
|external|gripperMover:2 / door_panel:door_vertical_grain_11|-0.0054764|17.735|0/125|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|intended|gripperStator:2 / door_panel:door_slab|-0.00332539|50.840|3/125|source-geometry intersection supported at this synthetic pose|
|self|wrist_up_mount:0 / link02:0|-0.00286982|159.720|99/4956|既有home部件对；未豁免|
|self|wrist_up_mount:0 / link02:0|-0.00286982|35.320|99/4956|既有home部件对；未豁免|
|self|wrist_up_mount:0 / link02:0|-0.00286982|34.280|99/4956|既有home部件对；未豁免|
|self|wrist_up_mount:0 / link02:0|-0.00286982|158.280|99/4956|既有home部件对；未豁免|
|self|wrist_up_mount:0 / link02:0|-0.00286982|170.300|99/4956|既有home部件对；未豁免|
|self|wrist_up_mount:0 / link02:0|-0.00286975|44.300|99/4956|既有home部件对；未豁免|
|external|link06:0 / lever_handle:long_grip|-0.00258111|19.440|0/360|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|gripperStator:0 / lever_handle:long_grip|-0.00120076|18.790|0/125|source envelope overlap remains unresolved; absence of contained samples is not separation proof|
|external|gripperMover:1 / door_panel:door_slab|-0.000534545|77.920|1/125|source-geometry intersection supported at this synthetic pose|
|intended|gripperStator:2 / door_panel:door_slab|-2.7965e-05|56.280|6/125|source-geometry intersection supported at this synthetic pose|
|intended|gripperStator:2 / door_panel:door_slab|-2.79648e-05|56.300|6/125|source-geometry intersection supported at this synthetic pose|

## 边界与停止

这轮没有物理关门、接触力、真实锁存或最终共同保持证据。门角在停车/部署时固定于旧首过点实测角；接近门通过后按预设曲线关闭，不能当作推关动力学反馈。实际q/qd、门角、接触、制动和释放仍待未来实测；本轮不进入物理试运行。
1条完整候选额度已使用，不换参数重跑。生成前仅在两个既有失败姿态及其接近邻域作43个局部端点求解，并检查1条局部部署弧线；没有用局部名义筛选多条完整路线。全部局部失败与最初导入错误保留。
物理、资产、R11、ACT与观测及运行专家源码保持不变；只新增独立CPU工具。未增加推关前底盘移位、未改穿越空间路径/终点、未恢复绕行。无仿真、录像、采集、转换、训练或策略推理，无GPU锁或后台监控。
正式结论以decision.json、geometry-summary.json、overlap-localization.json及全部未证区间为准。确定的非接触部位干涉或关键未解部分不得被低末端误差掩盖；本次冻结交付仅供方案对话独立审查。

## 局部解与连续解的差异

在门角相差-3.64028e-05rad的两姿态中，完整候选与局部全姿态解的工具旋转差为0.0521462rad，其中沿接触法向分量0.0521462rad。局部测试约束全姿态；连续求解只约束接触点和法向，剩余旋转自由度没有被净空约束控制。因此不能把局部静态净空直接延伸为整段安全证明。精确位姿、关节及净空见local-vs-continuous.json。

## 直接邻接运动部件补充检查

主审查排除的直接关节邻接组件已另外补查：相对几何只由其连接关节决定，因此在原轨迹全部不同关节值上作规范相对坐标检查，不生成第二条路线。固定装配内部部件仍单列清单，不误当作相对运动碰撞。
|连接关节|原序列不同参数值数|部件对子数|未证参数区间数|
|---|---|---|---|
|joint3|6456|6|69|
|jointGripper|4|21|3|

这里的参数区间不是主审查的时间区间，不能相加。负包络对子均已定位到原轨迹时刻，并用原始实体再查：

|源部件对|首次对应阶段|源间隔 m|实体内部采样点|
|---|---|---|---|
|link03:1 / link02:1|fold|-0.00029101|0/360|
|gripperMover:0 / wrist_down_mount:0|retimed_traverse|-0.000165494|0/125|
|gripperMover:0 / gripperStator:3|measured_anchor|-0.030988|42/125|

邻接关节装配／已有夹爪姿态的交叠不自动认定为本轮新增物理碰撞，也不从证据中删除；没有用过滤器改动物理。详见adjacent-self-summary.json、adjacent-self.jsonl与adjacent-self-localization.json。
