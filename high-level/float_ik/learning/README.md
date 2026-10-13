# 浮动底座DP入口

先读[交接文档](../../../docs/FLOAT_PUSH_TRAINING_NEXT.md)。当前forward专家已通过9组位置×2seed的18回合验证；该方案尚未训练DP。

## 当前路径

- wholebody_float.py：原全身任务的浮动后端。使用 --float_execution source；通过模式默认forward，可显式切换source。
- validate_forward_expert.py：18回合专家位置网格验证。
- collect_v3.py：批量采集模板；种子表仍为旧实验值，新采集时配置新的集合。
- high-level/dp/convert_door_raw_to_lerobot.py、train_door_dp.py：原数据转换和训练入口（仓库路径为 high-level/dp/）。
- checks.py、dataset_contract.py：数据、统计和checkpoint检查。
- offline_eval.py、analyze_v3.py：原控制器离线分析及执行曲线。
- run_tools/launch.py：单GPU选择、项目环境、进程运行时限与记账。
- run_tools/activate_train.sh：独立训练Conda激活。
- watch_training_plot.py：读取训练日志生成曲线。

finish_v3.py、resume_eval_v3.py、deliver_v3.py保留旧实验编排供参考，其集合与目录定义不作为新训练的默认计划。通用FK、原序列适配、分析和视频工具仍保留。

## 数据约定

真实27D状态、原10D末端动作、四路54×96图像、50Hz；schema为 float_push_upstream_ee_v3。当前 execution_profile 为 upstream_wholebody_float_v3_forward_pass，旧source模式保留独立profile。

任务/导航 → DP覆盖 → IK/底座执行顺序及源snap处理保留。训练继续使用原LeRobot读取及train-only归一化。

## 相关资料

[环境](../../../docs/环境配置.md) · [执行方案](../../../docs/FLOAT_PUSH_REPRODUCTION_PLAN.md) · [指标](../../../docs/FLOAT_PUSH_METRICS.md) · [专家验证](../../../docs/FLOAT_PUSH_FORWARD_EXPERT_VALIDATION.md)

CPU回归：python -m unittest test_forward_pass test_analysis。资源工具检查：python run_tools/test_budget.py。
