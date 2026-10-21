# R8执行后的只读审计脚本

这四个脚本是本轮实际运行副本，固定指向服务器R8目录；不是下一轮采集入口。
`check_dataset.py`使用真实训练factory及DataLoader，只创建ACT配置，不构造模型。
它以独占方式创建`dataset-check`，不可直接覆盖已交付结果。
`check_alias.py`和`plot_diagnostics.py`读取前述结果；`review_trials.py`读取原始执行日志和录像。
执行环境：dataset检查用既有ACT环境，其他数值/绘图检查用既有Gym环境CPU。
原始运行副本、早期诊断副本及日志仍随run封存，不以这些副本替换原证据。

最终结果和限制见相邻的`FULL_TASK_PILOT_R8_RESULT.md`。
