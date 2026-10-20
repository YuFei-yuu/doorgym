# 成功基线的检查实现参考

保留最终成功训练使用的有限值守卫、模型导出一致性、GPU worker检查、W&B监控和日志/选片方法，供复现手册引用。不是可直接启动的历史run。

文件中的ROOT、RUN、SOURCE、HEAD、模型SHA和训练步数是原执行示例常量。复制到新run时按实际版本、父checkpoint及预算替换；首训不用resume逻辑。不要尝试打开已清理的旧父run，也不要改作者模型/数据/物理来通过检查。

- `guarded_train.py`：作者训练调用前后的有限值、恢复状态与进度检查。
- `export_and_check.py`、`check_gpu_bridge.py`：真实数据预处理、导出权重、设备输出一致性。
- `guarded_policy_worker.py`：输入和动作有限值检查。
- `check_batch.py`：从原始角度日志重算成功率及代表录像绑定。
- `common.py`、`launch_train.py`、`watch_wandb.py`：GPU租约、作者恢复、在线监控实现参考。

所有旧结果、失败分析和中间模型均不在此目录。
