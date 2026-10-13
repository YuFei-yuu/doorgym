# 项目任务启动器

launch.py采样GPU三次，间隔2秒，选择空闲显存至少16GiB、平均利用率低于70%的卡。子进程使用项目Conda、单GPU、nice10和两线程。

```bash
python high-level/float_ik/learning/run_tools/launch.py \
  --root <本次run目录> --stage <唯一阶段名> \
  --timeout <单任务秒数> --total-seconds <本次run秒数> \
  -- python <入口> <参数>
```

训练使用 --environment train。bounded.py记录当前run的进程与耗时；其timeout和budget_limits.json会实际终止超时任务。用户给定的训练时间预算仅作预估，下一轮编排应据此设置运行参数，避免把估算时间直接作为正常训练的硬截止，不继承旧实验上限。

python test_budget.py 检查超时、记账、互斥和中断恢复。环境说明见[环境配置](../../../../docs/环境配置.md)。
