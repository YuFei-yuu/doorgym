ACT200k→300k独立执行入口，按本轮用户任务书适配自act200k，不直接启动历史脚本。
prepare.py：隐藏CUDA，核对服务器200k父文件与SHA、数据、容量与原16初态，不训练或仿真。
orchestrate.py TRAIN_GPU EVAL_CANDIDATES：调用launch/start/train，原作者全状态恢复；五节点导出/16回合/只读核验；训练退出后freeze.py发布只读交付。
独立运行：.repro/runs/wc4-act300k-20261030
完整服务器交付：.repro/reviews/wc4-act300k-20261030-FINAL
本地轻量审查包：/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act300k-20261030/FINAL-METADATA，非完整镜像。
恢复证据见parent-checkpoint-binding.json、inherited-recipe-check.json、trainer-attempts/*/restored*.json；首个新增更新应为200001。
冻结需完整100000次新增更新、80回合以及五个stage-audit全部通过。详细契约与停止边界见SCOPE.txt。
