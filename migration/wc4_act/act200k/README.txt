ACT50k→200k执行入口：prepare.py(隐藏CUDA只读验证父/数据与复制固定初态)、orchestrate.py TRAIN_GPU EVAL_CANDIDATES(调用launch/start/train；六节点导出/评测/只读验收；训练退出后冻结)。
独立运行目录：.repro/runs/wc4-act200k-20261029
独立冻结交付：.repro/reviews/wc4-act200k-20261029-FINAL
本地交付：/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act200k-20261029/FINAL
所有授权范围、保留契约和恢复边界见SCOPE.txt。新run与固定父checkpoint绑定见parent-checkpoint-binding.json。
freeze.py仅在完整150000新增更新和96回合均完成且逐节点审核通过后发布只读交付；任何异常保留现场，不追加预算。
