# R18 原配方50k→100k，固定h25

用户授权：原R15真实050000完整状态恢复，新增50000更新到100000；075000/100000各16固定62200–62215、每回合≤180秒，horizon25，不重跑50k，不补烟雾/替换失败/追加64或续训。原72/159529、seed1000、ResNet18默认ImageNet预训练、batch16、lr1e-5、chunk100、训练n_action_steps50、原float32及R10 float64时间戳比较修复不变。

复用已验收R13停止与原作者训练器；恢复后的模型234项、optimizer459项及153个step50000、参数组、全部RNG含CUDA、归一化逐项比较，之后才更新。DataLoader sampler/prefetch游标未保存，不声称样本序列精确连续。只将保存触发改为25000以输出所指定75k/100k；优化配方不变，cfg.env=None，无原生额外环境评测。单run与共享UUID锁，不自动恢复。

rollout.py从原R13 **h25**入口复制，继续使用原R13 ActionJournal和R14 h25实际校验器；未从R17 h5 runner复制。仅加被动耗时和当前stage权重/统计/初态相机绑定。worker显式assert horizon25、chunk100；真实回执每段25动作，25Hz/50Hz不变。CPU fixture通过真实原RPC/队列验证h25，0真实模型/训练/场景。导出沿用已有5固定索引，原官方/Door权重、统计、预处理/物理动作一致性检查，不新增闭环。

全部32整体RGB/状态/接触/原始深度/worker应用证据保留；每个已结束回合独立CPU raw audit通过后才执行下一回合。几何远侧不认证真实通行，穿模疑点单列原片复核，不加接触否决门槛。成功候选与审核成功分开；每阶段原片成功/失败各一，缺类不补跑。75k结果不改数据/统计/optimizer/模型选择；父trainer RNG在整个评测子进程前后核对相等。W&B主run及两个独立stage eval run显式互绑checkpoint，吞吐记录真实batch样本量，保留资源和异常原始日志。

监控/资源/证据错误使用STOP_REQUEST保存一致完成边界；若update内硬故障不能证明一致，保留UNSAFE_STOP/最近完整checkpoint并披露，不假称新快照可恢复。真停止必须回报并等人类，不因工程修复自行重新开训。达到100k和32回合后冻结交付，等待独立验收。
