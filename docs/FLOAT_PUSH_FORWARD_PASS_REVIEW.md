# 直接向前通过修复

用户已确认采用该方案，修复提交为 d2905f1。

原全身入口达到80°后，先导航回门前对齐点，再前往门后目标；这会在浮动底座已推到门平面时造成两次接近180°转向。另外，按开门时当前位置确定通过侧，可能在已越过门平面时把目标反向。

forward模式按初始接近侧确定门后目标，80°后直接进入pass_through。原导航函数、80°门槛、0.12m容差、snap_arm_to_zero、IK/门物理及DP覆盖顺序保留。浮动入口默认forward，原四足入口默认source。

同seed14400对照中，开门前状态/动作逐项一致；两者22.56s达到80°。修复后25.64s完成通过，原版40.32s；80°后反向行程由0.931m降为0。后续[18回合验证](FLOAT_PUSH_FORWARD_EXPERT_VALIDATION.md)全部成功。

完整前后对比保存在：
- 服务器 experiments/float-push/20261013-forward-pass-review/
- 本地 reproduction/pro5000/forward-pass-review-20261013/
- 并排视频 before_after_seed_14400.mp4，左原版、右修复版。
