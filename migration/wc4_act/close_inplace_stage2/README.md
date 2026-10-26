# 固定穿越终点：第二轮 CPU 候选检查

本目录只实现离线候选、源资产几何检查和已有输出汇总，不接入运行专家或物理仿真。

本轮用户范围：名义场景、隐藏 CUDA、最多三条完整候选（包含 0.45 m 起始方案）；保持配置穿越终点及单回合高度，核对合成交接、连续 IK/限位/限速、无侧向底盘运动、源资产全身外部净空、释放收臂及恢复朝向。不能以规划矩形代替资产；不能把源包络重叠直接称为物理穿透。不得增加预推关移位、恢复绕行或改变受保护内容。

独立输出：服务器 `.repro/reviews/wc4-close-inplace-stage2-20261026`，本地 `reproduction/wc4-act/wc4-close-inplace-stage2-20261026/FINAL`。目录日期是启动日期；本轮跨至北京时间 2026-10-27 冻结。

## 工具与证据语义

- `candidate.py`：从 stage4 实测首过点继续原线性穿越目标，使用源几何解析 Jacobian 和原 DLS 参数进行理想跟踪推算；其后生成独立后段。合成门角是规定参考，不是动力学反馈。每条方案保存完整尝试，失败后后缀仅用于诊断。
- `audit.py`：源 URDF 碰撞子基元、凸包和有限墙体；0.1s/阶段边界及不确定区段 50Hz 加密。给出保守分离下界与未证区段，不做 cooked PhysX 或完整自碰撞认证。`fast_gap`只省去重复 SAT 轴的去重及向量化分类，不改变候选。
- `summarize.py`：读取既有结果，定位重叠并生成比较表/图，不能启动新路线或 IK 搜索。

`c0/c1/c2`的配置和执行提交分别冻结于各自 `receipt.json` / `config.json`；不要用最终配置覆盖早期执行绑定。运行前的 `c0`记录入口异常未进入任何轨迹计算，保留在 `infrastructure-01`。第一版未完成的几何审查在 `audit-v1-incomplete`；仅终止本轮自有检查进程后用等价优化工具重查相同轨迹，没有重跑三条候选。

真实记录在 `measured-prefix.jsonl`；合成连续尝试在 `trace.jsonl`。两者分开保留，不作为训练数据，不宣称物理关门成功。第三条候选（c2）的12个静态接触 IK 初值诊断不是12条完整路线，原解全部保留。

## 复核命令

仅在既有 pro5000 Gym 环境运行；本轮已经耗尽三条完整候选额度，以下为复核说明，不是新增授权。需要重建应使用新的审查目录及对应执行提交。

```sh
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B \
 migration/wc4_act/close_inplace_stage2/candidate.py \
 --candidate c0 --output <新的获批目录>

CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 MPLCONFIGDIR=/tmp/wc4-close-stage2-mpl \
 /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B \
 migration/wc4_act/close_inplace_stage2/audit.py --candidate-dir <已有候选目录>
```

实际命令、退出码、SHA、提交差异、异常和最终判定以交付包为准。没有新数据或模型，原 ACT/观测、物理/资产、R11、原专家运行源码均不修改。
