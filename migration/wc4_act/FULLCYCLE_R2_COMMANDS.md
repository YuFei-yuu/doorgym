# 第二轮实际命令与输出

服务器工作目录均为 `/data/zyf/video2door-wc4-act`。以下是已执行的CPU命令记录，不是继续运行授权。输出目录都要求不存在；不要覆盖封存结果。源码版本以各目录 `source/` 快照为准。

## 无仿真检查

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 \
  /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B \
  migration/wc4_act/fullcycle_r2_checks.py \
  --source .repro/runs/wc4-fullcycle-nominal-r1-20261020 \
  --output .repro/runs/wc4-fullcycle-nominal-r2-20261020/offline \
  > .repro/runs/wc4-fullcycle-nominal-r2-20261020/offline.log 2>&1
```

首次因NumPy bool序列化退出；修复后以上同样参数使用新输出目录 `offline-v2`、日志 `offline-v2.log`，完成接触分析与当时版本的纯函数测试。

完整路径扫描：

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 \
  LD_LIBRARY_PATH=/data/zyf/video2door/.repro/envs/b1z1-blackwell/lib \
  /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B \
  migration/wc4_act/fullcycle_r2_checks.py \
  --source .repro/runs/wc4-fullcycle-nominal-r1-20261020 \
  --output .repro/runs/wc4-fullcycle-nominal-r2-20261020/offline-v3 \
  > .repro/runs/wc4-fullcycle-nominal-r2-20261020/offline-v3.log 2>&1
```

最终纯函数、控制函数及home内包络复核：

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 \
  LD_LIBRARY_PATH=/data/zyf/video2door/.repro/envs/b1z1-blackwell/lib \
  /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B \
  migration/wc4_act/fullcycle_r2_checks.py \
  --source .repro/runs/wc4-fullcycle-nominal-r1-20261020 \
  --output .repro/runs/wc4-fullcycle-nominal-r2-20261020/final-targeted-tests-v2 \
  --tests-only \
  > .repro/runs/wc4-fullcycle-nominal-r2-20261020/final-targeted-tests-v2.log 2>&1
```

较早一次 `--tests-only` 使用 `final-targeted-tests`、`final-targeted-tests.log`，同样保留。两者有各自执行源码快照。

## 独立时长提取

在同一CPU环境及相同 `CUDA_VISIBLE_DEVICES`、`OPENBLAS_NUM_THREADS`、`LD_LIBRARY_PATH` 下执行以下Python，未进行几何扫描或动力学：

```python
import sys, json
from pathlib import Path
from collections import Counter
sys.path.insert(0, "migration/wc4_act")
from fullcycle_r2_path import stress_trace
from fullcycle_geometry import URDF
out = Path(".repro/runs/wc4-fullcycle-nominal-r2-20261020/timing-check")
out.mkdir(exist_ok=False)
robot = URDF("high-level/data/asset/a2wz1/urdf/a2wz1.urdf")
door = URDF("high-level/data/asset/door_set/wc4/model.urdf")
samples, result = stress_trace(
    Path(".repro/runs/wc4-fullcycle-nominal-r1-20261020"), robot, door)
result["phase_durations_s"] = {
    k: v * .02 for k, v in Counter(x["phase"] for x in samples).items()}
(out / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
```

## 启动拒绝测试

实际由ACT CPU Python子进程调用以下argv，完整argv、返回码、stdout、stderr已保存在 `launch-refusal-check.json` 等文件中。

```bash
CUDA_VISIBLE_DEVICES= .repro/envs/door-act/bin/python -B \
  migration/wc4_act/fullcycle_expert.py \
  --validation .repro/runs/wc4-fullcycle-nominal-r2-20261020/offline-v3/gate.json \
  --output .repro/runs/wc4-fullcycle-nominal-r2-20261020/refused-run-must-not-exist \
  --geometry .repro/runs/wc4-fullcycle-nominal-r1-20261020/offline-v2/geometry.json
```

预期结果：返回码1、错误包含 `no simulator may be created`、输出回合目录不存在。它在Gym导入/创建模拟器之前停止，不是专家尝试。

## 绑定默认值与证据完整性

Gym CPU环境、隐藏GPU下，仅构造 `gymapi.SimParams()` 与 `gymapi.AssetOptions()`，读取 `contact_offset`、`rest_offset`、`thickness`、`vhacd_enabled` 到 `binding-defaults.json`；没有调用acquire_gym、create_sim或simulate。

```bash
cd /data/zyf/video2door-wc4-act/.repro/runs/wc4-fullcycle-nominal-r1-20261020
sha256sum -c SHA256SUMS > ../wc4-fullcycle-nominal-r2-20261020/r1-original-hashcheck.txt
```

17项测试、源码编译检查、入口拒绝、字段文档哈希、首轮原始证据哈希均完成。没有实际专家命令、GPU租约、回合seed替换、模型加载、录像执行或补跑命令。
