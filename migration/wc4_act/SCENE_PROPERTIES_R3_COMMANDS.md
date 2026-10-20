# 第三轮命令、配置与预算记录

服务器工作目录 `/data/zyf/video2door-wc4-act`。本轮仅一次实际场景构造，以下不是继续创建或运行轨迹的授权。

## 输入与无场景预检

起点7be9882，分支 `expert/wc4-scene-properties-r3-20261020`。新路线：

```text
/data/zyf/video2door-wc4-act/.repro/reviews/wc4-fullcycle-r2-independent-20261020-03/geometry-compact.json
SHA256 872dd74761d51bd241caa89682ccff285321a70ded53d3bc0988cd80fd16be47
```

```bash
CUDA_VISIBLE_DEVICES= /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B \
  migration/wc4_act/inspect_scene_properties.py \
  --output .repro/runs/wc4-fullcycle-properties-r3-20261020 \
  --geometry .repro/reviews/wc4-fullcycle-r2-independent-20261020-03/geometry-compact.json \
  --preflight
```

预检只测试代理拒绝规则与配置hash；另以AST确认检查入口没有simulate/fetch_results/轨迹入口调用。没有创建场景。

## 唯一场景调用

执行提交 `b3951e5d12143885c2ced716b001f5dea8dae50f`，工作区干净：

```bash
bash migration/wc4_act/inspect_scene_properties_launch.sh
```

脚本选择物理GPU2、获取原共享UUID flock，连续三次检查空闲后激活原Gym环境。完整argv及绑定在 `command.json`，实际参数与默认/asset覆盖后args分别保存。关键参数为：

```text
--rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0
--num_envs 1 --steps 0 --seed 62000 --headless
--door_name wc4 --asset_file urdf/a2wz1.urdf
--no_ikpush_env_randomization --no_enable_depth_camera_randomization
--no_enable_depth_noise --no_enable_depth_gaussian_blur
--skill_program_json <原wc4_smoke_skill.json>
--arm_ik_solver gym_jacobian --ee_pose_frame robot_base_full
--camera_intrinsics_mode legacy --no_collision_geom_check
--no_show_camera_images --no_preview_trajectory_at_spawn
```

`steps=0`只用于说明此检查没有执行循环，不是另一个专家时间预算。没有调用作者main、IK setup或run_demo。构造完成后封闭所有setter，再做属性get读取，最后destroy。退出码0；1次create_sim、1次create_env、1次prepare_sim、0次simulate/fetch_results、1次destroy_sim。

唯一创建凭据为根目录 `scene-creation-receipt.json`。之后的拒绝测试仅在凭据副本fixture上运行，在SDK导入前退出，且验证命令哨兵文件不被覆盖；没有重建真实场景。后续代码修正也没有重跑本脚本。

## CPU分析

成功版本 `a672ecf4cdc245465121889d93bdaf01a5c3c88d`：

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 \
  /data/zyf/video2door/.repro/envs/b1z1-blackwell/bin/python -B \
  migration/wc4_act/analyze_scene_properties.py \
  --scene .repro/runs/wc4-fullcycle-properties-r3-20261020 \
  --r1 .repro/runs/wc4-fullcycle-nominal-r1-20261020 \
  --expert .repro/runs/wc4-joint9-skill50-s62000-20261015/raw/episode_000000.npz \
  > .repro/runs/wc4-fullcycle-properties-r3-20261020/analysis-v2.log 2>&1
```

只读已保存JSON和原NPZ，使用源FK及圆柱SAT。不会导入Gym、创建模拟器、应用旧状态或执行轨迹。初版括号错误日志为 `analysis.log`，修复后另存新日志；原始属性快照保持不变。

另外在隐藏GPU的Gym CPU Python中只构造RigidShapeProperties/RigidBodyProperties数据结构并探测类字段，没有acquire_gym/create_sim。结果 `binding-field-probe.json` 明确区分“构造器字段可用性”和“实际actor读回值”，不能用构造器默认值填补actor缺失属性。

## 关键输出

- `execution-head.txt`、`execution-worktree.patch`、`source/`：唯一场景的真实代码版本。
- `scene.log`、`scene.exit`、`scene-creation-receipt.json`、`inspection-result.json`、`api-call-audit.json`：调用和预算。
- `effective-args-*.json`、`sim-parameters.json`、`actor-creation-arguments.json`：传入设置和原覆盖结果。
- `asset-imports.json`、`asset-properties.json`、`actor-properties.json`：原始SDK元数据。
- `generated-assets/`、`imported-assets/`、`cooked-cache/`及provenance：生成资产及已有cache；缺失的原MTL照实列出，未补造。
- `analysis/`：规范化视图、索引/过滤/坐标/home/历史接触分析；原始JSON未覆盖。
- `snapshot-validation.json`：数据完整性及已知材料缺失，不是几何或任务认证。
- `repeat-refusal-check.json`：凭据副本上的无SDK拒绝测试。

SDK未导出的几何/偏移/时间字段没有被补造；本轮没有实际专家argv、轨迹状态/接触新日志、执行录像、数据采集或训练命令。新路线和原任务阈值没有改变，下一轮是否实际执行由独立审查决定。
