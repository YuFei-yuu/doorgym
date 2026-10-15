# 更换技术基线前：训练产物仓库外归档

2026-10-15。用户明确选择“移出仓库归档，保留可恢复证据”。本次不删除任何模型、数据、日志、视频或失败记录，不训练、不仿真、不恢复旧监控。

## 已完成范围

服务器原仓库：`/data/zyf/video2door`。
归档位置：`/data/zyf/video2door-archives/20261015-baseline-switch-v1`。
所有原路径按原相对层级保存在归档的`tree/`下，例如：

|旧路径|新位置|
|---|---|
|`/data/zyf/video2door/experiments/`|`/data/zyf/video2door-archives/20261015-baseline-switch-v1/tree/experiments/`|
|`/data/zyf/video2door/low-level/logs/repro-smoke/`|归档`tree/low-level/logs/repro-smoke/`|
|`.repro`内实验/交付/同步历史目录及driver日志|归档`tree/.repro/`内同名位置，详见映射清单|

共归档**673014个常规文件、46个符号链接、12669479466字节**（逻辑大小约12.67GB；原磁盘占用约15GB）。同盘重命名，无删除；不会释放整个文件系统的空间。归档前后对每个文件的SHA256、大小、权限及每个符号链接文本逐项比对一致。

`before_manifest.json`与`after_manifest.json`的SHA256均为：
`ebbc9bd9660e4996656a19f191de529980210044b043fdba2fbd13a725a45f1d`。

完整文件清单、操作顺序和校验结果在归档根目录的`before_manifest.json`、`after_manifest.json`、`move_journal.json`、`summary.json`。仓库保留精简[归档摘要及全路径映射](evidence/baseline-switch-20261015/archive_summary.json)，本地管理副本在`/Users/yu/Data/Video2Door/reproduction/pro5000/maintenance/baseline-switch-20261015/`。

本地`/Users/yu/Data/Video2Door/code`没有待移动的生成模型/实验数据；已有`/Users/yu/Data/Video2Door/reproduction/pro5000/`证据包原就在仓库外，按用户选择保持原位，先前报告中的本地视频链接继续有效。

## 保留项

- `docs/`、`high-level/`、`low-level/`、`third_party/`中的当前源码、文档、配置与原始资产保持原位；没有移动任何Git跟踪文件。`high-level/experiments/isaacgym/b1z1_opendoor.yaml`是原始配置，保留。
- 专用环境`.repro/envs/`、Blackwell构建/运行支持`.repro/blackwell/`、安装包及运行缓存`.repro/conda-pkgs/`和`.repro/cache/`、GPU锁目录及`.repro`根层脚本保留，不改变环境或其他进程。
- 整个历史实验树（包括run内配置、源码快照、报告、失败日志、停止证据及所有模型/数据/视频）完整归档，不对原证据做路径替换或重新序列化。
- 额外将30份run内/历史管理目录独有辅助脚本逐字节复制到[evidence源码目录](evidence/baseline-switch-20261015/README.md)，并保留归档执行脚本，纳入Git供后续审查。它们是历史证据副本，不从新位置直接执行。
- 先前未提交的18个实验源码/文档已独立提交为`a999426`，保留观测分支、单回合batch8及C/L学习率配对实现与结果，未改变旧checkpoint含义。

## 旧路径与恢复规则

历史文档、checkpoint配置、manifest及脚本保留原绝对路径，防止修改既有证据。归档后旧服务器路径不再存在，历史脚本不能直接续跑；旧绝对符号链接也可能暂不可解析。读取时按`archive_summary.json`中的`moves`映射查找归档路径；需要复现实验时先明确恢复范围。

恢复使用`move_journal.json`中的每个`new → old`映射。在没有旧训练进程、且目标位置不存在时逐项目移回；如新基线已占用同名位置，先单独安排目标位置，不覆盖或合并。不要建立指回旧实验的仓库内快捷链接，也不要自动启动旧训练/监控。恢复后原绝对引用与符号链接重新对应原布局，可重新对照完整清单核验。

## 校验与同步

- 归档前本地和服务器18个待提交文件逐文件哈希一致，远程main与起始HEAD`5157bd4`一致。
- 执行前检查归档根不含Git跟踪文件，也没有Python进程引用选中输出目录；仅对本项目执行同盘移动，使用nice10及idle I/O优先级。
- 归档前后673014文件/46链接完整一致；当前代码和文档同步后检查Git差异、源码语法及辅助源码副本哈希。
- 本轮没有改变网络、数据契约、监督、执行器、任务阈值或训练预算。下一技术基线尚未实施，旧实验保持结束状态。
