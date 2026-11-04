# WC4流程文档集中入口

更新日期：2026-11-04。目录名保留以维持既有链接；新专家和两轮数据／训练尝试已整理归档，无下一轮执行授权。历史报告保持原文和当时的状态描述。

## 新对话阅读顺序

1. [新专家采集与ACT训练流程报告（最新）](WC4_新专家采集与ACT训练_20261104.md)：验收专家绑定、简化成功判据、50／100条数据链、50条ACT至300k与100条ACT从头至200k、评测结果、工程修复、源码和冻结交付定位。
2. [新专家路线：阶段1至8历史报告](WC4_新专家路线_阶段1至8_流程报告.md)：2026-10-25旧路线及50条／121,305帧数据，不能混同后来新专家的50条／77,356帧数据。
3. [旧完整任务：专家生成至200k历史报告](WC4_专家生成至200k_流程报告.md)：更早紧凑专家、72条数据与0/16结果。
4. [旧仅开门复现手册](REPRODUCTION_GUIDE.md)：环境、作者入口、Joint9、转换与ACT基础配方；旧数量、预算和开门单项判据不作为当前授权。

[2026-11-04文档及来源清单](DOCUMENT_MANIFEST_20261104.json)记录本次文档、来源文件及SHA；[2026-10-25清单](DOCUMENT_MANIFEST.json)作为历史快照保留，其中index_sha256对应旧版README，不应用于当前入口。原三份报告未改写。仓库提交和远程分支用于源码定位，实际执行提交、checkpoint及数据仍按各冻结交付绑定。

## 仓库文档导航

| 用途 | 入口 |
|---|---|
| 当前项目约定 | [AGENTS.md](../../../AGENTS.md) |
| WC4总入口及旧开门基线 | [README](../README.md)、[BASELINE.json](../BASELINE.json) |
| 新专家阶段1—3：设计与连续部署修正 | [阶段1](../newexpert_stage1/RESULT.md)、[阶段2](../newexpert_stage2/RESULT.md)、[阶段3](../newexpert_stage3/RESULT.md) |
| 新专家阶段4—6：名义试运行及扰动验证 | [阶段4](../newexpert_stage4/RESULT.md)、[阶段5](../newexpert_stage5/RESULT.md)、[阶段6](../newexpert_stage6/RESULT.md) |
| 新专家阶段7—8：50来源与转换 | [阶段7](../newexpert_stage7/RESULT.md)、[阶段8](../newexpert_stage8/RESULT.md) |
| 当前成功判据 | [开门80°＋合法重锁源码](../success_random5/criterion.py)、[评测事件](../act100_200k/events.py) |
| 历史完整任务判据 | [R11判据](../r11/CRITERIA.md)、[历史源码](../r11/criteria.py) |
| 历史紧凑专家与数据 | [R7](../FULLCYCLE_R7_RESULT.md)、[R11定版](../FULL_TASK_REAUDIT_R11_RESULT.md) |
| 历史ACT至200k | [200k最终报告](../FULL_TASK_ACT_R21_200K_RESULT.md)、[续接工程入口](../r21_continue/README.md) |
| 环境与已保留修复 | [环境附件](../reproduction_support/README.md)、[长视频读取修复](../LONG_VIDEO_READER_R10_RESULT.md) |

这些报告按阶段保留，未把历史结果合并改写成新的执行证据。`FULLCYCLE_*`、`FULL_TASK_*`属于历史链；`newexpert_stage1`至`newexpert_stage8`属于2026-10-25历史新专家链，编号不可混用。

## 原始交付在哪里

最新2026-10-28至11-04交付、完整服务器位置和本地覆盖范围见[最新报告的交付表](WC4_新专家采集与ACT训练_20261104.md#交付与恢复定位)。下列阶段1—8位置属于历史链。

- 本地代码：`/Users/yu/Data/Video2Door/code-wc4-act`
- 本地交付根：`/Users/yu/Data/Video2Door/reproduction/wc4-act`
- 各阶段本地交付：`wc4-newexpert-stageN-20261025/FINAL`，其中N为1—8。
- 服务器：`ssh pro5000`；代码：`/data/zyf/video2door-wc4-act`。
- 阶段1—3证据主要在服务器`.repro/reviews/`；阶段4—8在`.repro/runs/`，完整目录见新报告。

本地交付可能是完整镜像、无损JSONL归档或仅小型审查包，必须读各阶段INDEX，不能把“本地有索引”当作“本地有全部原始字节”。

## 2026-10-25存储维护补充

用户授权保留三份报告的核心成果并清理存储后，旧`wc4-fulltask-*`回合目录中、FINAL之外的大JSONL已无损压缩为原路径加`.gz`。只有解压回读SHA与原文件一致后才移除展开副本；旧报告及冻结SHA清单不改写，其原始路径可按清理回执定位归档或恢复。**原始路径暂不存在不等于证据内容已删除。**

全部checkpoint路径和字节保留，包括旧开门50k、完整任务200k及其恢复／主线节点；部分完全相同且允许写入的模型／bundle副本共享硬链接。只读冻结目录未解除保护，冻结文件不得原地编辑。NPZ、Parquet、视频、归一化统计、环境以及新专家阶段1—8原始日志没有删除或改写。另清理了两个可由既有Parquet重建的Arrow缓存。

- [清理结果与恢复说明](/Users/yu/Data/Video2Door/reproduction/wc4-act/storage-cleanup-20261025-01/README.txt)
- [三份报告保留清单](/Users/yu/Data/Video2Door/reproduction/wc4-act/storage-cleanup-20261025-01/retained-artifacts.json)
- [本地逐文件回执](/Users/yu/Data/Video2Door/reproduction/wc4-act/storage-cleanup-20261025-01/local/receipts)
- [服务器逐文件回执镜像](/Users/yu/Data/Video2Door/reproduction/wc4-act/storage-cleanup-20261025-01/server/receipts)
- [单文件恢复工具](/Users/yu/Data/Video2Door/reproduction/wc4-act/storage-cleanup-20261025-01/restore_log.py)

服务器维护目录为`/data/zyf/video2door-wc4-act/.repro/cleanup/storage-cleanup-20261025-01`。工具只按一个精确原始路径恢复并核对SHA，不覆盖已存在的不同内容；不要为读取一条日志而批量展开全部归档。
