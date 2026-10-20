# 已验收：40k→50k基线终点（实验2026-10-15，验收2026-10-20）

本目录归档本轮**实际执行脚本的原样副本**及精简证据，不是当前可直接重跑的默认入口。`tools/common.py`保留执行时固定源码HEAD `5ca33bf`、服务器绝对路径和历史run ID；预算/输出目录保护也原样保留。不要修改历史归档以适配新提交，不要重跑已完成训练或仿真。后续执行须依据新任务书建立新run及显式版本、预算和路径。

## 结果

从已验收40k恢复模型、优化器、RNG和统计，新增恰好10,000次更新至50k。保存42/44/46/48/50k恢复节点，**仅50k做一次64回合**；未补评或重跑旧模型。source/skill/物理/数据/预处理及频率未改变。

| 对比 | 共同成功 | 共同失败 | 改善 | 退步 | 净变化 |
|---|---:|---:|---:|---:|---:|
| 36k→50k | 45 | 9 | 3 | 7 | −4 |
| 40k→50k | 43 | 13 | 5 | 3 | +2 |

50k为48/64（75%），按最大绝对门角≥80°。相对40k在固定集上小幅净增2次，未超36k的52/64；不是独立泛化结论。7个持续失败仍全部失败；原8个退步恢复3个，原2个改善保持1个。50k全部16个失败均未记录到解锁标志，根因未确诊。没有独立未使用初态确认、部署或延长训练。

成功/失败原始整体视频各1段，每段550帧、25fps、22秒，从这64次内同步选取；未补跑或状态回放凑片。前置/腕部depth不能替代整体RGB。新run按用户减量授权只保留必要选片；其余新生成临时候选clip的SHA/receipt及全部原始日志保留。

## 代码与证据

- [精简证据完整报告](evidence/REVIEW.md)、[验收JSON](evidence/acceptance.json)、[配对逐回合结果](evidence/paired-comparison.csv)、[17个重点回合](evidence/FOCUS_CASES.csv)。
- [有效配置差异](evidence/training-config-diff.json)、[实际训练命令](evidence/train-command.json)、[运行时差异](evidence/runtime-adaptation.diff)。作者核心文件未改；新增内容为执行守卫、预算/GPU锁、在线监控、选片和只读日志分析。
- [脚本及证据原样复制SHA](archive-manifest.json)；[完整运行证据清单](evidence/evidence-manifest.json)；[模型哈希](evidence/model-hashes.csv)、[原始日志索引](evidence/RAW_LOG_INDEX.csv)、[视频索引](evidence/VIDEO_INDEX.md)。
- `evidence/analysis-existing`在50k评测前固定7/8/2清单；只读旧日志，不产生新推理或仿真。解锁标志只定位观察阶段，接触记录不等于有效抓握，目标/实测差不是因果控制故障证据。

## 完整产物位置

```text
本地：/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act-resume40k-50k-s1000-20261015/
服务器：pro5000:/data/zyf/video2door-wc4-act/.repro/runs/wc4-act-resume40k-50k-s1000-20261015/
50k官方checkpoint：服务器目录/train/checkpoints/050000/
Door manifest：服务器目录/checkpoints/050000/door-checkpoint/model_latest.pt
```

`evidence/REVIEW.md`及CSV中的绝对链接是原交付工作站/服务器的证据位置；历史字节未改，不表示媒体或大模型已放进Git。本目录不提交模型、数据集、视频、原始日志或环境缓存。完整本地镜像保留选片和gzip原始日志；大模型/optimizer保留服务器。

50k模型SHA256：`3f1f32f96eb3a78abd215f6affa6786d7b2cf76b94def6b8268bdeefe3bfbdb9`。

W&B：[训练xobd7ef0](https://wandb.ai/fi-yu-individual/door-dp-stop-tail/runs/xobd7ef0) · [评测lj85duyv](https://wandb.ai/fi-yu-individual/door-dp-stop-tail/runs/lj85duyv)。截至验收，训练、仿真及在线监控均已结束；后续候选与独立确认等待新任务书。
