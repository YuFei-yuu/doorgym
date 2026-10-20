# wc4 Joint9 ACT baseline

已完成并验收：50条成功专家轨迹、27500帧双深度数据、ACT训练至50k；最终64回合开门成功48次（75%）。当前没有待执行的续训或补采任务。

- [完整复现手册](migration/wc4_act/REPRODUCTION_GUIDE.md)：从固定作者源、环境构建到采集、转换、训练、导出和闭环评测。
- [成功基线与产物索引](migration/wc4_act/README.md)；[机器可读记录](migration/wc4_act/BASELINE.json)。
- [环境构建附件](migration/wc4_act/reproduction_support/README.md)。

主线仅保留本次wc4/A2W浮动底座、直接Joint9、ACT baseline及其源依赖。训练量30k以上，可按任务扩大到40k～50k；无需重走历史分段训练。开门成功判据为最大绝对门角≥80°，不代表穿越、碰撞合理性或停稳已经验收。

作者代码来自 [Sivan666666/doorgym](https://github.com/Sivan666666/doorgym) 的固定提交 `096938c44ecc1c7602304d42502f96658e46eb25`。保留模块未修改作者算法；无关实验、其他门型资产和历史交接文件已从主线移除。`high-level/lerobot/src`及`third_party`作为完整依赖库保留，不能把其中可选算法或SDK例程当成本项目实验入口。许可证见 [LICENCE](LICENCE)及各依赖自身许可证。
