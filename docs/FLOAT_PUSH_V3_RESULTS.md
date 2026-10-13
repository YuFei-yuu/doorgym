# 旧5000更新模型参考

这是直接通过修复前的source模式实验，保留作数据格式、训练链路和模型对照。当前forward方案另需建立训练集合。

- 源码实现：7f8d52c；固定源基线0acc2f050479fb54a0670ccb90bc72b98b52a54a。
- 20个专家回合；训练30330帧、离线验证8065帧。
- 随机初始化seed42、原LeRobot逐帧读取、5000更新。
- 完整评估：验证开门1/3、通过0/3；测试开门0/3、通过0/3。
- 最终模型：training/frozen_step_005000.pt。
- SHA256：fc4cb9c5eec379d0d8f2b7266c332cda241a4f9d90787fbc110ee0744984b0c6。

服务器目录：experiments/float-push/20261013-upstream-float-v3/。
本地目录：reproduction/pro5000/float-push-v3/。

保留data/raw_train、raw_val、完整dataset记录、最终模型、训练日志、完整闭环轨迹/视频、分析图和环境导出。data/lerobot已作为可重建的转换产物清理；原转换入口可从raw生成。保留文件清单见run内retained_sha256.json，原始交付清单记录的是清理前版本。
