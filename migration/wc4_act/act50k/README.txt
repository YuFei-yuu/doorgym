本轮：wc4-act50k-20261029-r2
实际训练/评测执行提交：949c98fcd8836df885d74a5cc4e67003097ad637。
输入冻结197a169：50回合、77,356帧；固定评测seed62300—62315。
新ACT仅使用ResNet18 ImageNet V1骨干初始化；seed1000、batch16、lr1e-5、chunk100、训练n_action_steps50；全量均匀采样。
同一训练进程累计50,000次更新，10k/20k/30k/40k/50k保存完整训练状态并各完成16回合闭环评测。执行源码与物理资产在运行期间保持冻结。
训练入口start.py调用train.py包装原作者lerobot_train.py；evaluate.py/export.py/rollout.py/worker.py分别管理节点评测、原作者导出及实际Joint9闭环控制。
本轮最终只读验收与冻结入口：freeze.py。audit_stage.py生成逐节点完整验收与文件SHA；pack_stage.py和receive_parts.py用于完整字节传输校验。后续新增文件只用于验收和冻结，不修改执行源码。
逐节点只读验收在pro5000既有ACT环境、隐藏CUDA的独立CPU进程完成；未新增依赖。
最终交付：/data/zyf/video2door-wc4-act/.repro/reviews/wc4-act50k-20261029-r2-FINAL
本地完整镜像：/Users/yu/Data/Video2Door/reproduction/wc4-act/wc4-act50k-20261029-r2/FINAL
当前输出和来源已冻结；未来任务按新版本及新目录适配，不直接复用本轮固定路径启动。
