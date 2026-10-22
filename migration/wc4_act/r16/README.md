# R16 离线诊断执行合同

用户授权：30k/40k/50k同组最多512个已有观测，所有一致性/重复前向均计入1536上限；CPU，零训练/采集/仿真/闭环。无自动重试或恢复。使用现有DoorPolicyController、原预处理和eval零VAE latent推理，单样本chunk100，分析前25和100；绝不传入未来动作、phase、门角或绝对朝向。

prepare.py只用原始专家阶段/真值和失败轨迹实测状态选点，72专家各4点、4失败各56点，实际LeRobot factory读专家输入及动作/padding；失败点全选真实重推边界，阶段仅作分析标签。观测和每点输入先冻结，之后predict.py的真实model.forward pre-hook在每次模型调用前持久化预算claim，完成后保存原始normalized/physical chunk与哈希。任何中断均保留现场，不重新运行predict.py。

专家GT严格同episode、padding屏蔽；失败轨迹没有专家动作真值，不套邻近专家标签。近邻相似度只是描述性分析，离线误差不等价闭环成功率，近似歧义不证明不可学习。CPU/GPU数值对照使用已有真实worker输出，不新增重复预测。物理期望仅作指令时序积分分析，不启动仿真。工程故障先停止回报，禁止自行修模型或规则停车。
