# 方法与保持不变的协议

目标是在能力保留时，把可执行规划/审核/恢复带来的决策能力转移进学生模型，并减少部署调用成本。

每轮固定模型，通过search轨迹提出两个模块候选，经search/dev配对评价选择有用模块H+。先在本轮独立cohort上做训练前A/B归因：至少30个task，配对收益95% bootstrap区间下界严格大于固定`min_external_gain`（默认0）才训练。不通过则discard候选，保持原checkpoint与residual Harness，跳过本轮训练且不重分配预算。通过后，学生使用关闭该目标模块的H−执行on-policy轨迹；生成该batch rollout的同一个更新前策略在同一公开状态运行H+，对学生已生成的相同动作token重新评分；完成评分后才更新参数，下一batch两侧一起使用新策略。teacher指导不进入学生rollout，也不执行环境动作。

训练保留task reward，并在目标模块触发或持续窗口内加入detach(teacherLP−oldLP)模块信号，使用外部veRL PPO更新。hindsight skill、episode skill和轨迹分析器不是监督来源。模块监督使用每批behavior policy；阶段冻结模型仅用于原有KL reference。具体公式、实现与验证见 [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md)。

| 模型 | H+ | H− |
|---|---|---|
| 训练前 | A | B |
| 训练后 | C | D |

撤除要求模块先前有用、D相对B改善、D接近A且接近C，并满足真实token/调用/工具/延迟约束。不能只凭C−D缩小而忽略共同退化。模型接受与模块退役分别判定：C−A配对区间上界低于−ε（默认0.02）则回滚旧模型并保留H+；独立任务不足也拒绝新模型。否则接受新模型，D达到原能力/成本要求则retire，未达到则retain。下一轮使用正式接受的checkpoint和剩余Harness，拒绝checkpoint留档。不显著退化的接受规则不等于非劣证明。

三轮、每轮两个候选、最多退役一个模块，训练总批次平均分配；search/dev/train/每轮retirement/test严格分离。主实验固定ALFWorld＋WebShop＋Search-QA，可扩展AppWorld。正式任务依赖与待运行baseline保持 [configs/experiment_protocol.json](../configs/experiment_protocol.json) 原内容不变。

仍需运行的对照：固定模型＋演化Harness、等预算RL/OPID、固定Harness蒸馏/OPHSD、SLIM生命周期、同策略文本技能、无退役、全步蒸馏、单轮蒸馏。当前CPU合成demo不能证明内化或论文创新。

AppWorld工程接入采用独立worker和官方terminal task success，复用相同P0流程；具体开发分区和上下文预算见 [AppWorld说明](benchmarks/APPWORLD.md)。本轮未运行真实benchmark，不改变历史主实验声明。
