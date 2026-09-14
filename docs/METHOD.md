# 方法与保持不变的协议

目标是在能力保留时，把外部 Harness 控制计算带来的决策能力转移进学生模型，并减少部署调用成本。现在支持版本化代码演化；原 planning/review/recovery 模块模式仍保留兼容入口。

## 运行接线与计数约束（2026-09-14）

ALFWorld/HotpotQA 的版本化生产后端提供五个入口，明确 internalization/evolution_only；前者缺少入口即启动失败，后者接受改进后不训练。运行参数经严格校验，完整解析为 effective_config 并连同 hash 传给各子进程；同批 H+/H− 评分、具名目标和统计规则不变。λ 对应 advantage.module_weight，0 是有效值；all/targeted 进入实际 scorer。

公开观察区分完整 context 和增量 delta；CodeRuntime 按事件发生顺序更新模型视图，同一次环境返回不再同时原样显示为 Harness tool result。重复事件不去重；完整环境事件账本仍独立累计回报。

计划批次、尝试批次、actor 调用和实际 optimizer.step 分别计数。没有学生决策的 batch 不更新；整阶段零 actor 调用保留旧模型与已接受的 H+，不保存新 checkpoint 或执行退役 C/D。细节、配置字段与命令见 [运行接线说明](EXECUTION_WIRING.md)。

## 版本化代码演化与选择性内化

`HarnessCandidate` 是完整改进，记录 candidate_id、parent_revision、带文件基准哈希的 patch、可实际执行的 full_revision 和 rationale。`InternalizationTarget` 是可选撤除目标，记录 full_revision、reduced_revision、removed_behavior 和可执行 supervision_adapter。H_parent 是接受修改前版本；H_plus 是完整改进；H_minus 是只旁路目标控制后的版本，**不要求等于 H_parent**。例如新增日志查询工具和诊断控制后，精简版本仍保留工具。

模型只提出 path/content 修改，文件基准 hash 由宿主从已验证的 parent revision 计算；应用时仍严格校验父版本。schema 2 将控制注册为有顺序的具名入口，目标只选择 target_control_id/removed_behavior，宿主只关闭该 ID，保留其他控制和工具。旧 schema 1 单 hook 桥继续兼容。不提供可用目标则跳过目标 API；模型选择仍需通过原可执行监督检查，不能凭声明获得训练资格。

每轮从正式接受的 checkpoint 和代码 revision 出发，固定模型生成两个候选。候选在独立代码目录构造，实际加载其入口、prompt、工具注册、实现和配置；search/dev 使用既有配对收益选择规则。无收益则保留父版本，不训练。候选沿用原 search/dev 门槛：两个配对收益区间下界均严格大于 0。外层直接复用已执行的 dev 逐题结果正式接受 H_plus，记录 decision_source=dev，不再分配 acceptance_i 或重跑 acceptance_plus/acceptance_parent。bootstrap 置信度、次数、任务聚类及选择排序规则保持不变；接受后再最多提出一个精简版本。

未提供目标、精简版本不可执行或监督接口不支持时，接受 old_model + H_plus，记录 `accepted_without_internalization`。有合法目标时在原 `retirement_i` cohort 计算 A/B 并复用 contribution gate。未通过则记录 `attribution_failed`，不训练，但保留已独立接受的 H_plus。这与 search/dev 无收益、整个改进被丢弃的分支不同。

通过 contribution gate 后沿用原训练器、优势公式、optimizer 和四格审计。accept/retire 部署 new_model + H_minus；accept/retain 部署 new_model + H_plus；rollback/retain 部署 old_model + H_plus。后续周期不会重置为 H0。退役 hook 的源码仍留档，但 reduced 配置不再调用它。未消耗预算不重分配。

当前监督桥支持显式内部计算入口：`hook(api, {context, step}) -> {suffix, selected}`。schema 2 的独立控制读取同一个公开基础上下文，按固定顺序组合；H_minus 仅关闭目标 ID。rollout 保存非目标控制的实际输出和位置，评分复用它们，只运行目标并插入原位置，不能任意追加到末尾。顺序依赖组合仍可运行但不支持内化。旧 schema 1 则仅将总 hook 配置置空。其他代码、prompt、工具、配置必须共享；更广泛的代码修改不能强行转换成这种监督。检查器在 search 任务的真实学生状态上执行精简程序与目标，训练每一步继续检查。评分不能访问新环境观察、隐藏答案或学生不可见的历史；有限状态预检查不保证所有未来状态，运行失败仍保留 old_model + 已接受的 H_plus。组合协议与边界见 [具名控制说明](NAMED_CONTROLS.md)。

代码执行使用独立 Python 子进程和 Linux Landlock + seccomp；缺少必需隔离直接报错，没有宿主执行回退。候选只能通过受保护 broker 使用实验已有 model/environment 能力，不能更换端点、权重、reward、成本计数或官方评分器。正常任务运行允许环境动作；评分阶段禁止环境动作。路径白名单仅负责补丁边界，不能替代执行隔离。具体限制、配置和产物见 [VERSIONED_HARNESS.md](VERSIONED_HARNESS.md)。

当前配对通过 `AcceptedAgentState` 一起保存 checkpoint、实际 Harness revision、manifest/protocol 身份和下一周期编号。导入器按计划周期数预先分配独立 acceptance/retirement，启动前检查完整分区。最终评价从同一 state 加载代码；初始空 Harness 必须显式指定 baseline。周期恢复保留原配置与未使用 cohort，不重切任务或重分配预算。详见 [数据到最终评价接线](ACCEPTED_AGENT_PIPELINE.md)。

版本化轨迹将环境事件与学生决策分开保存。任务 return 来自全部环境返回，覆盖 prepare/control/execute 中的 reward；只有实际生成 response 的决策进入 actor batch。prepare 直接完成任务也保留终局奖励，不补造 response；整批没有学生决策则记录结果与成本并跳过 update，不追加预算。broker 的公开参数/结果提供给允许的 search 提案轨迹，不拼入教师评分输入。详见 [环境事件与回报](ENVIRONMENT_EVENTS.md)。

## 兼容模块模式与共享训练协议

每轮固定模型，通过search轨迹提出两个模块候选，经search/dev配对评价选择有用模块H+。先在本轮独立cohort上做训练前A/B归因：至少30个task，配对收益95% bootstrap区间下界严格大于固定`min_external_gain`（默认0）才训练。不通过则discard候选，保持原checkpoint与residual Harness，跳过本轮训练且不重分配预算。通过后，学生使用关闭该目标模块的H−执行on-policy轨迹；生成该batch rollout的同一个更新前策略在同一公开状态运行H+，对学生已生成的相同动作token重新评分；完成评分后才更新参数，下一batch两侧一起使用新策略。teacher指导不进入学生rollout，也不执行环境动作。

训练保留task reward，并在目标模块触发或持续窗口内加入detach(teacherLP−oldLP)模块信号，使用外部veRL PPO更新。hindsight skill、episode skill和轨迹分析器不是监督来源。模块监督使用每批behavior policy；阶段冻结模型仅用于原有KL reference。具体公式、实现与验证见 [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md)。

| 模型 | H+ | H− |
|---|---|---|
| 训练前 | A | B |
| 训练后 | C | D |

撤除要求模块先前有用、D相对B改善、D接近A且接近C，并满足真实token/调用/工具/延迟约束。不能只凭C−D缩小而忽略共同退化。模型接受与模块退役分别判定：C−A配对区间上界低于−ε（默认0.02）则回滚旧模型并保留H+；独立任务不足也拒绝新模型。否则接受新模型，D达到原能力/成本要求则retire，未达到则retain。下一轮使用正式接受的checkpoint和剩余Harness，拒绝checkpoint留档。不显著退化的接受规则不等于非劣证明。

三轮、每轮两个候选、最多退役一个模块，训练总批次平均分配；search/dev/train/每轮retirement/test严格分离。早期预注册组合为ALFWorld＋WebShop＋Search-QA，可扩展AppWorld；本轮运行接线对象为ALFWorld和已存在的HotpotQA，WebShop完整接入留待任务二。历史预注册与待运行baseline保持 [configs/experiment_protocol.json](../configs/experiment_protocol.json) 原内容不变。

仍需运行的对照：固定模型＋演化Harness、等预算RL/OPID、固定Harness蒸馏/OPHSD、SLIM生命周期、同策略文本技能、无退役、全步蒸馏、单轮蒸馏。当前CPU合成demo不能证明内化或论文创新。

AppWorld工程接入采用独立worker和官方terminal task success，复用相同P0流程；具体开发分区和上下文预算见 [AppWorld说明](benchmarks/APPWORLD.md)。本轮未运行真实benchmark，不改变历史主实验声明。
