# 实现状态

## 2026-09-14：dev 直接正式接受 Harness

已取消独立 acceptance_i 数据集要求和 acceptance_plus/acceptance_parent 额外评价。原 search/dev 配对区间下界均>0的规则与排序保持；RevisionSelection 返回父/候选 dev 逐题结果、配对收益和接受判定，外层复用并记录 decision_source=dev。无目标、不兼容、A/B未通过仍保留已接受H+；训练、C/D及模型rollback未改。

新生成manifest不再含acceptance；旧分区可闲置保留，不挪用、不重划既有retirement/test。旧协议续跑的接受来源变化只记录于新输出，保留原协议hash及next_cycle。

全套 **202项测试通过**（新增7项，225.071秒，无skip），旧CPU demo **90文件逐字节一致**；本轮开始时的17个training/内化目标/统计文件哈希保持。两个候选的合成证明共6次实际评价调用，接受本身0次额外调用，成本证据复用不重复计费。真实HF/veRL/GPU/官方数据未运行。

说明见 [DEV_ACCEPTANCE.md](DEV_ACCEPTANCE.md)，验收见 [验证报告](validation/dev-acceptance-report.json)。以下为历史记录；其中关于独立 acceptance 的描述已经被本节替代。

## 2026-09-14 任务一：运行接线与记账修复

ALFWorld/HotpotQA 的版本化后端已配齐五入口，internalization 缺入口启动即失败，evolution_only 明确不训练。严格 execution 配置经主 CLI 和实际 JSON subprocess 进入 runner/ModuleTrainer/scorer/HF/VerlPolicy；保存 effective_config + SHA256，续跑绑定配置身份。λ=0、all/targeted、任务/rollout 数、学习率和 max_steps 均有回归验证。

完整环境上下文采用替换语义，增量观察按事件追加；同次环境返回不再被重复呈现为工具结果，真实重复事件及原始奖励/成本账本保留。planned/attempted/actor/optimizer 独立计数，实际 optimizer.step 由 post-step hook 观测；未知时为 null。整阶段无 actor 更新不保存新模型，外层保留旧模型和已接受的 H+。

最终全套 **195 项测试通过**（新增13项，257.889秒，无 skip）；旧CPU三周期demo **90文件逐字节一致**，8个算法/协议/隔离关键文件未改。模型/官方数据为测试替身，部分实际执行代码沙箱、HotpotQA worker、CPU SGD/AdamW和子进程；真实API proposer、HF、外部veRL执行和GPU/官方任务均**未运行**。证明产物：`runs/task1-config-proof`、`runs/task1-zero-update-proof`。

配置、修改清单及准确命令见 [EXECUTION_WIRING.md](EXECUTION_WIRING.md)，完整验收见 [验证报告](validation/execution-wiring-report.json)。本轮不新增benchmark，WebShop完整适配留待任务二。以下条目是历史记录，以本节为最新运行状态。

## 2026-09-14：环境事件回报与学生 loss 分离

版本化 rollout 改用 EventTrajectory：环境事件是 total_reward 的唯一来源，覆盖 prepare/control/execute；旧 Transition 保留但不会把奖励重复加到事件回报上。无学生决策时不补造 response；全空 batch 跳过教师评分/actor update，保存 return、成功、成本和跳过次数，checkpoint step 为实际 update 次数。混合 batch 只保留真实决策行。

broker 立即保存公开调用参数、结果、奖励及 phase/step；初始公开观察、本地工具 dispatch 和完整事件进入 search proposer 可见轨迹，不序列化隐藏 evaluator/环境对象，也不进入教师 prompt。prepare 完成任务后不继续控制或学生生成；prepare 环境调用后 execute 本地工具的成本漏计也已修复。

全套 **182 项测试通过**（254.208秒，无skip），新增11项；具名控制17项针对性回归通过。旧三周期90文件逐字节一致、原20项测试对应源文件哈希不变。7个优势/策略同步/veRL/评分/统计/循环关键文件本轮哈希不变；trainer 只增加零决策跳过及实际更新次数记录。示例位于 runs/environment-events-proof：prepare 终止 return=3、零决策；下一 prepare 终止 return=3.5、仅原两个 response token。真实隔离与 CPU mock/SGD 验证通过，真实模型/API/GPU/官方任务未运行。详见 [说明](ENVIRONMENT_EVENTS.md) 与 [验收记录](validation/environment-events-report.json)。

## 2026-09-14：具名控制与单 ID 撤除

新增 schema 2 ordered controls registry 与 target_control_id；宿主只关闭所选 enabled ID，保留其他控制、工具、源码与顺序。独立控制共享公开基础输入；rollout 记录实际 ControlContext，评分复用非目标输出并在原位置插入目标。组合记录缺失或不一致明确拒绝。sequential_suffix 可以运行但不能内化，有效候选保留 H+。旧 schema 1、旧目标和旧 Transition JSON 保持兼容。

全套 **171 项测试通过**（244.863秒，无skip），新增17项。验证首/中/末位置组合、随机非目标输出缓存、两周期 retain recovery→retire review、rollback 保留两个控制、不兼容保留、不触发/no-op/同批策略/mask/梯度隔离。具名运行真实使用 Landlock/seccomp；配置开关读权限进一步收紧，没有放宽原隔离。

旧 CPU demo **90文件逐字节一致**，原20项测试对应源文件哈希不变。7个优化器编排/优势/策略同步/统计/外层循环关键文件本轮哈希不变。示例 `scripts/demo_named_controls.py` 已运行至 `runs/named-controls-proof/`，保留 recovery 与日志工具，只增强 review；模型脚本化，实际 optimizer updates=0。另有 CPU 小模型真实 SGD 测试；真实 API/模型/HF/veRL/GPU/官方任务未运行。详见 [协议与命令](NAMED_CONTROLS.md) 和 [验收记录](validation/named-controls-report.json)。

## 2026-09-14：宿主绑定补丁与确定性精简版本

CodeProposer 的编辑响应只接受 path/content。宿主从锁定父版本填入 before_hash，再经原 apply 和候选谱系校验；归档格式仍含完整 hash。目标 API 只返回待撤除行为与已注册 hook，宿主确定性构造 reduced revision，保留其他代码与工具。无 hook 时跳过目标 API 并记零调用成本；有 hook 时仍有一次行为选择调用，之后仍必须通过可执行兼容性检查。

全套 **154 项测试通过**（243.064 秒，无 skip），新增10项覆盖 hash 绑定、父版本篡改拒绝、减法确定性、保留工具实际隔离运行、不兼容监督拒绝和独立 target 子进程。原接口 mock 只调整为新的 path/content 响应，原断言保留。训练、优势、策略同步、统计规则与外层循环9个关键文件本轮哈希不变；前次接线修改保留。真实 API/模型/GPU/官方任务未运行。详见 [接口说明](VERSIONED_HARNESS.md) 与 [验证记录](validation/proposer-binding-report.json)。

## 2026-09-14：数据、已接受 Agent 与最终评价接线

统一 `AcceptedAgentState` / `load_accepted_state` 已接通新版循环、CLI 周期恢复和最终评价，绑定 checkpoint、可执行 Harness、manifest/protocol 身份。通用、ALFWorld、AppWorld 导入器按周期数生成 acceptance/retirement；启动前集中检查，数据不足报错。最终评价要求 --state 或显式 --baseline，未知/损坏状态不回退为空 Harness。

全套 **144 项测试通过**，旧三周期 demo **90 文件逐字节一致**，原20测试文件未改。合成接线证明保存于 `runs/accepted-agent-pipeline-proof`：mock 搜索分数，最终 runner 实际在隔离进程调用已接受的新工具。旧代码 deployment + 原 protocol 加载已实际复核。

边界：LawBench 专用整类原生评价器暂不支持代码 revision，现明确报 unsupported；部分后端未配置可选 target/check_internalization 时仍只演化、不训练；未执行真实 HF/veRL/GPU/API proposer 或官方任务。详见 [运行说明](ACCEPTED_AGENT_PIPELINE.md) 与 [验收记录](validation/agent-pipeline-report.json)。

## 2026-09-13：版本化代码演化＋选择性内化

完整接口、准确命令、产物位置和边界见 [VERSIONED_HARNESS.md](VERSIONED_HARNESS.md)。新模式显式传入 Harness revision，旧模块模式保留。

| 项目 | 当前状态 |
| --- | --- |
| 完整 HarnessCandidate 与可选 InternalizationTarget | 已实现；完整树、patch、parent、hash、rationale 和 full/reduced 产物可读取和执行 |
| 实际代码/工具执行与失败隔离 | Linux Landlock ABI4 + seccomp 真实执行通过；多文件工具注册实际调用，禁止直接读取宿主私有文件、写文件、联网、fork，超时和失败不污染父版本 |
| 独立 Harness 接受与可选内化 | acceptance_i 独立门槛；无目标/不兼容保留 H_plus 不训练；contribution gate 失败也保留已接受 H_plus |
| P0-1 / P0-2 / P0-3 | 原训练和统计实现未改；新增桥覆盖 same-batch、no-op、mask、共享上下文；rollback 保留已接受 H_plus 和新增工具 |
| 三周期状态传播 | 最终部署保存实际接受 checkpoint + revision；演示第一轮只撤除诊断，后续仍运行日志工具 |
| 可内化范围 | 支持一个显式内部计算 hook 的旁路；任意工具/上下文代码差异自动转换为监督未实现，不能对齐则保留有效候选并跳过训练 |
| 测试和证据 | 全套 127 项通过；最后日志/隔离诊断细化后另跑 21 项针对性测试通过。完整记录在 [versioned-harness-report.json](validation/versioned-harness-report.json)；旧 100 项测试及原断言保留，旧 demo 90 文件逐字节相同 |
| 真实模型/官方任务/API proposer/GPU | 本轮未运行；生命周期模型和训练 checkpoint 转换是 mock，另有 CPU 小模型真实 SGD 更新测试 |

下表记录版本化扩展前已有功能及限制。新模式的独立 Harness 接受先于训练归因，因此不再把贡献门槛失败自动解释成撤销整个代码改进。

2026-09-13：独立化重构完成。审计与删除门槛见 [MIGRATION.md](MIGRATION.md)，机器可读验收见 [validation/migration-report.json](validation/migration-report.json)。

| 项目 | 状态 |
|---|---|
| 模块runtime、候选/search/archive、四格评价、三周期外层 | 已实现，CPU测试通过 |
| 原20测试 | 文件SHA256不变，全部通过 |
| 独立proposer/trainer、数值/格式/任务/checkpoint回归 | 原20＋迁移12＋批次策略对齐7＋训练前归因9＋模型接受10＋AppWorld13＋新增benchmark29，共100项通过 |
| CPU三周期demo | 原有轨迹/分数/决策一致；77文件字节相同，8文件增加接受字段；表格模型、合成成本 |
| OPID/Meta-Harness工程依赖与patch | 已解除，指定目录与文件在前置检查后移除 |
| module on-policy trainer | 每批behavior policy共同执行H− rollout和H+评分；CPU三批更新、snapshot守卫、零模块效应与checkpoint测试通过 |
| 训练前Attribution Gate | 已实现；固定task-bootstrap门槛；不通过则discard且不训练，保留原checkpoint与residual Harness |
| Model Acceptance / Rollback | 已实现；模型accept/rollback与模块retire/retain三分支、跨周期checkpoint/Harness传播与审计记录通过CPU测试 |
| 普通外部veRL PPO adapter | 已实现接口与固定配置；实际执行未运行，默认单进程 |
| HF frozen KL reference、ALFWorld adapter | 已实现，提示fixture与外部环境mock通过；真实模型/数据未运行 |
| WebShop/Search-QA | 保留任务身份/划分约束；完整环境adapter未实现，实验未运行 |
| AppWorld | adapter、隔离worker、manifest、官方评分与训练/最终评价入口已实现，13项CPU/mock测试通过；真实engine/data/模型未运行 |
| Terminal-Bench 2/SWE-bench Pro | 容器/官方评分接口、任务导入、训练/评价入口已接入；mock通过，真实Docker/Harbor/Pro测试未运行 |
| HotpotQA | distractor固定context的search/lookup/final及joint指标已实现；真实自有worker＋两批CPU mock更新通过，官方数据未运行 |
| LawBench | 20题型官方dispatcher与完整题型最终评价入口已实现；mock通过，官方依赖未运行；逐episode训练只支持官方分数有定义的样本 |
| 真实API proposer、GPU、论文baseline | 未运行 |

没有把CPU演示称为GPU复现，也没有把新增文件名当作benchmark完成证明。原兼容模式仍使用受限表达式触发＋三类固定流程；新增版本化模式允许配置工作区内的完整 Python Harness 代码演化，在独立内核隔离下执行。

当前benchmark状态见 [BENCHMARK_ROADMAP.md](BENCHMARK_ROADMAP.md)，新增四项说明见 [benchmarks/ADAPTERS.md](benchmarks/ADAPTERS.md)。最新三周期demo90个文件与AppWorld完成时逐字节一致；验收见 [validation/benchmark-adapters-report.json](validation/benchmark-adapters-report.json)。
