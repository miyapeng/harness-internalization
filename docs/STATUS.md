# 实现状态

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
