# 独立实现迁移

> 本文记录独立化迁移当时的阶段冻结实现。随后已按最终算法改为每batch同behavior policy完成H− rollout与H+评分；当前定义和P0-1状态以 [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) 为准。

2026-09-13。本次迁移先完成 import、subprocess、路径和 patch 审计，再保存原20测试与 CPU 三周期基准，随后迁移实现。审计原文在 [history/MIGRATION_AUDIT.md](history/MIGRATION_AUDIT.md)。

## 功能迁移表

| 原功能 / 来源 | 当前实现 | 保留边界 |
|---|---|---|
| 本项目 `records.py`、`manifest.py` | `core/types.py`、`core/manifest.py`、`core/trajectory.py` | State、Cost、EvaluationResult（EpisodeResult兼容名）、真实任务ID、互斥划分；增加完整 Trajectory / Transition |
| 原单体 backend 协议 | `core/interfaces.py` | 独立 ProposerBackend、TaskRunner、TrainingBackend、RetirementEvaluator；Components 注入 |
| Meta-Harness 提案思想；本项目 `meta_proposer.py` | `evolution/proposer.py` | 当前源码、公开 execution traces、search score、历史候选；可注入 mock transport |
| 原 outer_loop 候选搜索 | `evolution/candidate.py`、`archive.py`、`search.py` | parent、内容hash、轮次version、候选ID；成功/失败归档；生成→执行→配对评分→选择 |
| 本项目模块解释器及教师流程 | `harness/module.py`、`runtime.py`、`planner.py`、`reviewer.py`、`recovery.py` | 相同源码hash、触发条件、持续窗口、调用顺序；统一动作执行 |
| OPID multi-turn rollout 和本项目 student patch 的有效路径 | `training/rollout.py`、`student_context.py` | 当前学生在H−实际交互；保存公开原始历史、实际学生prompt、response token IDs、old logprobs |
| 本项目 OPID bridge | `training/teacher_scoring.py` | 冻结教师在相同公开学生状态执行H+；指导仅用于教师打分；精确tokenization校验 |
| OPID `gigpo/core_gigpo.py` 的有效 advantage 分支 | `training/module_advantage.py` | 自写task reward归一化与module teacher signal；数值fixture对照 |
| OPID trainer / launch 的编排 | `training/trainer.py` | 每次更新前重新采样、冻结阶段教师、任务allowlist、预算、训练日志 |
| OPID内部veRL的PPO更新 | `training/verl_backend.py` → 正常外部 `verl==0.5.0` | 委托PPO、KL、entropy、梯度累积与优化器更新；不复制分布式框架 |
| 旧 checkpoint merger subprocess | `training/checkpoint.py`、policy保存API | 默认HF完整权重与tokenizer、optimizer状态、阶段manifest；不需要分片merger |
| OPID ALFWorld提示、history、projection与任务patch | `benchmarks/alfworld.py`、`alfworld_prompts.py`、`alfworld_projection.py` | 外部ALFWorld环境；仅迁移提示/解析适配；保留原Apache头与文本reward |
| WebShop/Search任务patch | `benchmarks/webshop.py`、`search_qa.py` | session split与真实task ID约束；完整环境adapter仍未实现 |
| 本项目四格评价 | `evaluation/retirement.py`、`cost.py` | 配对bootstrap、非劣门槛、成本条件不变 |
| 旧 scripts 与 argv | `scripts/propose.py`、`train.py`、`evaluate_alfworld.py`，`training/entrypoint.py` | 独立文件协议；旧脚本名只作自身入口兼容 |

`outer_loop.py` 只依赖本项目接口和数据结构。旧 flat Python 模块保留为本项目内的薄 reexport，使原20项测试与已有用户import继续可用；它们不加载上游私有类。Meta-Harness 的 benchmark/reference implementation、OPID 环境实现、hindsight analyzer、episode skill、检索技能库未迁移。`opid_adapter.py` 的兼容测试字段不进入生产训练路径。

## 没有改变的算法和协议

- 三个外层周期，每轮两个代码候选，每次最多撤除一个模块；固定预算平均分配。
- search/dev 配对增益区间下界必须严格大于0；同分按dev均值增益、较低token成本排序。提案器只收到search信息，dev/retirement标签不回流。
- 学生在H−产生自己的轨迹，教师固定为本轮初始checkpoint并从相同原始公开历史重算H+；保留模块由当前学生执行。教师规划/草案/审核不改变环境。
- 训练组内同任务重复采样共享环境seed，使用不同replica ID；仅策略采样产生分支。保存实际学生响应IDs，不把工具输出或教师指导作为学生生成token训练。
- 环境episode总reward重复用于该episode每个决策，再扣该步无效动作惩罚0.1。task统计沿原路径按决策行计权；sample std使用correction=1，单元素组均值0、std1，epsilon=1e-6。
- 模块信号为detach(teacher logprob − old student logprob)，乘学生响应mask和目标模块active mask；默认权重0.001，不normalize/clip；step outcome与episode skill分支不启用。
- PPO clip0.2、dual clip3、token mean、KL low_var0.01、entropy0.001、AdamW lr1e-6、weight decay0.01、grad clip1、epoch1、mini8/micro1；每次采样默认4任务×2重复。
- ALFWorld文本模板与最近5步历史、原动作解析器、10×won reward保留；不引入视觉或删除公开历史。
- A/B/C/D使用相同task×seed；每轮独立撤除队列，至少30任务、95%区间、2000次bootstrap、性能margin0.02、token节省至少5%和原成本门槛。
- 主实验仍是ALFWorld＋WebShop＋Search-QA；AppWorld可扩展。Terminal-Bench与SWE-bench文件仅标记未实现，未加入主实验。

## 工程边界与验证范围

`optimizer_steps` / `total_optimizer_steps` 保留旧文件协议键。它们在旧launcher中实际控制训练循环的采样/更新批次数，不保证每批只有一次AdamW.step；新实现记录 `training_batches_completed` 来说明单位。公平预算仍须另外报告实际token、样本、优化器更新和设备时间，不能将300批当作等算力。

与审计初始计划相比，默认保存方式采用HF完整checkpoint，因此无需veRL分片merger。默认外部veRL adapter为单进程串行采样＋标准PPO actor，没有移植旧Ray/FSDP/vLLM集群启动器。要运行分布式系统，应注入使用正常veRL worker的TrainingBackend；本项目未新增自己的分布式框架。这里的数学对照和CPU演示不构成GPU优化器、吞吐或真实benchmark等价性的证明。

真实环境由外部ALFWorld包提供，提示适配通过原方法输出fixture和模拟外部环境接口验证；实际数据/包端到端仍未运行。`FrozenHFBackend` 的checkpoint fingerprint基于文件元数据，实验资产还应另存权重内容checksum。保存optimizer状态已实现，但恢复分布式训练中断状态的流程未验证。

## 验收与记录

原20测试文件保持SHA256不变；新增独立proposer/trainer mock、优势数值、ALFWorld格式、checkpoint与off-policy拒绝测试。CPU三周期demo比较迁移前全部原有输出文件，新增候选archive单独记录。具体结果见 [validation/migration-report.json](validation/migration-report.json) 和 [validation/tests-after-removal.txt](validation/tests-after-removal.txt)。

删除门槛在移除嵌套仓库、patch与installer前执行；删除后再次运行测试和demo，扫描运行代码、配置、脚本、import与全项目旧路径。旧路径只允许在 `docs/history/` 的历史文档出现。CPU demo仍是表格学习和合成成本，不是论文实验。

GPU训练、外部veRL优化器执行、实际API proposer、ALFWorld真实任务、WebShop、Search-QA及所有论文baseline：**未运行**。版权、源URL、commit和迁移文件见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
