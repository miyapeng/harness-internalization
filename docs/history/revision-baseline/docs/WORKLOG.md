# 工作记录

## 2026-09-13 独立化重构

1. 先审计import、subprocess、私有类型、目录和patch依赖，输出迁移表；原文归档 `history/MIGRATION_AUDIT.md`。
2. 在修改前执行原20测试和CPU三周期demo，保存原测试SHA256、stdout、运行目录与数值/提示格式参考。未运行GPU。
3. 提取core/harness/evolution/training/evaluation/benchmarks，自有接口替代混合backend；旧flat导入只作项目内兼容。
4. 从学生H− rollout→冻结H+教师评分→task/module advantage→外部优化器→checkpoint构建独立训练通路，移除生产hindsight/skill路径。
5. 保留search/dev选择、四格撤除、预算及主benchmark协议。加入候选archive、typed trajectory完整序列化、同初始状态重复采样；保留原ALFWorld格式/parser。
6. 保留MIT、Apache与NOTICE，记录原URL/commit/逐文件迁移；未复制上游benchmark、环境或分布式训练框架。
7. 原20测试加12项迁移测试通过；原CPU demo文件字节对照一致后，按用户指定移除嵌套工程、patch与installer。随后再执行独立测试、demo、代码和路径扫描。完整结果以 `validation/migration-report.json` 为准。

验证范围为CPU合成流程、真实CPU tensor参数更新mock、公式fixture、外部环境接口mock和静态检查。API proposer、真实veRL/GPU、ALFWorld数据及其他benchmark均未运行。默认外部veRL adapter是单进程，分布式执行仍需标准框架backend接入与验证。

前一版20测试、旧patch与实现状态日志原样保存在 `history/`，不得引用为本次GPU或训练结果。

## 2026-09-13 P0-1：每batch同behavior policy监督

用户最终定义为当前batch采样策略在H−/H+下的logprob差，不再要求阶段冻结双教师差分。实现新增BehaviorPolicySnapshot，只读复用当前policy，在更新前完成H− rollout、H+模块评分与优势构造；更新后下一batch重新绑定。阶段冻结对象只保留原KL reference用途，日志与checkpoint明确两种snapshot。

新增7项CPU回归，原20测试文件未改；迁移trainer mock改为同policy不同模块context的真实监督差异。验证39项通过、三batch参数更新中两侧snapshot始终一致、无评分效应时差分仍零；CPU三周期toy原85文件保持一致。详细记录见validation/batch-policy-report.json，审计P0-1已修订；旧阶段冻结审计归档history/IMPLEMENTATION_AUDIT-stage-frozen.md。

本轮只修复P0-1。训练前A/B gate、训练退化rollback、旧模块再审计尚待实现。GPU、external veRL真实优化器、真实benchmark均未运行。

## 2026-09-13 P0-2：训练前Attribution Gate

新增AttributionPolicy与evaluate_attribution，复用task-cluster配对bootstrap。固定默认门槛为至少30个task、95%区间下界严格大于0（2000次重采样，seed42）；CLI支持预先提供configs/attribution.json。策略在任何rollout前落盘。A/B后、trainer前强制准入；未通过记录no_external_contribution、完整候选谱系/源码/哈希与归因证据，保持原checkpoint和residual Harness，不训练、不跑C/D，不重分配未使用预算。候选搜索返回含谱系的Candidate，搜索评分不变；归因事件不会提供给proposer。

新增9项归因测试；全套48项通过，无skip，原20测试文件哈希不变。持久化mock复核A=B=.5时零trainer调用且discard；正收益通过后可训练并退休；已有模块保留后连续两轮归因失败仍保留原接受状态。CPU三周期demo的原85文件与迁移前逐字节一致；新增归因侧录不改变通过门槛时的原有外层行为。日志与报告见validation/attribution-tests.txt和validation/attribution-report.json。修改前审计归档history/IMPLEMENTATION_AUDIT-before-attribution.md，当前审计P0-2已改为IMPLEMENTED。

本轮没有改变P0-1训练公式、退役规则或benchmark。GPU、真实veRL优化器、真实HF模型、真实benchmark及API proposer未运行。P0-3训练退化回滚和retained模块后续再审计仍待实现。

## 2026-09-13 P0-3：Model Acceptance / Rollback

新增evaluate_model_acceptance，复用运行前固定RetirementPolicy与task-cluster bootstrap；C−A区间上界严格小于−performance_margin时判定明显退化，独立任务不足时以单独原因拒绝模型。原退役能力/成本条件不变，最终retirement_accepted还必须model_decision=accept。默认容差0.02，95%区间，2000次bootstrap，min_tasks30，seed42；不是按均值简单回滚，也不声称“不显著退化”等于证明非劣。

outer loop显式区分accept/retire、accept/retain、rollback/retain；恢复正式checkpoint并保留原有及新增模块，下一轮搜索、评价、训练与KL reference均从此状态继续。双decision与before/proposed/accepted checkpoint分别写入retirement/state/events/deployment归档。旧decision只为模块别名；自定义Evaluator缺少双字段或返回rollback/retire会报错。outcome候选事件保存完整源码、parent/hash与判定，但不传给proposer。被拒绝checkpoint不删除，已花费预算不退回。

新增10项回归；58项测试全部通过，无skip，原20测试文件SHA256不变。A=1/C=0反例回滚、三分支跨周期、D良好但C退化时回滚优先、容差与不确定性、重复seed不足、成本失败不回滚、被拒绝toy checkpoint仍保留均验证。CPU三周期demo保持原轨迹/评分/决策：77文件逐字节一致，8文件仅新增模型接受相关字段；verify_migration新增显式schema选项，默认字节检查未放宽，反例测试确认原评分或decision变化仍被拒绝。持久化mock见runs/acceptance-proof；日志与报告见validation/acceptance-tests.txt和validation/acceptance-report.json。

P0-3修改前审计归档history/IMPLEMENTATION_AUDIT-before-acceptance.md；当前审计标为IMPLEMENTED。未运行GPU、真实HF/veRL训练、真实benchmark或API proposer；旧retained模块再审计与实际端到端模型加载验证仍未完成。

## 2026-09-13 P1-1：先接AppWorld

本轮按用户要求只启动一个benchmark，选择AppWorld。核对官方v0.1.3.post1环境/任务/评分接口与发布依赖后，自写JSON-lines环境worker、native adapter、scenario整组manifest builder、官方TGC/SGC聚合与独立最终评价脚本。环境作为外部固定包放在单独Python进程；未复制上游环境/数据/参考解。主训练/evaluate入口支持--benchmark appworld并复用P0流程，outer_loop未修改。AppWorld单独指定context/action/步数预算，ALFWorld原有默认值和优势公式不变。

worker只公开task允许字段与REPL输出，terminal reward取官方TestTracker.success；complete_task不等于success；grader异常报错。保留ground_truth_mode=minimal供官方evaluate使用，但不读取或转发隐藏答案/评测诊断。每episode独立experiment输出和worker，成本分开记execute/API调用。任务按scenario整组分区，训练/search只用官方train，dev/retirement用未训练的train/dev，test不进outer loop；统计仍按既有task ID协议，尚未新增scenario bootstrap。

新增13项AppWorld测试，71项总测试通过；包括实际子进程协议（明确的自写stub包）、重复seed隔离、终止/评分/异常、任务划分、两批CPU参数更新、评价与训练命令分派。原20测试文件不变；P0-3后的CPU三周期demo90文件逐字节一致。机器记录见validation/appworld-report.json，运行说明见benchmarks/APPWORLD.md。

真实状态：本机无AppWorld包/数据、无Docker；普通下载DNS失败，申请网络执行后PyPI代理连接返回403，pip缓存亦无AppWorld。没有实际安装、运行官方engine/task或GPU训练；CPU stub成功率不是benchmark成绩。Terminal-Bench 2为下一项，SWE-bench Pro/HotpotQA/LawBench未开工。没有声称这些任务全部属于Meta-Harness论文原实验。

## 2026-09-13：追加TB2 / SWE-bench Pro / HotpotQA / LawBench

根据用户追加范围完成四项自有适配。TB2连接官方Harbor Trial/BaseAgent lifecycle；Pro在独立容器执行学生动作并导出patch，使用外部固定官方CLI评分；HotpotQA使用固定distractor context的search/lookup和官方数学指标；LawBench保留20题型官方评分和原生单响应。新增通用隔离worker、配置、实际任务catalog/manifest导入、无模型smoke和独立最终评价入口。没有修改outer_loop、module signal、归因/退役/rollback协议，也未复制官方环境/数据/评测源码。

实现时核对了LawBench的fraction尺度、题型聚合不可用singleton均值替代、3-4/3-5跳过样本规则和2-1固定临时文件。专用LawBench最终评价先生成全部预测后按题型评分，不给无定义单例编造reward。并加入venv解释器符号链接保护、catalog stage hash、Pro agent/grader镜像一致性检查及官方grader缺失artifact报错。

29项新增回归、共100项测试通过；涵盖worker实际子进程、两批CPU toy参数更新与同batch策略、response mask、三个容器/官方评分mock接口、20题型分派、最终题型聚合、数据隔离与task hash。持久化CPU接线证明见runs/benchmark-adapters-proof：合成HotpotQA环境、4条trajectory、两次mock optimizer更新；不属于真实benchmark成绩。三周期demo90个文件与上一版逐字节一致，原20测试文件不变。

真实验收未运行：本机没有Docker、Harbor、external veRL、HF模型依赖或LawBench中文评分依赖；本轮pip download harbor==0.23.0返回无可用distribution，未安装或验证官方wheel。官方数据、Pro镜像/测试、LawBench/ChERRANT、官方HotpotQA逐样本parity和GPU均NOT YET VERIFIED。TB2/Pro/LawBench默认只有最终test分区，不能把它们自动改作训练数据。文档见benchmarks/ADAPTERS.md，机器记录见validation/benchmark-adapters-report.json。
