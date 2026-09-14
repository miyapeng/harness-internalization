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


## 2026-09-13：版本化 Harness 代码演化＋选择性内化

先读取本地实现并运行原 100 项测试，全部通过；修改前 97 个源代码/配置/测试文件哈希和关键文件副本保存于 history/revision-baseline。工程原有内容均未提交，因此没有 reset、清理或整体覆盖；仅增量修改 10 个既有源码入口，其余实现以新文件加入。原有全部测试与 fixture 文件未修改。

新增 HarnessCandidate、完整可执行 HarnessRevision、带基准哈希的 FileEdit 与可选 InternalizationTarget。完整树独立存储，候选包含多文件 patch、parent、hash 与 rationale。新 CodeProposer 沿用既有授权 API transport，可独立 mock；search/dev 实际运行每个 revision，非法 patch、语法、执行超时与评价失败独立记录。允许新增工具及注册，不再用 planning/review/recovery kind 验证代码候选；旧模式原样保留。

新运行入口加载候选代码、prompt、工具配置，所有不受控 Python 代码在独立进程安装 Landlock/seccomp 后执行。实际检测本机 Landlock ABI4 与 libseccomp；只读本次代码快照及标准库，禁止直接宿主写入、联网、fork/exec，固定 CPU/内存/输出/调用预算。候选通过受保护 broker 调用当前模型和已有环境；不传模型对象、凭据、隐藏标签或环境句柄。缺少隔离会明确报错，无宿主执行回退。正常任务允许真实工具动作；评分 hook 不开放环境能力。最后补充了启动隔离错误的明确上报和失败 dispatch 前已消耗模型调用的持久化侧录。

新 outer 分支先在独立 acceptance_i 上接受 H_plus，再尝试一个 reduced_revision。无目标或不兼容接受 H_plus 不训练；A/B contribution gate 失败也保留已接受的改进。训练仍用同 batch old policy；仅增加显式内部计算 hook 旁路桥，H_minus 其他代码/工具/prompt 与 H_plus 共享，复用学生实际上下文与 response IDs，不重生成非目标指导。没有改变 trainer、优势公式、veRL、P0 统计规则或 benchmark。模型回滚保留 old_model + 已接受 H_plus；退役部署 new_model + H_minus，因此新增工具可继续存在。

完整回归 127 项通过（原 100 + 新 27），无 skip；最后日志/隔离诊断细化后另跑 21 项针对性测试全部通过。新增案例涵盖实际注册工具执行、混合四格版本、无目标跨周期传播、不兼容保留、归因失败保留、模型回滚、补丁和运行隔离、独立 proposer/mock trainer，以及 CPU 小模型两批真实 SGD 更新的 same-policy、no-op、mask、梯度隔离和非目标上下文共享。原三周期 demo 90 个文件与修改前逐字节相同，原 20 测试文件 SHA256 不变。首次隔离测试错误来自测试自身指向不存在的 tools/ 目录；更正为已有 agent/ 目录后，实际写入拒绝测试通过，没有放宽隔离策略。

默认三周期配置为 configs/versioned_demo.json，真实 ALFWorld 进程配置入口为 configs/versioned_alfworld_backend.json；命令和接口迁移详见 VERSIONED_HARNESS.md。演示运行产物保存至 runs/versioned-harness-final，完整代码、精简版本、目标、四格评价和正式部署状态均留档。演示是实际候选代码/工具/隔离 + scripted model/trainer，实际 optimizer update 数为 0；另外的训练单元测试执行了 CPU SGD。没有将脚本 checkpoint 转换解释为学习收益，没有降低 30 task / 95% / 2000 bootstrap / 0.02 非劣门槛。

未执行：真实 API proposer、HF/外部 veRL/GPU、官方任务。未增加自动分解、复杂 DAG、可内化性分类器或任意程序差异编译器；当前监督桥只接受可执行且同状态兼容的内部 hook，其他有效候选保留 H_plus。原 retained 目标自动再审计仍不是本次实现范围。机器验收记录与完整日志见 validation/versioned-harness-report.json。


## 2026-09-13：首次 GitHub 发布整理

按用户指定的 miyapeng/harness-internalization 公共仓库准备首次 main 提交。更新 README 为当前版本化 Harness/可选内化流程、可从 clone 后运行的命令及真实验证边界；补充项目自有代码 MIT LICENSE，保留原第三方版权和许可证。pyproject 添加 README、仓库 URL 和打包许可证配置。扩展忽略规则，运行产物、checkpoint、凭据、构建与解释器缓存均不提交；既有源码、测试、配置、示例与历史记录保留，没有修改算法或新增 benchmark。

发布前全套 127 项测试再次通过（226.705 秒，无 skip）；wheel 构建通过且包含 LICENSE、第三方说明和 Apache 许可证。暂存内容常见凭据格式扫描无命中，未包含嵌套 Git 仓库或运行/模型产物。首次构建因默认 /root pip cache 只读而失败，改用 /tmp 专用缓存后构建成功；未修改系统权限。网络代理拒绝 GitHub，直连后成功核对目标 PUBLIC 空仓库与已有 miyapeng 登录。提交使用仓库范围的 GitHub noreply 身份，不改变全局 Git 配置。验证详见 validation/publication-report.json；真实 GPU/模型/官方任务仍未运行。


## 2026-09-14：贯通数据准备、版本化状态与最终评价

确认两个实际断点：make_catalog_manifest 只生成三个 retirement，不生成 acceptance；evaluate_benchmark.py 仅解析旧 harness_version/active_modules，未传 state 时默认空 Harness。相同旧状态逻辑还存在于 AppWorld/LawBench 的最终脚本。未修改训练器、优势、冻结策略同步、统计门槛或官方 evaluator。

新增 core/accepted_state.py，统一保存/加载 checkpoint、代码/模块 Harness、manifest_hash、protocol/hash 和 next_cycle。新版循环写入完整配对，最终评价直接使用其中的 revision；旧状态只能在身份可验证且源码齐全时兼容。旧模块名字列表无法恢复代码则拒绝，不猜测、不回退。历史模块恢复读取原 attribution_policy 侧录，防止改变门槛。最终脚本必须选择 --state 或 --baseline，checkpoint 对 state 仅作可选一致性检查。

数据导入按 --cycles 生成独立 acceptance/retirement，默认三周期通用训练池从180增至270；保留 --legacy-modules。ALFWorld/AppWorld 同步添加模式/周期选项，AppWorld 保留完整 scenario family，数据不足不借 test。TaskManifest 增加声明哈希检查和全计划预检查。CLI --state 从原协议恢复总预算、周期、seeds、统计设置，仅运行后续周期；中间 acceptance、已完成 deployment 或不同 manifest/配置拒绝恢复。

新增17项接线回归。通用划分旧fixture增至270并检查acceptance；AppWorld旧规模fixture显式legacy，新增较大fixture检查新版独立分区；原20测试文件不变。全套144项通过（248.049秒，无skip）；旧90文件逐字节一致。持久化合成证明见 runs/accepted-agent-pipeline-proof，验证导入→mock演化评分→新版deployment→最终实际隔离工具执行；已复核之前真实落盘的旧版本化deployment可由原protocol加载。没有将mock评分或模型视为官方benchmark成绩。

LawBench专用整类原生评价尚无代码revision运行路径，本次改为明确拒绝并在模型加载前停止，不替换官方评价语义。部分既有backend未配置可选target/check_internalization时仍保留只演化不内化分支。本次未增加训练后端、GPU/模型/官方任务运行。接口、命令、周期恢复边界与未运行项记录于 ACCEPTED_AGENT_PIPELINE.md 及 validation/agent-pipeline-report.json。

## 2026-09-14：Proposer 不再计算 hash 或改写精简代码

确认 API CodeProposer 原来要求 before_hash，但输入只有文件内容；target 又让模型生成实际只允许关闭 supervision 的 patch。本轮保留所有未提交接线修改，仅在 code_proposer.py/revision.py 增加宿主确定性操作。RevisionStore.bind_patch 从已验证 parent_revision 计算 before_hash；原 apply/候选谱系校验不放宽。API 编辑格式只允许 path/content，原 FileEdit 与候选归档继续带 hash，已有 Python 提案器不受影响。

目标模型只选 removed_behavior/supervision_adapter 或返回 null；InternalizationTarget.from_supervision 验证注册入口，程序构造只将 config.supervision 置空的新快照。其他源码、工具注册、prompt 和配置语义保留。模型不再输出精简 patch；无 hook 时跳过 API，last_cost 归零并保存零调用记录。存在 hook 时仍一次行为选择调用，未宣称所有第二次调用均消除。可执行监督检查、只读评分限制、P0门槛、回滚和不支持时保留 H_plus 的分支不变。

新增 test_code_proposer_binding.py 10 项；原 test_revision_decisions.py 的 mock payload 去掉 before_hash，所有原断言保持。首轮新测试9通过、1失败，原因是测试把 JSON list 与 dataclass tuple 直接比较；改为经正式 from_dict 反序列化后比较，未放宽运行条件。修正后新增10项通过；全套154项通过（243.064秒，无skip）。测试实际在沙箱运行减法后保留的日志工具，并拒绝需要新环境观察的监督；独立 target 子进程在无 API 配置、无 hook 时返回零调用。9个训练/同步/统计/循环关键文件与本轮修改前哈希一致。git diff --check 通过。

完整日志与检查记录为 validation/proposer-binding-tests.txt、validation/proposer-binding-report.json。模型提案使用 mock transport，工具运行使用真实隔离；未运行真实 API proposer、真实模型、HF/veRL/GPU 或官方任务。未新增 benchmark、训练算法或搜索调度，既有未提交修改未覆盖。

## 2026-09-14：具名控制注册与部署一致的选择性评分

按用户批准的范围增量实现 schema 2：controls 为有序 id/entrypoint/enabled 列表，composition 指明 independent_suffix 或 sequential_suffix；无 Planning/Review/Recovery 类别限制。InternalizationTarget 增加可选 target_control_id，宿主从 ID 解析入口并只关闭该项。完整源码、工具注册和其他控制配置不变；旧 schema 1、旧目标字段和归档 hash 不重写。CodeProposer 继续只提出 path/content，目标新响应只选 ID/行为；无开启控制不调用目标 API，顺序依赖明确 unsupported。前两轮未提交接线与 hash 修复均保留。

新增 control_runtime，独立模式中每个控制读取相同基础上下文，通过独立沙箱与无环境 broker 执行；顺序模式读取累积上下文但禁止进入训练桥。新 RevisionTransition/ControlContext 保存实际基础输入和每个控制的输出、顺序及入口；序列化往返兼容旧格式。评分先复核 H-minus 组合逐字等于学生 prompt，再仅执行目标并插入 H-plus 部署位置，非目标内容不重算。具名候选的配置开关文件由 Landlock 屏蔽，prepare/execute 也不能通过读取开关重新组织基础上下文；其他工具文件仍可读。没有弱化沙箱、修改优势公式或新增程序依赖分类器。

新增17项回归：13项注册/运行/选择/结构/多周期/回滚检查，4项复用旧训练断言的 CPU SGD 测试。全部既有测试断言保留，本轮没有调整旧测试。两周期 mock 的第1轮 accept/retain recovery，第2轮 accept/retire review，实际原30-task/95%/2000-bootstrap门槛生效；另测第2轮训练退化回滚到旧模型并保留两个控制，以及顺序依赖改进保留但不训练。首位/中间/末位目标的增强 prompt 与 full 部署组合一致；随机非目标输出不重新生成；缺失/损坏组合记录拒绝。CPU 更新继续检验同批old policy、同response IDs、no-op、inactive mask、梯度隔离。

全套171项通过（244.863秒，无skip），针对性17项通过。旧三周期 demo90文件与上轮逐字节一致，原20项测试源文件哈希不变，独立源码边界检查无私有上游引用。7个训练编排/优势/策略同步/统计/外层循环文件与本轮修改前哈希一致。完整日志、parity及报告位于 validation/named-controls-*。

新增 examples/named_controls 可运行工作区及 scripts/demo_named_controls.py。实际运行产物为 runs/named-controls-proof：full 开启 recovery/review，reduced 只开启 recovery；实际执行日志工具并在同一学生状态只增强 review。演示模型/环境是脚本fixture，optimizer updates为0，不将其成功率作为学习证据。真实 API proposer、HF/外部veRL/GPU与官方任务未运行。任意 Python 程序等价性、环境元数据旁路或未访问状态不由这些有限测试证明；使用明确接口与组合边界，不能对齐则不内化。METHOD、STATUS、README、VERSIONED_HARNESS 与新 NAMED_CONTROLS 文档均更新。

## 2026-09-14：修复 prepare 终局奖励漏记并分离环境事件

确认旧 rollout_revision 在 prepare done 分支仅记录 cost/success，无 transition，因此 total_reward 丢失该次奖励。新增 EnvironmentEvent、RuntimeCall、EventTrajectory，版本化 total_reward 独立从完整环境事件求和；保留旧 Trajectory/Transition 序列化和读取。事件包含 prepare/control/execute 阶段、决策位置和控制 ID；没有学生 response 的事件照常保存奖励。reset 的公开观察单独保存，不假设 reset 提供 reward。

broker 每次环境返回立即写事件与公开 capability 参数/结果；隐藏环境对象、evaluator state 与额外参考解不进入序列化。独立具名控制使用共享日志容器，但继续不共享环境能力或其他控制输入。纯本地 dispatch 另记公开 action/observation/stop；修正 prepare 已调用环境导致后续本地调用漏计成本的问题。CodeProposer 可读取允许 search 任务的事件与实际内部调用，即使没有学生动作；原任务 allowlist 不变。事件日志不参与 teacher 输入构造。

prepare 直接完成环境时不再执行无用控制/学生生成。trainer 对全无决策 batch 记录 batch_skipped、episode return/cost，不调用 scorer 或 actor、不补采预算；下一批权重保持原快照。混合 batch 仅真实 response 构造 loss 行；原任务优势公式、episode return 重复到真实决策末 token 和 invalid-action penalty 规则均不改变。summary 区分 actor_update_calls 与 skipped batch，checkpoint step 使用实际调用更新次数。

新增 test_environment_events.py 11项全部通过，具名控制17项回归通过；完整182项通过（254.208秒，无skip），未修改旧测试断言。案例覆盖首个 prepare 终局、下一 prepare 终局、多次环境返回不重算、无事件回退禁止、公开日志/失败后奖励审计、proposer 可见性和隔离、全空/混合/跳过后训练 batch。旧CPU demo90文件逐字节一致，原20项测试源文件哈希不变；7个优势/同步/veRL/评分/统计/外层循环文件与修改前哈希一致。

持久化示例 runs/environment-events-proof 使用测试fixture与实际沙箱：首例return=3、零学生决策和模型调用；次例return=3.5、仅两个既有response token；另保存两批纯工具episode、零actor update的checkpoint/summary。实际模型/环境为mock，CPU训练回归执行了真实小模型SGD；真实API proposer、HF/veRL/GPU/官方任务未运行。完整日志与parity见 validation/environment-events-*。ENVIRONMENT_EVENTS、METHOD、STATUS、NAMED_CONTROLS和README已更新；既有未提交修改保留。旧事件缺失的历史轨迹只能兼容读取，无法追补已丢失奖励，需要重采受影响数据。
