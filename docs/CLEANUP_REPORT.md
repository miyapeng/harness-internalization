# 受控代码清理报告

清理基线 HEAD：`3b096ebfa4fd2d5a6bf96ef2b9204cd9ed6e7fa4`，开始时工作区干净。仓库及其父目录未发现适用的 AGENTS.md。不覆盖用户工作、不 reset、不自动提交或推送。以下为实施前逐符号依赖审计决策；原始调用记录保存在 [dependency-audit.txt](validation/cleanup/dependency-audit.txt)，配置与受保护代码指纹在 [before.json](validation/cleanup/before.json)。

## 清理依据（先审计后实施）

依赖根包括 budget_v1 CLI、CodeProposer、代码 runtime、ModuleTrainer、真实 smoke、各 benchmark 阶段/最终评价及 AcceptedAgentState。检索覆盖 Python import、子进程 argv、JSON/YAML 路径、字符串 entrypoint、测试/示例和当前文档；不能仅凭名称或零搜索命中删除。

| 路径/符号 | 实际调用方 | 决定 | 原因 |
|---|---|---|---|
| 根 harness_modules/ | 历史状态测试读取 recovery；旧CLI/doc | recovery 样例移 tests/fixtures，其余删除 | 不再提供固定三类生产模板 |
| 平铺 harness_modules/records/manifest/backends/teacher_harness/retirement_eval.py | CLI、数据脚本、旧测试、alfworld_eval | 改显式正式路径后删除 | 无独立实现 |
| 平铺 student_harness/opid_adapter.py | 旧student/tensor测试 | 删除并迁移行为测试 | 生产内化使用revision rollout/scoring |
| evolution/proposer.py 的初始化/request_json | CodeProposer | 提取公共API transport | 原请求、认证、解析、参数和成本逐字保留 |
| 旧模板 CONTRACT/APIProposer.propose | 旧入口、migration测试 | 删除 | 新CodeProposer提示词不改 |
| TensorTeacherScorer/OPIDProvider | 仅tensor桥测试 | 删除；mask/输入信息/错误传播测试迁revision | 不保留旧phase/tensor训练协议 |
| ModuleTeacherScorer/ModuleSignal | trainer、revision_scoring、具名/版本化测试 | 保留revision路径，删除旧模块评分分支 | 同策略公式与当前scorer不变 |
| harness/runtime.py Completion/ModelBackend | HF/veRL、各benchmark/mock、代码runtime | 保留 | 共享数据/接口 |
| runtime.TeacherHarness/Guidance、harness.module Harness/HarnessModule | 历史状态反序列化、runner原生baseline/历史评价、LawBench final | 保留评价所需最小实现 | 禁止改变历史评价语义；不再可用于正式演化/训练 |
| runtime.distillation_selected、training/student_context.py | 被删除旧scorer/student桥 | 删除；选择与保留上下文行为迁revision测试 | 新桥用具名目标与真实学生上下文 |
| outer_loop 旧三类循环、core.adapters ToyBackend fallback、demo.py | 旧toy/demo测试 | 删除旧分支；保留LoopConfig/components包装 | 正式演化只接受HarnessRevision，旧状态仍可加载评价但不能续跑旧循环 |
| evolution.search.search_candidates、candidate.Candidate、archive.CandidateArchive | 旧outer/proposer/tests | 删除旧符号；保留evaluate_tasks/HarnessCandidate | 版本化归档由revision_search事件/文件承担 |
| harness/planner.py、reviewer.py、recovery.py | 旧模板生成用，无当前执行根 | 删除 | 新示例中的具名控制保持原样 |
| meta_proposer.py/train_opid.py/launch_opid_alfworld.sh | 已由propose.py/train.py替代；旧文档 | 删除并更新有效引用 | 内容/功能重复，shell仅转发训练 |
| CLI prepare-phase/validate-module/旧demo | 旧phase/三类模板 | 删除 | 保留run/retirement、独立版本化demo与smoke |
| alfworld_eval.evaluate_request | 无生产入口；所有评价脚本走training.entrypoint | 删除 | 与request/response阶段包装重叠，无独立final权限职责 |
| evaluate_alfworld.py/evaluate_benchmark.py/evaluate_appworld.py/evaluate_lawbench.py | 阶段JSON与最终评价分别使用 | 保留 | task权限、聚合与入口职责不同 |
| configs/versioned_alfworld_backend.json | 文档引用；与alfworld_backend.json逐字相同 | 删除重复，引用改alfworld_backend | 不改参数 |
| configs/budget_v1/、其他execution/backend、alfworld.yaml | 正式入口及环境/测试、不同预算配置 | 全部保留 | 参数不同不算重复 |
| configs/experiment_protocol.json | 描述性预注册/历史文档，CLI不消费 | 原字节移docs/history/experiment_protocol.json | 不作为当前execution配置 |
| scripts/verify_migration.py | 旧hash/demo兼容验收测试 | 删除；边界检查迁当前测试 | 历史报告不改，原始测试hash不再限制合理迁移 |
| docs/history/revision-baseline/src/ | 无有效测试/入口引用，Git已记录完整内容 | 删除重复源码备份 | 不创建legacy工程副本；历史报告、hash、文档保留 |
| THIRD_PARTY_NOTICES/licenses/upstream.lock | 来源与版权记录 | 保留有效记录 | 解除运行依赖不取消attribution |

## 验证结果

最终全量回归 **210 项通过，244.782 秒，0 skip**。下文记录测试迁移、首次 fixture 错误及最终验证明细。本轮仅 CPU/mock、真实子进程与隔离测试；无真实 API/GPU 训练。

## 本轮明确不处理

不实施optimizer有效更新检查、扩展真实preflight、多卡、依赖版本更新、WebShop初始化优化、数据或模型下载、真实API/GPU训练。采样顺序、seed、预算、PPO/优势/同策略评分、具名控制、A/B/C/D统计和benchmark语义保持原样。

## 实际接口迁移与必要保留项

- `outer_loop.run_outer_loop` 仅转到 `revision_loop.run_revision_loop`；`LoopConfig` 默认值和校验不变。无 revision 或旧模块状态不再启动旧循环，明确报错。`core.adapters.components_for` 保留 `Components` 和 `backend.components()`，移除 ToyBackend 自动包装。
- `evolution.proposer.APITransport` 仅保留公共初始化、HTTP/注入 transport、JSON 解析和成本记录。`CodeProposer` 只改 import/基类名；候选与目标提示词、参数、重评分接口和宿主 hash 绑定未改。
- 生产提案响应只接受版本化 `candidates`，不再接受 `candidate_sources`。训练 request 使用可执行 `InternalizationTarget`；旧模块名字符串明确报错。`ModuleTeacherScorer` 保留相同构造参数与 `ModuleSignal` 字段，只委托现有 `revision_scoring.score_revision`。
- `harness/runtime.py` 的 `Completion`、`ModelBackend` 仍被 HF/veRL 和代码 broker 使用。`TeacherHarness`/`Guidance` 与 `harness/module.py` 的 `Harness`/`HarnessModule` 仍被 `InteractionTaskRunner` 的原生/历史**评价**分支、`benchmarks/lawbench_final.evaluate`、`core.serialization.harness_from_dict`、`core.accepted_state` 调用，因此保留。它们的类 AST 与 HEAD 一致；只删除无调用方的 `ControlModule` 别名及蒸馏选择 helper。历史 inference 路径的 training-only 分支删除，实际评价调用顺序、prompt、动作、reward 不变。
- 可执行 revision schema 1 单 hook、schema 2 具名控制及其状态加载完全保留；它们不等于旧 Python `Harness`。原生基线仍须显式 `--baseline`，历史模块状态仍能读取评价；不重写任何旧 state/manifest/checkpoint/protocol。
- `CommandBackend.serialize_harness` 的薄调用包装仍被阶段请求及测试调用，委托唯一的 `core.accepted_state.serialize_harness`；它不是恢复被删除的 flat 类型导出。直接 request 的旧预算字段 `optimizer_steps` 解析别名暂留，已有 adapter 请求测试仍使用；本轮不改变运行计数协议。

正式导入路径现在是 `core.types`、`core.manifest`、`training.teacher_backend`、`harness.runtime`、`harness.module`、`evaluation.retirement`。未新建 `import *` 转发层。`scripts/build_alfworld_manifest.py` 及测试调用方均已改用正式位置。

## 配置与文档

`configs/alfworld_backend.json` 与被删除的重复文件在清理前 SHA256 相同，现有内容不变。`configs/budget_v1/`、其他不同参数的 backend/execution 配置、`alfworld.yaml` 原字节保留。`experiment_protocol.json` 原字节移至 `docs/history/`，不是当前 CLI 配置。

README、RUNBOOK、METHOD 和当前接口说明改指当前入口，正式参数及数据准备统一链接 BUDGET_V1。原 MIGRATION 与 IMPLEMENTATION_AUDIT 报告原字节移至 `docs/history/*-pre-cleanup.md`，旧位置保留说明页；报告内历史行号与命令仅描述当时版本。STATUS/WORKLOG 只新增本轮条目并说明下文是历史记录，不改原有验证结论。已记录在 Git 的 `docs/history/revision-baseline/src` 八份重复 Python 源码删除，历史报告/哈希/验证证据保留。版权、依赖版本、许可证和 upstream.lock 来源记录不变。

## 测试迁移与退役

逐个旧测试名称见 [test-migration.json](validation/cleanup/test-migration.json)。名称消失不代表其行为断言被丢弃：

| 旧测试/fixture | 本轮处理与现有断言位置 |
|---|---|
| 原20测试文件 SHA256 固定、旧 demo 文件字节/schema 比较 | 退役：仅证明已删除旧接口仍兼容，不适合作为清理约束。保留原验证报告；源码边界改由 `test_source_boundaries` 验证 |
| APIProposer/CandidateArchive mock | 迁至 `test_source_boundaries`：实际 CodeProposer + 注入 transport，两个可验证候选、父 hash、完整源码、认证/请求/成本、成功/失败历史及私有证据过滤 |
| 旧 phase/tensor bridge 七项 | 迁至 `test_revision_scoring_boundary`：同 response IDs、有效 token/padding mask、不读取未来/截断历史、任务 allowlist、评分错误传播、revision 篡改拒绝、复用非目标公开上下文 |
| `CoreTests.test_response_tokens_only` | 旧手写 mask helper 退役；新版 batch builder 的真实响应/padding/优势断言在 `test_revision_scoring_boundary`，梯度隔离与 no-op 仍由 revision/named-control training 覆盖 |
| placeholder task helper | helper 随 tensor 桥删除；真实 ALFWorld task ID/原始 gamefile、split overlap、catalog allowlist 的 adapter/manifest 测试保留，不用 parquet 假 ID 替代 |
| StudentHarness 的 retained module 输入隔离 | 迁至既有 `test_named_control_training` 和新增评分边界测试；保留具名目标关闭、非目标上下文、工具、公共历史和学生输入隔离 |
| 每批 retained guidance 重新生成 | 旧运行时专用断言退役；新版要求缓存复用，由既有 shared-context 调用次数断言及新增 `test_non_target_context_is_reused_without_regeneration` 覆盖 |
| 旧 attribution / model acceptance / ToyBackend 多周期测试 | 改用真实 revision、具名控制和 target fixture；统计 evaluator 保持真实调用，评分/训练转换为 CPU mock。保留 A/B gate、四格配对、三分支、跨周期 checkpoint/控制、候选源码/父 hash、拒绝 checkpoint 留档和信息隔离断言 |
| A/B 不通过后 discard 整个改进 | 断言改为 **保留 dev 已接受 H+，不训练**；与清理前版本化循环既有语义一致，不修改 gate 或 revision_loop |
| 非法自定义退役 verdict 必须向外抛异常 | 改验版本化循环既有 `training_or_audit_failed` + rollback/retain 状态；失败不得接受新模型，仍保留已验证 H+ |
| ALFWorld/AppWorld/HotpotQA/子进程配置训练测试 | 换成 `tests/fixtures/code_training.py` 的最小可执行 H+/H−；真实参数、batch、reward、old-policy、mask 断言继续保留。新版 seed 读取 `revision_execution.jsonl`，教师记录读取 `revision_teacher_state` |
| 历史 recovery 源码 | 只迁移实际被状态恢复测试读取的一份，置于 `tests/fixtures/historical_recovery.txt`；未复制整套旧模板 |

第一轮全量执行 210 项，其中 209 项通过；新增配置路径测试因未提供 `${...}` 环境变量 fixture 报错。补齐仅供解析、不会访问数据/服务的 fixture 后，边界组 6 项独立通过；未调整生产配置或隐藏失败断言。首次失败日志保留在 `tests-first-attempt.log`，后续完整结果见下节。

## 最终验收结果

| 检查 | 实际结果 | 证据 |
|---|---|---|
| 完整 unittest 收集与执行 | **210/210 PASS**，244.782 秒，0 skip；包含 CPU SGD、真实 JSON 子进程、Linux 隔离、工具/环境事件及多周期生命周期 | [tests.log](validation/cleanup/tests.log) |
| 当前入口 import/CLI/config | 17 个保留入口 `--help` 返回 0；backend 命令路径与配置解析、CodeProposer mock、Components 注入、旧 CLI 拒绝通过 | [checks.json](validation/cleanup/checks.json)、[边界组](validation/cleanup/boundary-tests.log) |
| 配置与受保护文件 | 已存在的 91 个记录对象 SHA256 不变；唯一有意变化的受保护文件是 CodeProposer 的 import/基类名。benchmark/评价/budget_v1 专项 34 文件原字节一致 | [before.json](validation/cleanup/before.json)、[checks.json](validation/cleanup/checks.json) |
| 传输与提示词 | API transport 的初始化/request_json AST 完全一致；CodeProposer 全文件反向替换基类名后等于 HEAD，包含所有提示词 | [checks.json](validation/cleanup/checks.json) |
| 同 fixture/seed 配置与行为 | 配置解析、8题 search 调度/候选决策、normal/no-op/shared-context 三种训练的评分输入/输出、响应 IDs/优势张量、三种退役/回滚结果完全一致 | [清理前](validation/cleanup/behavior-before.json)、[清理后](validation/cleanup/behavior-after.json)（SHA256 相同） |
| baseline/历史状态与 schema | 原生评价相关 6 个类 AST 不变；历史状态、schema 1、具名控制、state 恢复、最终评价权限回归通过 | [checks.json](validation/cleanup/checks.json)、完整测试日志 |
| 依赖与收集范围 | 当前 src/scripts/configs/examples 无已删除路径依赖；无私有上游 import/path；无历史 Python 备份被收集；正常测试收集范围是 tests | `test_source_boundaries`，完整测试日志 |
| 文档、空白与版权 | 当前说明页链接检查 0 失效；`git diff --check` 通过；pyproject、THIRD_PARTY_NOTICES、licenses、upstream.lock 原字节不变 | [checks.json](validation/cleanup/checks.json) |
| 官方环境/API/HF/veRL/GPU/真实 smoke | **未运行**。未下载数据/模型，未执行官方 benchmark、真实 proposer 或训练，不宣称 GPU 验收通过 | 本轮限定范围 |

CPU 对照可重新生成（使用新文件名，避免覆盖已有证据）：

```bash
PYTHONPATH=src:tests python3.12 tests/fixtures/cleanup_probe.py /tmp/cleanup-parity-recheck.json
PYTHONPATH=src:tests python3.12 -m unittest discover -s tests -v
git diff --check
```

`checks.json` 内保存精确指纹与入口清单。首次与中途测试错误日志保留，仅最新 `tests.log` 是最终全量结果；旧历史验证报告没有被覆盖。正文中的历史命令/文件名只作为迁移说明，不是保留入口。

## 清理后工作区与 diff

HEAD 仍为 `3b096ebfa4fd2d5a6bf96ef2b9204cd9ed6e7fa4`；本轮没有创建 commit、没有 stage、push 或远端合并。开始时工作区干净，结束时仅留下本轮删除、修改、新 fixture 与报告。完整路径状态见 [working-tree.txt](validation/cleanup/working-tree.txt)，新增/删除/修改清单见 [changes.json](validation/cleanup/changes.json)。配置/报告的原字节迁移已单独校验，Git 未暂存时会将移动显示成删除加未跟踪文件。

工作区可直接 `git diff` 审阅既有文件改动；交付的 `/tmp/harness-internalization-cleanup.patch` 另含新增源码、fixture、当前/历史说明文件，不含体积较大的 `docs/validation/cleanup/` 验证产物（这些文件已单独保存）。未保留一套 legacy 工程来替代删除。
