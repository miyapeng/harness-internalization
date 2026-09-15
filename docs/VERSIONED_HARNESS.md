# 版本化 Harness：接口、运行与验收

当前生产 proposer 为 [ClaudeCodeProposer](CLAUDE_PROPOSER.md)，APITransport/CodeProposer 仅供显式历史回归 fixture。以下早期 API 编辑说明描述宿主仍使用的 canonical path/content 协议，Claude 本身编辑工作区而不输出 patch JSON。

当前新增 schema 2 具名控制：只撤除 target_control_id，保留其他控制；训练复用非目标输出并按部署位置插入目标。完整新协议见 [NAMED_CONTROLS.md](NAMED_CONTROLS.md)。下文单 supervision hook 的描述对应仍保留的 schema 1 兼容模式。

本文保留初次版本化扩展的运行边界与历史证据；当前提案接口以 [STRUCTURED_PROPOSALS.md](STRUCTURED_PROPOSALS.md) 为准。旧三类模板生产入口已清理，不再默认回退。历史报告不改写。

## 接口迁移

| 文件 | 新接口或变化 | 原接口兼容性 |
| --- | --- | --- |
| `harness/revision.py` | `WorkspacePolicy`、`FileEdit`、`RevisionStore`、`HarnessRevision`、`InternalizationTarget` | 不改变原 `Harness` / `HarnessModule` |
| `evolution/candidate.py` | `HarnessCandidate` 记录 parent/patch/full revision/rationale/hash/evidence_refs/可选目标 | 旧 `Candidate` 已删除 |
| `evolution/code_proposer.py` | `CodeProposer.propose`；一次 transport 返回两个结构化候选和可选目标 | 不再用三种 kind 验证代码候选 |
| `evolution/proposer.py` | 公共 `APITransport.request_json` | 旧模板 proposer 已删除 |
| `evolution/revision_search.py` | 实际 search/dev 运行与选择；候选失败独立归档；公开历史过滤 | 复用原 paired interval 和评分规则 |
| `harness/code_runtime.py` | `CodeRuntime.prepare/execute`、受保护 `CapabilityBroker`、`augment_context` | 实际使用 revision 的代码、prompt 和工具注册 |
| `harness/sandbox.py`、`sandbox_worker.py` | 每次入口调用独立进程、Landlock、seccomp、资源限额 | 不把原 AST 模板限制当作任意代码沙箱 |
| `training/rollout.py`、`revision_rollout.py` | `rollout(..., harness=HarnessRevision, ...)`；显式 path/hash、独立 memory/tool registry | 旧 Harness 分支保留 |
| `training/teacher_scoring.py`、`revision_scoring.py` | `InternalizationTarget` 分派；可执行兼容性检查和同状态增强评分 | 原旧策略同步、token mask、优势构造均保留 |
| `core/interfaces.py`、`serialization.py` | 候选内置目标序列化、版本格式 `code_revision_v1` | `Components.targets` 和独立目标后端已删除 |
| `outer_loop.py`、`revision_loop.py` | `initial_harness=revision` 启用新循环，先接受改进再尝试内化 | 正式循环要求可执行 revision；保留必要历史状态读取 |
| `command_backend.py`、`training/entrypoint.py` | JSON 传输 revision/target；仅保留 `check_internalization` 可执行预检；移除独立 `target` stage | 训练器仍由原外部 veRL adapter 提供 |
| `cli.py` | 工作区/revision 初始化，`--state` 恢复已接受配对 | 状态加载与最终评价统一，见 [接线说明](ACCEPTED_AGENT_PIPELINE.md) |
| `revision_demo.py`、`scripts/demo_versioned_harness.py` | 新增明确标注 mock 的 CPU 生命周期演示 | 不作为真实 benchmark 或学习效果证据 |

训练器 `training/trainer.py`、`module_advantage.py`、`behavior_policy.py`、`verl_backend.py`、归因/退役统计实现和所有 benchmark 文件本次未修改。新增目标通过 `HarnessRevision.without(target)` 与原训练器对接；没有引入第二个阶段冻结 scorer。

## 可执行工作区与权限

通用演示实例在 `examples/versioned_harness/base/`；三个正式 benchmark 的 H0 在 `seed_harnesses/{alfworld,webshop,hotpotqa}/`，见 [SEED_HARNESSES.md](SEED_HARNESSES.md)。可编辑前缀默认是 `agent/`、`prompts/`、`tools/`、`controls/`、`config/`；可多文件修改，支持 Python、JSON、文本与 Markdown，总计最多 64 文件、256 KiB。补丁是 `FileEdit(path, before_hash, content)` 列表：原文件 UTF-8 文本 SHA256 防止套错父版本，新增文件 before_hash=null，content=null 只从新快照排除文件。不会原地删除父快照文件。

API proposer 只输出 `{"path":"tools/example.py","content":"完整的新文件内容"}`；不再要求或接受模型提供 `before_hash`。`RevisionStore.bind_patch(parent, changes)` 在宿主验证锁定父版本后计算文件 hash，再交给原 `apply()` 独立复核。归档的 `HarnessCandidate.patch` 仍含宿主填入的 `before_hash`，已有使用 `FileEdit` 的 Python/归档接口不变。API mock 或外部生成器需将响应更新为仅 path/content；新增文件 hash 为 null，删除文件仍用 content=null。父版本被篡改、重复路径、越权路径或错误基准仍拒绝。

完整树连同固定权限配置一起计算版本哈希；每个候选写入独立的新目录，文件设为只读。`config/harness.json` 指定 `entrypoint: "agent/main.py:run"`、`schema: 1` 和可选 `supervision`。入口必须真实返回以下协议：

* prepare 输入公开 history/step/memory，返回 prompt/tools/memory。
* execute 输入学生 action/history/step/memory/tools，返回 observation/memory/stop。
* 评分 hook 输入学生实际 context/step，返回 suffix/selected；非目标上下文由核心复用，不重跑 prepare。

工具描述进入真实动作 prompt。示例入口根据注册表导入并调用 `tools.log_query:run`，因此新增工具不是 archive 中未执行的源码。候选也可以自行实现其他合法注册和分派策略。局部工具、broker 模型及环境调用分别计数；成功率和 reward 仅来自受保护环境，候选返回值不能覆盖。

任意候选代码只在 `python -I -S` 子进程执行。每次调用清空 import/global 状态，每个 episode 独立 memory 和工具注册。Linux Landlock ABI 至少 3、`libseccomp.so.2` 是必需条件；本机实际检测 ABI=4。只允许读取本次快照及 Python 标准库树，禁止直接写文件、联网、fork/exec/ptrace，通过 syscall 限制和资源限额执行；环境和模型只能调用父进程 broker。运行目录和学生任务解答目录分开，不向子进程传凭据、模型对象、隐藏标签或环境句柄。缺少隔离则报错，没有不受控宿主回退。

默认每入口 30 秒 wall deadline、5 秒 CPU、256 MiB 地址空间、16 次 broker 请求、1 MiB 输出上限。broker 的同步模型/环境调用仍由现有 backend 的超时机制约束；本地 wall deadline 在 broker 返回后检查，不能抢占已阻塞的父进程 backend。此限制有明确边界，不能把路径白名单、单一子进程或上述 smoke 测试称为完整安全审计。内核接口参照 [Landlock 官方文档](https://docs.kernel.org/userspace-api/landlock.html) 和 [libseccomp 手册](https://man7.org/linux/man-pages/man3/seccomp_init.3.html)，此处集成代码为项目自写，未复制第三方实现。

## schema 1 兼容模式的选择性内化范围

当前桥只支持旁路 `config.supervision` 指定的一个内部控制 hook。H_plus 与 H_minus 的其他文件、注册、prompt、配置保持一致，hook 源码保留归档但 H_minus 不调用。它可以多次调用当前 policy 做内部计算，但在 teacher scoring 中不能接触环境；要求新观察的候选可用于正常任务运行，同时被判为不支持当前训练桥。

正式 proposer 不再单独选择目标或发现 schema-1 总 hook。一次候选响应包含 `evidence_refs` 和可选 `internalization={target_control_id, removed_behavior}`；宿主在同一 proposer 子进程调用 `InternalizationTarget.from_control()`，确定性生成只禁用一个 schema-2 independent_suffix 控制的新快照。旧 schema-1 运行/状态/合成演示的 `from_supervision()` 支持仍保留，没有第二次 API 的兼容入口。

候选 JSON 内置 full/reduced 快照和目标；声明无效时写 `proposals/candidate_i/internalization_declaration_error.json`，目标为 null，完整合法 H+ 继续 search/dev。接受后才执行预检，或无目标时保留 H+、不训练。完整输出协议、错误分类和测试迁移见 [STRUCTURED_PROPOSALS.md](STRUCTURED_PROPOSALS.md)。

兼容性检查使用 search 任务的一条真实 H_minus rollout 和同 token 评分；实际训练继续逐状态检查。预检查不等于穷尽所有状态。如果随后遇到不兼容、timeout 或训练/评价异常，保留旧模型和已接受的 H_plus，并记录失败；不部署部分训练 checkpoint。

每 batch 的公式仍为 `A_outcome + lambda * g * (logp_old(enhanced, response_ids) - old_log_prob)`，默认 lambda=0.001。H_minus 实际 prompt 和 response IDs 缓存用于监督，H_plus scorer 与 rollout 共用 `BehaviorPolicySnapshot`；优势构造结束后才 actor update。新增 hook 的工具返回和内部模型 token 不进入 response loss。原有上下文溢出拒绝、禁止隐式截断约束保持。

## 可运行命令

以下 CPU 演示已运行；需要 Python 3.12、CPU torch（兼容性快照使用）和上述 Linux 内核隔离。输出目录必须不存在，重复运行请使用新的目录名。

```bash
cd /data/miyapeng/harness-internalization
PYTHONPATH=src python3.12 scripts/demo_versioned_harness.py \
  --config configs/versioned_demo.json \
  --output runs/versioned-demo-new
PYTHONPATH=src python3.12 -m unittest discover -s tests -v
```

可用 `--scenario tool_only`、`unsupported`、`rollback`、`attribution_failed` 分别复核其余路径，仍使用原统计阈值。默认 mixed 为三个周期、每轮两个候选、三 seeds、每 cohort 30 个独立 task；300 是 mock 训练预算分配标记，**演示实际 optimizer update 数为 0**。诊断 mock 显式等待 20ms 模拟辅助调用，延迟字段按实际 wall time 记录；模型、token 和任务成功来自脚本 fixture，不能解释为真实部署收益。

已有 ALFWorld 的真实后端配置入口是 `configs/alfworld_backend.json`，本次**未运行**。准备已有模型/veRL/环境依赖和授权 proposer 环境变量后，可使用：

```bash
PYTHONPATH=src python -m internalization.cli run \
  --manifest /absolute/path/to/predeclared-manifest.json \
  --backend configs/alfworld_backend.json \
  --checkpoint /absolute/path/to/student-checkpoint \
  --harness-workspace seed_harnesses/alfworld \
  --revision-store runs/alfworld-code-revisions \
  --output runs/alfworld-code-experiment --train-steps 300
```

此命令的路径占位必须换成实际资源。真实 manifest 需预先含不重叠的 train/search/dev/test、retirement_0..2；不再要求 acceptance_0..2。旧文件中多出的 acceptance 分区保留不用，不自动重新划分既有数据或借用 retirement/test。生产 Claude session 的模型在 execution.proposer 中固定，认证使用 ANTHROPIC_API_KEY；不再读取 HI_PROPOSER_*。新版导入器按 `--cycles` 生成完整分区。周期级恢复使用 `--state .../cycle_XX/state.json`，保持原 manifest/protocol，跳过已完成周期；完成的 deployment 用于最终评价。三个正式 benchmark 的初始/最终评价都必须使用实际 `--state`。详见 [数据到最终评价接线](ACCEPTED_AGENT_PIPELINE.md)。

## 产物与验收证据

最终三周期产物在 `runs/versioned-harness-final/`（首次验证另存于 `runs/versioned-harness-mixed/`）：

| 产物 | 内容 |
| --- | --- |
| `revisions/` | 父版本、完整候选和精简版本的可执行全文件快照 |
| `experiment/cycle_00/candidate_0/candidate.json` | candidate_id、parent_revision、完整 patch、full_revision、rationale、evidence_refs、internalization_target |
| `experiment/cycle_00/internalization_target.json` | full/reduced 路径和 hash、待撤除行为、可执行 hook |
| `experiment/cycle_00/harness_acceptance.json` | 复用 dev 逐题结果与配对区间的 Harness 接受记录（decision_source=dev，无额外评价） |
| `experiment/cycle_00/compatibility/` | 实际 H_minus action IDs、同状态 hook 执行检查 |
| `experiment/cycle_00/A/` 至 `D/` | 各版本实际代码执行轨迹、任务、模型、hash、分数和成本 |
| `experiment/cycle_00/retirement.json` | 四格能力、成本与模型接受的独立判定 |
| `experiment/deployment.json`、`events.jsonl` | 正式接受状态及完整历史；proposer 只收到过滤后的 search 历史 |
| `proof.json` | mock 边界、零实际 optimizer updates、跨周期 parent/checkpoint |

混合例首轮 A=1/B=0/C=1/D=1，accept/retire 后日志工具仍在；第二、三轮从该 checkpoint + reduced revision 搜索，没有进一步收益则不训练。这些数值是合成生命周期证据。源代码演化、实际工具调用、内核隔离是真实执行；学习结论、真实 API proposer、HF/veRL/GPU、官方任务均未验证。

`tests/test_versioned_harness.py` 验证实际工具、混合四格版本、unsupported 保留、失败隔离、回滚及门槛；`test_revision_decisions.py` 验证 dev 接受、无收益、无目标跨周期、非法目标和独立 proposer mock；`test_revision_training.py` 用 CPU 小模型真实 SGD 更新验证两批同步、零效应、inactive mask、非目标上下文只生成一次。以下是当时的历史验证结果，不是新候选序列化的字节一致性承诺：原 100 项测试保留，旧三周期 90 文件逐字节一致。历史机器记录见 `docs/validation/versioned-harness-report.json`。

保留的范围限制：没有自动模块分解、DAG 搜索、可内化性分类器、任意代码差异编译器或旧 retained 目标的自动再审计调度；当前版本只支持上述可执行 hook 桥。没有降低统计阈值来让演示退役。
