# Model-Conditioned Proposer Scaffold Library

**Proposer Model != Proposer Scaffold.** 模型负责提议改进，scaffold 负责让该模型阅读证据、编辑文件和结束会话。本项目只保留一个方法协议和一条候选 materialization 链，不在线比较 scaffold 后择优。

## Shared Proposal Protocol

唯一方法规范仍是 [PROPOSER_SPEC.md](../src/internalization/evolution/PROPOSER_SPEC.md)，本轮字节未改。
`WorkspaceProposalHost` 从同一 verified parent 构造以下目录，所有 scaffold 完全相同：

```text
<proposal-output>/proposer_workspace/
  parent/
  candidate_0/
  candidate_1/
  evidence/scores.json
  evidence/trajectory_*.json
  evidence/workspace_policy.json
  history.json
  PROPOSER_SPEC.md
  proposal.json
```

parent、证据、history、spec 受保护；可编辑内容仅两个 candidate 和 proposal.json。模型无需返回 patch/hash；host 做 canonical diff，交给原 `materialize_candidates()`：parent hash 绑定、证据引用、search task ID 禁止硬编码、重复 revision 检查、可选目标、错误目标 fail-soft、确定性 H−。没有第二个 target proposer，没有第二次 repair session。

`proposal.json` 和原候选定义不变：恰有两个 materially distinct 候选，每个一个主要机制，含 rationale、evidence_refs、workspace、internalization。完整 Harness 的性能是首要目标；null 目标无惩罚。内化声明不构成贡献、可训练性或退役证明。host 仍按原规则决定 search/dev 接受与之后的预检、A/B、训练、C/D、退役/回滚。

工作区只包含原调度提供的全部8题分数、最多4条代表轨迹和 search-only history。不会复制整个 request：checkpoint 路径、manifest、dev、retirement、test、evaluator 和训练产物都不进入 workspace。source/观察/轨迹被视为不可信数据。

## Scaffold adapters 与信息边界

| Scaffold | 模型配对依据 | 运行与边界 | 当前验证 |
| --- | --- | --- | --- |
| Claude Code | Claude: first-party；DeepSeek: officially supported integration | 复用 MIT-attributed runner；`-p --bare`、固定 Read/Glob/Grep/Edit/Write、显式 hook、空 MCP/plugin/ambient config、独立 cwd | fake subprocess、真实宿主 guard CPU 测试；未调用真实 Claude |
| Codex | GPT/Codex: first-party | 官方 `codex exec --json`；忽略用户配置/rules、独立 home、禁 web/apps/skills/subagents、native workspace-write 且工具 network=false；外层最小 bwrap 文件系统 capsule | fake subprocess、命令/隔离构造测试；真实 namespace/CLI 未运行 |
| Qwen Code | Qwen: first-party | 官方 `qwen -p` stream-json、native turn/wall/tool 限制；本机配置不可见，显式原生 file-tool hook 拒绝其他工具；同一 bwrap capsule | fake subprocess、guard CPU 测试；真实 namespace/CLI 未运行 |

参考官方文档：[Codex CLI](https://developers.openai.com/codex/cli/reference)、[Codex security](https://developers.openai.com/codex/security)、[Qwen headless](https://qwenlm.github.io/qwen-code-docs/en/users/features/headless/)、[Qwen settings](https://qwenlm.github.io/qwen-code-docs/en/users/configuration/settings/)、[Qwen hooks](https://qwenlm.github.io/qwen-code-docs/en/users/features/hooks/)、[DeepSeek API/integrations](https://api-docs.deepseek.com/)。核对日期：2026-09-15。

Codex/Qwen capsule 仅挂载运行所需的 `/usr`、库、证书/解析配置和本次工作区，不挂载宿主 home、工程或实验父目录。工作区根及 parent/evidence 只读，仅两个 candidate 和 proposal.json 可写。临时 CLI home 仅用于本次配置与运行记录，不作为候选。CLI 必须安装在 `/usr` 下；会改写 CODEX_HOME 的用户 shell wrapper 被拒绝。需要 Linux `bwrap` 和允许使用的 namespaces；缺少任何必要能力时 fail closed，不在宿主直接运行 native CLI，不下载容器镜像。

外层 capsule 允许 CLI 自身连接显式 provider，**不是全进程禁网**。Codex 工具网络由 native sandbox 禁止，Qwen 不提供 shell/web/MCP/agent 工具，并通过可信 PreToolUse 再次限制原生文件操作。外层文件隔离和 CLI 工具权限分别负责不同边界；不把 cwd/路径白名单称为完整沙箱。真实 CLI flags/hooks/嵌套 sandbox 的有效行为仍须用固定版本做独立验证，当前 fake 通过不等于生产隔离已验证。未知 CLI 功能、退出错误、缺少有效终局事件或边界破坏均不接受产物。

## Workflow library: WHAT 与 HOW

`PROPOSER_SPEC.md` 是 WHAT。`evolution/workflows/{claude_code,codex,qwen_code}.md` 是 HOW：读规范和 parent，读全部 scores，深读代表轨迹，编辑两个独立候选，写 proposal.json，停止。三份 workflow 允许不同原生文件工具用法，候选定义不分叉。

Claude workflow 保留原显式 [SKILL.md](../.claude/skills/harness-internalization-proposer/SKILL.md) 内容，兼容已有测试；生产从包内 workflow 读取，二者字节一致由测试维护。不依赖自动 skill discovery。wheel 包含 spec、workflows、profile 示例，未 vendoring coding agent。

## 显式 profiles、Native 与 Fixed

profile 位于 [configs/proposers](../configs/proposers)。正式配置仅引用：

```json
{"proposer":{"profile":"deepseek-v41-flash_claude-code_v1"}}
```

解析后完整保存 profile_id、scaffold、provider、model、limits、workflow、spec/workflow hash、exact cli_version、experiment_mode 和 resolved_schema。unknown fields/scaffold/provider protocol 报错，不按模型名字自动路由，不 silent fallback。`deepcode` 尚未实现，显式报 unsupported scaffold。

Native Scaffold Profile 用模型对应的官方/官方兼容 scaffold，研究 practical proposer system；Fixed Scaffold Evaluation 用事先选定的同一 scaffold 和同一可支持 provider protocol，研究 proposer-model effect。`experiment_mode=native/fixed` 写入冻结配置，二者不能合并报告。开发辅助 `suggest_default_profile()` 不被正式执行入口调用。

| 示例 profile | 默认配置 | 待配置事项 |
| --- | --- | --- |
| claude-sonnet5_claude-code_v1 | Claude Code / Anthropic / claude-sonnet-5 | 精确 CLI version、账号可用性 |
| deepseek-v41-flash_claude-code_v1 | Claude Code / DeepSeek official / deepseek-flash | 精确 CLI version、账号可用性；换供应商须显式更换 provider/model |
| gpt-codex_codex_v1 | Codex / OpenAI Responses | exact model、CLI version、bwrap runtime 验证 |
| qwen-coder_qwen-code_v1 | Qwen Code / 百炼 OpenAI-compatible | exact model、CLI version、bwrap runtime 验证 |

DeepSeek 官方截至上述日期将 V4.1 Flash 的 API 名称列为 `deepseek-flash`；不擅自换成旧模型。该 provider 名称可能由供应商未来映射到更新权重，CLI/version/config hash 不能证明远端权重永远不变。记录调用时间、provider 和返回事件；需要不可变远端版本时应与供应商确认。本轮未验证实际账户。GPT/Qwen 示例中的 model=null 表示尚未选定，不能当成可直接启动的正式配置。

所有示例 `cli_version=null` 是显式待填写的模板；运行前填写 CLI `--version` 的完整输出（不含尾部换行），之后每阶段精确比较。没有安装 CLI、模型/版本未填写或版本变化都会报清晰 preflight error。库不自动调用付费服务查询模型，不自动换版本。

Claude/DeepSeek 默认12 turns、600s，Qwen12 turns、600s；tool cap 默认 null 表示未设置额外工具上限。Codex exec 当前没有可确认等价的内部 model-turn/tool cap：示例显式 null，仅强制600s wall-time；配置非 null 会拒绝，不伪造 `--max-turns` 或把一次用户 turn 记成一次内部模型调用。不同 native limit 单位不能混作相同计算预算。**search=8、candidate=2、dev=32 完全未改**。如研究要求严格相同的内部调用预算，Codex 当前不能纳入该实验设定，须另行提供可验证运行时；不能放宽后默默比较。

凭据只通过 profile 指定的 `api_key_env` 读取。Claude adapter 将该值传为 Anthropic credential，仅采用显式 base_url/model；不会要求 `model.startswith('claude-')`。不将 secret 写入 profile、effective_config 或 invocation log。正式路径不消费 HI_PROPOSER_*。更换 provider 不能继承另一个 provider 的 credential。

## 启动与恢复

ALFWorld 的 budget_v1 引用 DeepSeek/Claude Code profile；task model 仍通过现有 checkpoint 参数传入（本实验计划 Qwen3-4B），未改训练后端。WebShop/HotpotQA 暂保留 Claude profile 引用，未擅自更改 task model 或 benchmark 参数。

准备一次正式配置时，先在管理员环境确认所用 CLI version，再复制并填好对应 profile 的 model/version。可把完整 profile 对象写入新的 execution JSON 的 proposer 字段；也可先使用仓库中填好的显式 profile 引用。配置解析会将其展开到 effective_config.json 并参与原 hash。示例命令（只解析，不调用 CLI/API）：

```bash
cd /data/miyapeng/harness-internalization
PYTHONPATH=src python3.12 -c 'import json; from pathlib import Path; from internalization.core.execution_config import resolve_execution; print(json.dumps(resolve_execution(json.loads(Path("configs/budget_v1/alfworld.json").read_text())), indent=2))'
PYTHONPATH=src python3.12 scripts/propose.py --help
PYTHONPATH=src:tests python3.12 -m unittest test_proposer_scaffolds test_claude_proposer proposer_calibration.test_suite -v
```

正式三 benchmark 命令继续见 [BUDGET_V1.md](BUDGET_V1.md)，本轮没有执行。不要把 API key 放进命令行参数或 JSON。

恢复时 CLI 从 accepted state 中取已保存 resolved proposer；同名 profile reference 只核对 ID，**不读取可能改变或消失的 profile 文件**。显式换 reference、provider/model/limits/CLI version 会被拒绝；spec/workflow 字节变化也会拒绝。其他 effective_config 和原 protocol 的严格检查不变。旧 state、manifest、checkpoint、历史报告没有被重写。旧三字段 proposer 配置须另开实验显式转换，不能把它重新解释为新 profile 后静默续跑。

## 日志与 Calibration Suite

每阶段保存 `proposer_protocol.json`、统一 `session.json` 和 `scaffold_session/` 原生事件、stderr、退出码、超时状态及调用配置。统一结果包括 scaffold/version/provider/model/session、tokens/cache（若返回）、USD、tools/files、duration、limits、spec/workflow hash。未知 USD 明确为 null；Claude CLI 的内置价表不能作为 DeepSeek 等第三方 provider 的计费依据，因此第三方统一费用也为 null，原始 CLI 报价仍在原生日志；Codex 没有可靠文件读取列表和内部调用数时也为 null。原 Cost 数值账本为兼容现有接口而保留，未知计数的0不能解释为实际零成本；跨 scaffold 比较以 nullable session 字段为准。

[tests/proposer_calibration](../tests/proposer_calibration) 提供20类 synthetic toy Harness 案例及 fake runner。覆盖 prompt、多文件 import、工具、记忆、parser、named control、invalid/null declaration、distinct/duplicate、只读边界、任务 ID、证据、语法等。无 ALFWorld/WebShop/HotpotQA/retirement 数据。输出 candidate/completion/distinct/declaration-validity/boundary rates 和 tool/token/time/cost；boundary 指检测到的违规尝试，不是违规被成功执行的比例。

本轮三个 scaffold 使用同一批 scripted 输出，得到相同 host 判定只证明**基础设施一致性**，不证明哪个 Model × Scaffold 更强。未来真实 calibration 必须独立于 benchmark，预先选 profile、固定后再做正式实验。不得在 ALFWorld search 上在线选择 scaffold。

迁移：`ClaudeCodeProposer` 的工作区逻辑移到 `WorkspaceProposalHost`；原 constructor 留给注入测试，正式 entrypoint 使用 shared host。`ClaudeCodeRunner` 保留原文件与 attribution，通过 `scaffolds/claude_code.py` 转换统一 session；Codex/Qwen 是独立 native subprocess adapters。CommandBackend/outer loop/search/materialization/训练算法和 benchmark 实现均未改。

验收和未验证项以 [proposer-scaffold-library-report.json](validation/proposer-scaffold-library-report.json) 为准。当前尚未运行真实 paid proposer、真实 native namespace、官方 benchmark 或 GPU，也没有证明任何模型的 best scaffold。
