# Claude Code proposer：最小执行适配

最初基于 `fc72574011b9da907030c835aa056d985dc46895` 适配；当前已纳入 [共享 Scaffold Library](PROPOSER_SCAFFOLDS.md)，生产不再限定 Claude Code。
生产入口仍是 `CommandBackend → scripts/propose.py → propose stage`。
在此 adapter 下每周期一个 Claude Code session、恰好两个候选，没有 target proposer 或第二次修复 session。
single-call 在此指一次 proposal phase，并不表示 Claude session 内只有一次底层模型调用。

## 方法和接口没有分叉

唯一方法规范为
[`PROPOSER_SPEC.md`](../src/internalization/evolution/PROPOSER_SPEC.md)。
[`SKILL.md`](../.claude/skills/harness-internalization-proposer/SKILL.md)
只规定阅读、诊断、编辑两个候选、写 metadata 和停止的工作流程；其相同内容保存在包内 workflows/claude_code.md，由宿主显式读取并注入。
不通过 Claude 自动 skill discovery 加载。spec 随包安装，skill 随 wheel 安装到
`share/harness-internalization/SKILL.md`；源码运行优先读取上述唯一源文件。

Claude 编辑 `candidate_0/1`，不生成完整 patch JSON。session 成功后，宿主对比 parent 和各
candidate 的 UTF-8 文件树，按路径排序生成 path/content edits，再调用共用
`materialize_candidates()`。before_hash、证据引用、精确 search task ID 扫描、full revision
去重、可选 `InternalizationTarget.from_control()` 和 fail-soft 规则都沿用原实现。
AST 对照记录见 [invariants.json](validation/claude-proposer/invariants.json)。

`proposal.json` 恰有 `candidates`；每行恰有 `workspace/rationale/evidence_refs/internalization`，
workspace 按顺序必须为 candidate_0、candidate_1。没有新增 teacher/student prompt、预测分数
或训练/退役决定。无目标不会被惩罚；错误目标声明留下错误产物、降为 harness-only，不修复重试。
文件树非法则仅该候选无效；受保护输入被改变或 session 失败则拒绝整个 session 的产物。
完整 H+ 仍只凭既有 search/dev 接受，之后执行原预检、A/B、训练、C/D 和 rollback。

## 文件与调用边界

```text
<proposal-output>/
  proposer_workspace/                ← 唯一 Claude cwd
    parent/                       ← verified 当前 revision 的完整复制，只读
    candidate_0/                  ← 同一 parent 的独立可编辑复制
    candidate_1/                  ← 同一 parent 的独立可编辑复制
    evidence/
      scores.json                 ← 当前 search 的所有分数
      trajectory_0.json ...       ← 原调度选出的代表轨迹，含 step
      workspace_policy.json       ← 既有 Harness 可编辑路径/权限
    history.json                  ← 原 search-only 反馈；去掉宿主 revision 绝对路径
    PROPOSER_SPEC.md               ← 唯一方法规范的只读复制
    proposal.json                 ← session 写入的严格 metadata
  scaffold_session/
    isolated_config/              ← 独立 CLI 状态，不放进 Claude 工作区
    empty_plugins/
    settings.json                 ← 宿主提供的固定权限 hook
    tool_permissions.jsonl        ← 实际 hook 的 allow/deny 记录
    stdout.jsonl                  ← 原始 stream，包括无法解析的行
    stderr.log
    session.json                  ← events、tools/files、usage/cache/USD、session/CLI version
    cost.json
  proposer_protocol.json          ← model、CLI version、session、工具、spec/skill hash
  cost.json                       ← 现有 Cost 格式，外部 request/response 不变
  candidate_0.json / candidate_1.json
```

这里不会复制 manifest 的 dev/retirement/test 分区、checkpoint、训练产物、evaluator、隐藏答案。
search 公开数据按既有函数投影，history 的入参验证仍使用原 search-only 规则。旧历史原文件不改写。

工具精确配置为 `Read,Glob,Grep,Edit,Write`；不开放 Bash、Agent、WebSearch、WebFetch 或 MCP。
不使用 `--add-dir`、`--dangerously-skip-permissions` 或 repo 根目录 cwd。
继承参考 wrapper 的 `--setting-sources ""`、`--disable-slash-commands`、严格空 MCP 和空 plugin
目录；另使用 `--bare`、独立 `CLAUDE_CONFIG_DIR`、关闭自动记忆和全局 CLAUDE.md 读取，避免环境发现。
依据 [Claude CLI](https://code.claude.com/docs/en/cli-reference) 和
[bare mode](https://code.claude.com/docs/en/headless#start-faster-with-bare-mode) 的公开接口；真实 CLI 本轮未运行。

宿主通过显式 `--settings` 注入 `PreToolUse`，只准在工作区读取，只有两个候选目录及 proposal.json
可以写入；拒绝路径穿越、符号链接和其他工具。hook 是受保护宿主程序，不向模型开放 Bash。
策略实现见 `claude_tool_guard.py`，接口参照
[官方 hook 协议](https://code.claude.com/docs/en/hooks#pretooluse)。
session 后还复核只读输入字节、文件树、实际工具日志和模型标识；最终执行候选仍由原
Landlock/seccomp CodeRuntime 隔离，**没有**将路径白名单或 CLI 工具权限称为 OS 沙箱。

这依赖受信任的 Claude CLI 正确执行 flags/hooks；本轮只做 fake CLI 和真实宿主 hook 测试，
未验证真实 CLI 在此机器上的行为，也没有证明任意 CLI 漏洞/托管设置下的隔离。CLI 不支持必需
flags、没有初始化权限证据、有效工具或模型不同、意外 MCP/plugin、非零退出、max-turns 失败、
超时或缺终局 result 都拒绝候选，不回退到无边界的模式。用户/系统原有设置不被修改。

## 配置与实际运行

生产改为 [共享 Scaffold Library](PROPOSER_SCAFFOLDS.md)：`proposer: {"profile":"..."}`。
Claude Code 是其中一个 adapter，支持 Claude 和显式 Anthropic-compatible provider（包括官方 DeepSeek）。
profile 指定 provider/base_url/api_key_env/exact model/limits/exact CLI version，解析进 effective_config 和 hash。
不再要求模型名以 claude- 开头，不按字符串自动路由；正式运行未配置 model/version 会报错。
ALFWorld 引用 deepseek-v41-flash_claude-code_v1；另外两个 benchmark 保留 Claude profile。

Claude adapter 仍保持原五工具、bare、禁 ambient discovery、显式 guard、一次 session。
只有解析后的 provider 可以设置端点和凭据；不消费 HI_PROPOSER_*。未知费用记录 null。
恢复从 accepted state 的 resolved profile 读取，不依赖日后更改的同名文件。

```bash
PYTHONPATH=src python3.12 scripts/propose.py --help
PYTHONPATH=src:tests python3.12 -m unittest test_claude_proposer test_proposer_scaffolds -v
```

以下历史240项测试记录仍保留；library 新测试见新报告，不能把历史测试当作真实 CLI 证明。

## 记账和失败语义

`ClaudeCodeSessionResult` 保存 model、cwd、session_id、CLI version、原始 events、完整 stderr、
exit code、timeout、duration、token_usage（含返回的 cache counters）、total_cost_usd、
tool calls 和 files_read/files_written。工具输入与返回按 tool_use_id 关联；文件统计区分尝试和完成。
total_cost_usd 是 CLI 报告的估计，不自行按价格表推算。

最终 result usage 覆盖对应的消息累计值，不能把两者相加；重复 message ID 不重复计数。
既有 `Cost.input_tokens` 累计 uncached + cache creation + cache read，原始分项另存；
model_calls/auxiliary_calls 记录可观测的独立 assistant message 数，不把整个 session 记成一次模型调用，
也不把 max_turns 当实际调用数。tool_calls 是观察到的工具尝试数，包括失败操作。
网络重试等不可见请求不能由这些计数证明；保留 CLI 原始 usage 供核对。

超时会停止专属子进程组、收集剩余 stdout/stderr，并保存失败 session 和已有成本，不自动重试。
session 失败时顶层 propose 进程报错；已有外层停止行为不改。该失败的 wrapper 成本保存在
proposal/cost.json 与 scaffold_session/ 下，不能把它误当成功生成候选或已进行 search/dev。

## 变更与未变范围

| 文件 | 变化 |
| --- | --- |
| `evolution/claude_code.py` | 最小 Claude subprocess、stream 解析、会话成本与日志 |
| `evolution/proposer_host.py` | 复制共享候选 workspace、一次会话、canonical diff、公共 helper |
| `evolution/claude_tool_guard.py` | 显式五工具文件权限 hook；不执行候选代码 |
| `evolution/materialization.py` | 提取既有 public evidence/schema/evidence/hash/target 校验 |
| `evolution/code_proposer.py`, `proposer.py` | 保留明确的 API 回归 adapter，生产无选择入口 |
| `training/entrypoint.py` | 只在 propose 分支换构造器，evaluate/train 分支不动 |
| `core/execution_config.py`, `configs/budget_v1/*.json` | 仅 proposer schema/值变化，未知字段继续严格拒绝 |
| `PROPOSER_SPEC.md`, `.claude/skills/.../SKILL.md`, `pyproject.toml` | 唯一规范、显式 workflow、wheel 资源打包 |
| `THIRD_PARTY_NOTICES.md` | 精确 Meta-Harness 来源、MIT 与适配函数说明 |

原 `HarnessCandidate`/目标结构、`revision_loop`、`revision_search`、采样、训练、评分、mask、
同 batch old policy、优化器、A/B gate、retirement、rollback、benchmark、seed 代码均未改。
不迁移上游 benchmark/reference agent，也不增加 Web/MCP/subagent/plugin 执行能力。

来源固定为 [Meta-Harness wrapper](https://github.com/stanford-iris-lab/meta-harness/blob/0cbc31e97c9e6d24232d1dc754827c02e1ec415c/reference_examples/terminal_bench_2/claude_wrapper.py)，
只适配会话调用、stream parsing、记账和显式 skill 注入；详见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
没有把本项目的候选内化语义归因给该 wrapper。

## 验证状态

专项测试使用 injected process/session，覆盖同父两个候选、真实文件编辑到 FileEdit、原证据与
hardcoding 校验、null/合法/错误目标、去重、search-only 数据、工具/路径边界、缺 binary、
超时/非零退出、token/cache/USD/文件日志和外层单 phase。JSON subprocess 测试走真实 propose
entrypoint，但 Claude process 是 fake；宿主 hook 使用真正 Python subprocess 验证。

没有调用真实 Claude（包括 version/help）、真实模型 API、GPU 或 benchmark 下载。
真实 CLI flag/hook 兼容性、认证、模型可用性、真实成本和候选质量均为 **NOT YET VERIFIED**。
本轮不提交、不 push。此前 README.md 和 requirement.txt 安装说明改动保留原样。

最终全量 **240 项通过**（254.436 秒，无 skip），16项Claude专项测试与3个CLI help通过。离线wheel中spec/skill资源校验通过。完整日志见 [results.json](validation/claude-proposer/results.json)；可执行目录与候选目标示例见 [example.json](validation/claude-proposer/example.json)，这些路径来自本机fake session，跨机器需重新生成。
