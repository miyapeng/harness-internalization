# Third-party sources and notices

This project separates algorithm provenance from copied source. A repository's
top-level MIT license does not replace notices on its incorporated files.
The pins below describe inspected sources, not runtime checkouts.

| Source | Inspected revision | Use and destination |
|---|---|---|
| [OPID](https://github.com/jinyangwu/OPID) | `37a15a5f3c0f1ecc651e4be4a0c257b313fa0756` | Mechanism reference for on-policy module teacher scoring and reward combination; independently implemented in `training/{rollout,teacher_scoring,module_advantage,trainer,checkpoint}.py` |
| [Meta-Harness](https://github.com/stanford-iris-lab/meta-harness) | `0cbc31e97c9e6d24232d1dc754827c02e1ec415c` | Code proposal/search reference; minimal adaptation of `reference_examples/terminal_bench_2/claude_wrapper.py` into `evolution/claude_code.py`, as detailed below. No benchmark or complete reference agent implementation copied |
| [veRL](https://github.com/volcengine/verl/tree/v0.5.0) | external package `verl==0.5.0`, tag `v0.5.0` | Optional normal dependency. `training/verl_backend.py` calls its PPO actor and DataProto APIs; no veRL framework source vendored |

All destination paths in these tables are relative to `src/internalization/`
unless explicitly prefixed with `configs/` or `tests/`.

## Minimal Claude Code proposer scaffold (2026-09-15)

Source: [Meta-Harness claude_wrapper.py](https://github.com/stanford-iris-lab/meta-harness/blob/0cbc31e97c9e6d24232d1dc754827c02e1ec415c/reference_examples/terminal_bench_2/claude_wrapper.py),
exact commit `0cbc31e97c9e6d24232d1dc754827c02e1ec415c`.
MIT, Copyright (c) 2026 Yoonho Lee; complete text remains in
[licenses/Meta-Harness-MIT.txt](licenses/Meta-Harness-MIT.txt).

| Source functions/concepts adapted | Local destination | Adaptation |
| --- | --- | --- |
| `SessionResult`, `build_command`, `run` | `evolution/claude_code.py::ClaudeCodeSessionResult`, `ClaudeCodeRunner` | `claude -p`, stream-json, explicit cwd/model, timeout/exit and session accounting; fixed five file tools, no skip-permissions, no additional directories |
| `parse_stream_events`, file-read/write accounting, `log_session` | `evolution/claude_code.py::parse_stream`, `ClaudeCodeRunner.run` | Tool ID/result correlation, tokens/cache/USD/session metadata and raw events; cumulative result usage overrides message totals; no generic artifact extraction |
| Explicit skill loading/injection in `load_skill` / `run` | `evolution/proposer_profiles.py::workflow_text`, `ClaudeCodeRunner.run` (formerly `claude_proposer.py::skill_path`) | One host-selected workflow injected with the method spec; no skill discovery or multi-skill loader copied |

The source's WebSearch/WebFetch, Agent/subagent tools, MCP integration, plugins,
demo utilities, generic artifact extraction and TB2-specific agent behavior are not
migrated. The empty plugin directory and strict empty MCP configuration disable
ambient integrations; they are not plugin or MCP implementations.

`evolution/proposer_host.py` workspace copying/canonicalization (moved from `claude_proposer.py`) and
`evolution/claude_tool_guard.py` path checks are project-written.
`evolution/materialization.py` is this project's unchanged candidate validation
mechanism extracted from its previous single-call API adapter. The method spec
and explicit workflow are project-authored; no claim is made that upstream uses
our structured internalization candidates. Earlier historical migration reports
that stated no Meta-Harness code was adapted describe their original commits.

## Files retained or adapted from the OPID snapshot

The following files carry the original **Apache-2.0** notices, including
Copyright 2025 Nanyang Technological University (NTU), Singapore and the
verl-agent (GiGPO) team. Headers are preserved in the migrated files.

| Original path in the pinned OPID source | Migrated file | Nature of use |
|---|---|---|
| `agent_system/environments/prompts/alfworld.py` | `benchmarks/alfworld_prompts.py` | Copied prompt constants, preserving exact whitespace and original header |
| `agent_system/environments/env_package/alfworld/projection.py` | `benchmarks/alfworld_projection.py` | Copied action parser and original header; retains original validity semantics |
| `agent_system/environments/env_manager.py`, `agent_system/memory/memory.py`, `agent_system/environments/env_package/alfworld/envs.py` | `benchmarks/alfworld.py` | Adapted prompt/history formatting and text reward projection; preserves attribution. Actual environment provided by installed ALFWorld, not copied |
| `agent_system/environments/env_package/alfworld/configs/config_tw.yaml` | `configs/alfworld.yaml` | Configuration copied unchanged; provenance retained here (source has no file license header) |
| `gigpo/core_gigpo.py` | `tests/fixtures/advantage_reference.json` | Numeric outputs of original enabled functions; no function source copied. Independent mechanism implementation in `training/module_advantage.py` |
| Prompt constants and formatting methods above | `tests/fixtures/alfworld_prompt_reference.json` | Regression outputs generated from pinned original methods before removal; contains Apache-licensed prompt text |

Complete licenses and notices are preserved in:

- [OPID MIT license](licenses/OPID-MIT.txt): Copyright 2026 AIMING Lab.
- [Meta-Harness MIT license](licenses/Meta-Harness-MIT.txt): Copyright 2026 Yoonho Lee.
- [OPID NOTICE](licenses/OPID-NOTICE.txt): includes Bytedance and NTU / verl-agent attribution.
- [Apache License 2.0](licenses/Apache-2.0.txt): applies to the Apache-marked retained/adapted files and fixture text above.

OPID also incorporates veRL, verl-agent and SkillRL heritage. Its broader
trainer sources contain additional Bytedance, SGLang and ModelBest notices.
Those framework implementations, skill analyzers, hindsight/episode-skill
code and environment implementations are **not** migrated here. The original
NOTICE is retained without attempting to relicense this heritage as MIT.
Future direct imports of such source must preserve their own file notices.

The rest of this refactor reorganizes this project's pre-existing code or
implements the specified algorithms independently. The `opid_adapter.py`
compatibility name reexports project code only; it is not imported OPID code.

## AppWorld external runtime (P1-1)

The project-owned adapters `benchmarks/appworld.py`, `benchmarks/appworld_worker.py`
and `benchmarks/appworld_manifest.py` call the public API of the separately installed
[AppWorld 0.1.3.post1](https://pypi.org/project/appworld/0.1.3.post1/).
API inspection used the official [v0.1.3.post1 tag](https://github.com/StonyBrookNLP/appworld/tree/v0.1.3.post1).
No AppWorld implementation, task data, evaluator source or reference solutions
were copied into this project. The test double is independently written synthetic code.
The external package retains its Apache-2.0 and applicable bundled code/data terms.
Its published wheel SHA256 is `db77f8003982502383a50fa2974983894bd1c54f64e2fd3f7e1540d5edd037eb`;
this records the release artifact, not a claim that it was installed or locally verified.
No upstream commit hash was resolved in this network session; the version tag and
published artifact hash identify the inspected interface/release without inventing a commit.

## Additional benchmark adapters (2026-09-13)

The files below are independently written integration code, not copies of benchmark
engines, reference agents, evaluation functions or data. Existing upstream notices
above remain unchanged. External installations retain their own licenses and notices.

| External source | Inspected interface / release | Project integration files | Source commit |
| --- | --- | --- | --- |
| [Harbor](https://github.com/laude-institute/harbor), Apache-2.0 | `Trial.create/run`, `BaseAgent`, `BaseEnvironment.exec`; normal dependency `harbor==0.23.0` | `benchmarks/terminalbench.py`, `harbor_agent.py` | Not resolved; inspected main and published 0.23.0 metadata; no copied source |
| [SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os), MIT, Copyright (c) 2026 Scale AI, Inc | `swe_bench_pro_eval.py`, official image mapping and CLI/result protocol | `benchmarks/swebench.py` | Not resolved during this session; actual external checkout requires exact 40-character commit at runtime |
| [HotpotQA](https://github.com/hotpotqa/hotpot) | `hotpot_evaluate_v1.py`, distractor data schema and mathematical metric definitions | `benchmarks/hotpotqa.py` | Not resolved; no evaluator source/data copied; do not infer a repository license from another project |
| [LawBench](https://github.com/open-compass/LawBench), Apache-2.0 | `evaluation/main.py`, category scorer return contracts and data schema | `benchmarks/lawbench.py`, `aggregate.py`, `lawbench_scoring.py`, `lawbench_final.py` | Not resolved during this session; actual external checkout requires exact commit at runtime |

All integration paths in this table are under `src/internalization/`. HotpotQA
metrics are our expression of the published normalization, overlap and joint-score
equations, not a vendored upstream script. Task directories/data are imported from
user-prepared external paths with content hashes; no official tasks, hidden answers
or solutions are shipped as test fixtures. Tests use explicitly synthetic examples.

Harbor 0.23.0's published wheel SHA256 is
`8747400dbb2a5e2298e1338e17e88eba38433c0433fd700f34d1a9021bba5c37`
([release metadata](https://pypi.org/project/harbor/0.23.0/)); the wheel was not
downloaded or locally hash-verified. Pro and LawBench evaluator commits, task hashes
and image digests are recorded in runtime configuration/artifacts once real external
dependencies are prepared. No placeholder hash is asserted to be an upstream commit.


## WebShop budget_v1 adapter (2026-09-14)

`src/internalization/benchmarks/webshop.py` is original integration code calling
[WebShop's official environment](https://github.com/princeton-nlp/WebShop/blob/master/web_agent_site/envs/web_agent_text_env.py)
and its official reward through the external simulator; no engine/evaluator source
or dataset was copied. Runtime requires a clean external checkout and exact commit,
plus product/attribute/index hashes recorded in configuration. Reference API review
used the public source; no resolved runtime commit or official execution is claimed.
The module-level data locations are configured only inside the isolated worker.
Existing WebShop task identity helpers are preserved.

## Benchmark seed Harnesses (2026-09-15)

`seed_harnesses/{alfworld,webshop,hotpotqa}/agent/main.py` are project-written
passthrough entrypoints using the existing isolated runtime. No upstream runner,
environment, evaluator, model, or dataset has been bundled with these seeds.

The following Apache-2.0 sources retain their original NTU / verl-agent headers;
the repository-level OPID MIT license does not replace these file-level notices.
Reference: [jinyangwu/OPID](https://github.com/jinyangwu/OPID), commit
`37a15a5f3c0f1ecc651e4be4a0c257b313fa0756`.

| Local file | Upstream file | Reuse / adaptation |
| --- | --- | --- |
| `src/internalization/benchmarks/webshop_prompts.py` | `agent_system/environments/prompts/webshop.py` | Exact runtime `WEBSHOP_TEMPLATE_NO_HIS` literal; project-owned rendering of existing full public history and public action fields |
| `src/internalization/benchmarks/webshop_projection.py` | `agent_system/environments/env_package/webshop/projection.py` | Adapted tagged action extraction/lowercasing; retains bare commands; excludes language/think penalties and tail fallback |
| `src/internalization/benchmarks/alfworld_prompts.py`, `alfworld_projection.py`, `alfworld.py` | Existing ALFWorld prompts/projection, `env_manager.py`, `memory/memory.py` | Existing attributed implementation unchanged, source rechecked for H0 |

Protected source URLs, commits, hashes and adapter contracts are recorded in
`configs/seed_harnesses.json`; public explanation is in `docs/SEED_HARNESSES.md`.
HotpotQA uses the unchanged project-written distractor wrapper from commit
`80e985af4429061417a4979d13daa5998afb0b34`, not an official or original ReAct Harness.
The existing Apache-2.0 and MIT texts under `licenses/` remain applicable.

## Model-conditioned scaffold library (2026-09-15)

The shared host now calls `scaffolds/claude_code.py`, which adapts the existing
MIT-attributed `evolution/claude_code.py` runner above. The Meta-Harness source
commit, original functions and license remain unchanged; attribution is retained.

`scaffolds/codex.py` invokes the external OpenAI Codex CLI (`codex exec`):
https://github.com/openai/codex and https://developers.openai.com/codex/cli/reference.
`scaffolds/qwen_code.py` invokes the external Qwen Code CLI (`qwen -p`):
https://github.com/QwenLM/qwen-code and https://qwenlm.github.io/qwen-code-docs/en/users/features/headless/.
No Codex/Qwen implementation source is copied or vendored. Adapter, capsule and
normalization code is project-authored from public interfaces. External CLI
versions must be pinned explicitly in resolved profiles and are checked at
preflight; actual version/session metadata is written per proposal. Real runtimes
were not executed in this refactor.
