# 三个正式 benchmark 的初始 Harness H0

正式 fresh run 使用 `seed_harnesses/alfworld/`、`seed_harnesses/webshop/`、`seed_harnesses/hotpotqa/`。三者都是现有 **HarnessRevision + schema 2**，`controls=[]`、`composition=independent_suffix`，没有额外工具、辅助控制或模型调用。`examples/versioned_harness/base` 继续用于通用演示；`examples/budget_v1_smoke` 继续用于独立更新检查，均不再充当正式 H0。

本次没有修改 budget_v1 配置、数据或训练/评分算法。ALFWorld 与 HotpotQA adapter 保持原文件内容；WebShop 按本任务要求引入固定来源提示与标签解析，因此其提示不是清理前那个简短裸动作基线。既有实验产物没有重写，历史 WebShop 的真实模型结果不能当作这个新 H0 的结果。

## 提示、历史和解析各在何处

| Benchmark | 单一提示与历史实现 | 环境动作解析 | seed 的职责 |
| --- | --- | --- | --- |
| ALFWorld | `benchmarks/alfworld.py::AlfworldPrompt`，`alfworld_prompts.py`；原最近 5 步历史、当前观察和可行动作（不展示 help） | `alfworld_projection.py::alfworld_projection`，由 `AlfworldEnvironment.step` 调用 | 原样透传 adapter 渲染后的公开上下文，通过 broker 提交模型原始响应 |
| WebShop | `webshop_prompts.py::render_public_history`，由 `WebShopEnvironment.reset/step` 调用；保留已有完整原始公开页/动作历史，模板只渲染一次 | `webshop_projection.py::webshop_action`，由 adapter 调用后交给官方 `env.step` | 同上，无通用 JSON 本地工具说明 |
| HotpotQA | `hotpotqa.py::HotpotQAEnvironment`；原问题、distractor 标题、已检索内容与交互历史 | 同一 adapter 内 JSON search/lookup/final 分派 | 同上，不套 OPID/Search-R1/ReAct 模板 |

三个 `agent/main.py:run` 只是短入口，不是三个 runner。prepare 返回收到的 `history`、空工具表及原 memory；execute 调用一次 `api.environment(payload['action'])`。所有 seed 均由原 `CodeRuntime`、broker 和 `SandboxedCode` 实际执行，不在宿主 exec 候选。schema 2 没有开启任何控制，不触发 auxiliary completion。

这次保留提示/解析在受保护 adapter 中的单一实现，没有把完整 adapter、infos 或 evaluator 复制到可编辑 seed。后续候选仍可在自己的入口修改公开上下文和代码；改变受保护的底层观察/解析约定需要另行修订实验来源，不能靠候选补丁越权。

完整上下文沿用 `observation_kind=context` 替换规则；不将已套模板的历史再追加并套模板。ALFWorld 自身模板的任务/观察字段保持上游语义，其固有字段呈现不算新增包装。不同时间真实发生的重复环境事件不会去重。

## 固定来源和适配差异

受保护清单为 [`configs/seed_harnesses.json`](../configs/seed_harnesses.json)，包含每个 seed 的名称、benchmark、revision hash、来源仓库/commit/文件 SHA256、提示与解析约定，以及固定 adapter 文件 SHA256。它在候选工作区之外，不是权限配置，不经 proposer 修改；原 WorkspacePolicy 不变。

ALFWorld 与 WebShop 的参照版本固定为 [OPID `37a15a5f3c0f1ecc651e4be4a0c257b313fa0756`](https://github.com/jinyangwu/OPID/tree/37a15a5f3c0f1ecc651e4be4a0c257b313fa0756)：

- **ALFWorld 原样复用**：[提示文件](https://github.com/jinyangwu/OPID/blob/37a15a5f3c0f1ecc651e4be4a0c257b313fa0756/agent_system/environments/prompts/alfworld.py)的模板字面量和 [projection 文件](https://github.com/jinyangwu/OPID/blob/37a15a5f3c0f1ecc651e4be4a0c257b313fa0756/agent_system/environments/env_package/alfworld/projection.py)。本地旧实现即使用它们，此次未改。历史渲染沿用本地从 `env_manager.py` / `memory/memory.py` 适配的实现，最近 5 步、可行动作及解析保持原样。官方外部环境及成功奖励保持原样。
- **WebShop 原样模板 + 接口适配**：复制 [提示文件](https://github.com/jinyangwu/OPID/blob/37a15a5f3c0f1ecc651e4be4a0c257b313fa0756/agent_system/environments/prompts/webshop.py)中的 `WEBSHOP_TEMPLATE_NO_HIS` 字面量，保留 `<think>` / `<action>` 约定；文件中的行尾空格用 `\x20` 表示，运行字符串与原模板一致。`task_description` 固定指向下方公开购物指令；`current_observation` 是现有完整原始公开历史，不重复拼一份任务页、不读隐藏 goal、不采用新的历史截断窗口。`available_actions` 只取公开 `has_search_bar`、字符串 `clickables`，不透传可用动作对象中的其他字段。
- **WebShop 解析适配**：参考 [projection](https://github.com/jinyangwu/OPID/blob/37a15a5f3c0f1ecc651e4be4a0c257b313fa0756/agent_system/environments/env_package/webshop/projection.py)，从唯一 `<action>…</action>` 提取并小写化 `search[...]` / `click[...]`；仍支持原裸命令。多组标签或无匹配文本不被猜测成合法动作。没有复制上游的 think 缺失/语言惩罚或末尾 20 字符 fallback，避免引入新的有效性/奖励规则。合法动作判断与官方 reward 不变；终局 debug HTML 仍用 `Purchase submitted.` 回执替代。没有自动登录、搜索、选商品或购买。
- **HotpotQA 本项目代码**：沿用 [清理后 commit `80e985af4429061417a4979d13daa5998afb0b34` 的 distractor adapter](https://github.com/miyapeng/harness-internalization/blob/80e985af4429061417a4979d13daa5998afb0b34/src/internalization/benchmarks/hotpotqa.py)。这是 **本项目 distractor-agent wrapper**，不是官方 HotpotQA Harness，也不是原版 ReAct。search 只检索题目提供的 context、保留现有 top-k（默认 3）和排序；lookup 的参数是完整 title、返回带句子下标的文档；final 使用 answer 与 `supporting_facts=[[title, zero_based_sentence_index], ...]`。没有 Wikipedia 网络 lookup、finish 或 Search-R1 工具。评分仍是原 joint_f1，答案与金标准 supporting facts 只留在受保护评分器。

示例合法 HotpotQA 响应：

```json
{"action":"search","query":"tower"}
{"action":"lookup","title":"Tower"}
{"action":"final","answer":"Example City","supporting_facts":[["Tower",0]]}
```

这些是三次独立响应，不是一次响应中的三条动作。三个 seed 的 prepare 不接收完整 catalog、goal、参考解、infos 或评分器对象；它只接收 runtime 已有的合法公开 payload。底层用于校验 WebShop session 的 goal hash 仍在 adapter 内，绝不变成 Agent 输入。

ALFWorld 和 WebShop 的 NTU / verl-agent Apache-2.0 文件声明保留；具体迁移来源见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)，原许可证文件未删。小型上游源码已按固定 commit 核验，结果见 [provenance.json](validation/seeds/provenance.json)；未下载环境数据或模型。

## H0 绑定与实际 Agent state

| Seed | 可执行 revision SHA256 |
| --- | --- |
| alfworld_h0_v1 | `383e8ac9b0aeccc1e5773fd7865c416e011f5116d890b01c6021e5b8859f556b` |
| webshop_h0_v1 | `3b3ba1f00b8e2ca8ad1c472ce483c6920f47bfde32259c12dd8499d8c64c74d2` |
| hotpotqa_h0_v1 | `ad3b0b79ba797d30f16f1cac2e067fdb9087647435fbdefb18e682b90d9d643a` |

revision hash 按原 RevisionStore 的文件和 WorkspacePolicy 计算，运行路径是导入后的不可变快照。整体实验身份还包括 protocol 内固定 adapter 哈希和 initial checkpoint，不把代码快照 hash 冒充模型权重 hash。

fresh CLI 与程序化 revision_loop 都检查 `manifest.benchmark` 对应的注册 H0；CLI 另检查 backend benchmark。错 benchmark、通用示例、smoke 或未注册的修改版 H0 在任何 rollout/proposer 前报错。来源中的 adapter 哈希漂移也报错，不能悄悄改变基线。

启动时先导入/校验快照、写 `protocol.json` 的 `initial_checkpoint` / `initial_harness` / `initial_seed`，随后保存 `initial_agent.json`，再做搜索。`initial_seed` 含来源清单、清单 hash 和 registry hash；模型实际 snapshot ID 由现有 rollout 日志记录。原预算 effective_config/hash 保持不变。

续跑 `--state` 使用已接受 checkpoint + revision + protocol，不再读取/导入 seed 目录，也不要求当前接受的版本等于 H0。已有合法 schema 1 / 历史状态加载约定保留；没有重写任何 state、manifest 或 checkpoint。若状态所依赖的 adapter 在代码升级中变了，原历史协议的行为复现仍应使用其对应代码 commit，不能仅凭旧 state 声称跨代码版本复现。

正式运行的三个完整命令在 [BUDGET_V1.md](BUDGET_V1.md)，必须分别显式给出：

```text
ALFWorld: --harness-workspace seed_harnesses/alfworld
WebShop:  --harness-workspace seed_harnesses/webshop
HotpotQA: --harness-workspace seed_harnesses/hotpotqa
```

初始基线与最终系统都用 `scripts/evaluate_benchmark.py --state`：分别指向同一次 run 的 `initial_agent.json` 与 `deployment.json`，读取其中的实际 revision、checkpoint、manifest/protocol 和 effective_config。不传 state 不会默认空 Harness；三个 benchmark 的 `--baseline` 明确拒绝，因为空的旧 Python Harness 不代表 H0。其他 benchmark 的显式原生 baseline 支持未动。测试输出仍不进入 outer loop。

## 验收与边界

运行（不调用真实模型或 GPU）：

```bash
PYTHONPATH=src:tests python3.12 -m unittest test_seed_harnesses -v
PYTHONPATH=src python3.12 -m unittest discover -s tests -v
```

`tests/test_seed_harnesses.py` 使用 scripted model、公开观察 fixture 和真实 CodeRuntime 隔离执行，验证三 seed 的导入/执行、ALFWorld 原参考逐字一致、WebShop 标签提取和公开动作、HotpotQA JSON 三动作及 top-k/证据格式、原 response IDs 与 old-log-prob 保留、无新增辅助调用、来源保护、错 seed 拒绝、未接受候选时初始/最终 state 同 H0、恢复使用演化后的接受版本。HotpotQA 经过实际 benchmark worker；ALFWorld/WebShop 的外部官方模拟器用契约替身，无官方分数声称。

原始训练响应保留在 `training/revision_rollout.py` 的 `Transition.action/response_ids`；只有 adapter 的 native step 使用解析动作。没有用短命令重新 tokenize 来替换动作损失，现有 mask/同策略评分/控制开关/回滚测试继续运行。

旧测试的两条通用候选状态链改为从显式历史接受状态继续，保留其工具执行/恢复断言；它们不再冒充正式 HotpotQA fresh H0。空 Harness baseline 测试保留对其他 benchmark 的验证，并增加三个正式 benchmark 的拒绝约束。首次新增测试的工具调用计数、临时 catalog 独占写入问题均只修正 fixture，不改生产语义。

最终执行结果和不可变边界核验见 [validation/seeds/results.json](validation/seeds/results.json)。**真实模型生成、GPU 更新、完整官方 ALFWorld/WebShop 环境、正式基线/最终大规模评价均未执行。** 本次没有创建辅助控制，没有变更训练算法或预算，也没有自动启动实验。
