# 一次结构化提案：迁移与验收

生产执行层已适配为 [共享 workspace scaffold library](PROPOSER_SCAFFOLDS.md)。本页记录 fc725740 的候选语义及当时 API transport 证据；当前模型编辑候选目录，由宿主得到相同 path/content patch，再复用这里的校验和构造流程。独立 target stage 仍不存在。

基线为 e5ec547。开始时 README.md 和 requirement.txt 已有未提交的安装说明改动，本轮保留，不并入本次提交。

| 路径/符号 | 调用方 | 本轮处理 |
| --- | --- | --- |
| evolution/code_proposer.py 的三个 contract / propose_target | training/entrypoint.py | 合并为单一结构化候选 contract；只调用一次 transport |
| evolution/candidate.py::HarnessCandidate | proposer、CommandBackend、search、archive、demo | 增加 evidence_refs 与可选内置 InternalizationTarget，一起序列化 |
| core/interfaces.py::InternalizationTargetBackend、Components.targets | revision_loop、CommandBackend | 删除独立目标发现接口 |
| revision_loop.py 的 target provider | 通过 dev 的候选 | 直接消费内置目标；保留可执行 preflight 与后续决策 |
| command_backend.py / training/entrypoint.py 的 target stage | 后端 JSON、测试 subprocess | 移除 stage，internalization 要求四个入口 |
| evolution/revision_search.py::public_history | 下一轮 proposer | 仅输出独立记录的 search 反馈，不能通过 status 泄露 dev 判定 |
| configs/*backend.json、configs/budget_v1/*backend.json | CLI / 子进程 | 删除 target 命令；execution 数值和算法参数保持一致 |
| tests/test_code_proposer_binding.py 及具名控制/循环测试 | CPU 回归 | 将第二次目标 API 测试改为一次提案和确定性目标构造测试 |

这是一项本项目的候选协议设计，不声称 Meta-Harness、AutoSaddler 或 HarnessCompass 已实现同样机制。训练公式、优化器、采样顺序、预算数量以及 search/dev/A/B/C/D 阈值均不在本轮修改范围。

## 当前流程与接口

```text
当前公开 search 源码、8 题 scores、最多 4 条代表轨迹、search-only history
    → ONE proposer API request → 两个独立候选 (patch, rationale, evidence_refs, internalization?)
    → 宿主分别应用同一 parent，构造 H+ 和可选 H−，一起返回主进程
    → 同 8 题 search 平均正收益初筛 → 最多一个进入 dev
    → dev 接受 H+（不新增 acceptance）
        → 无内置目标：保持模型，部署 H+
        → 有目标：executable preflight → A/B → 训练 → C/D
            → accept/retire：新模型 + H−
            → accept/retain：新模型 + H+
            → rollback/retain：旧模型 + H+
```

这是 budget_v1 路径。非 budget 调度继续使用原 search/dev 配对区间规则。调用数指代码设计 proposer；task model 在正常 Harness 执行、预检和评分中的内部计算仍单独计费，并没有消失。

LLM 响应严格只有顶层 candidates。每个候选恰有四个字段；下列占位文本应在真实响应中换成实际源码与输入中的精确引用：

```json
{
  "candidates": [
    {
      "rationale": "Observed failure -> targeted mechanism -> generalizable effect",
      "evidence_refs": [{"task_id": "exact-representative-task-id", "step": 2}],
      "patch": [{"path": "controls/review.py", "content": "complete new source text"}],
      "internalization": {"target_control_id": "review_v1", "removed_behavior": "Independent public-context review"}
    },
    {
      "rationale": "A distinct harness-only improvement",
      "evidence_refs": [{"task_id": "exact-representative-task-id", "step": 1}],
      "patch": [{"path": "prompts/system.txt", "content": "complete new prompt"}],
      "internalization": null
    }
  ]
}
```

完整 H+ 必须已经注册声明的 enabled named control；示意 JSON 本身不是可直接执行的 patch。真实可执行构造与子进程示例由 `test_code_proposer_binding.py` 的 `named_row()` 和 `test_real_proposer_subprocess_returns_embedded_snapshot_and_one_call_cost` 提供。

`HarnessCandidate` 的主进程序列化新增 `evidence_refs` 和 `internalization_target`，candidate_id 同时绑定这些内容与 full/reduced hash；旧已接受 state/revision/checkpoint 不改写。旧 candidate-only 进程响应必须更新为此协议，不能使用已删除的 target 子进程补齐。归档中的旧事件不重写，也不再当作下一轮 search-only 反馈。

API 返回的 patch 不含 before_hash。宿主 bind_patch 校验 parent 后填入 hash；content=null 仅从新的快照删除文件。所有候选独立从相同 parent 应用，不能串联两个 patch。重复 full_revision 在 proposer 和 search 两层拒绝，不再重复评价；不同源码的语义/机制是否足够不同，仍需要真实评价和人工研究审阅，不能靠 hash 证明。

目标声明只提供 ID 与行为说明。宿主 `from_control()` 要求 schema 2、independent_suffix、已开启的唯一 ID；只将该控制 enabled 设为 false。工具、源代码、其他控制与顺序保持不变。null 没有惩罚，选择排序也不考虑 target 是否存在。旧 schema-1 可执行版本仍供历史 state/原生运行和合成生命周期演示使用；正式 CodeProposer 不再提议 schema-1 总 hook 的撤除。

## 失败处理与信息边界

| 情况 | 结果 |
| --- | --- |
| 顶层响应或候选数量错误 | 整次请求失败，保留原始响应与调用成本 |
| 非法 patch、错误引用、精确 search task ID 写入文件、重复 full_revision | 该候选 invalid；合法兄弟候选继续 |
| optional declaration 类型错误、ID 不存在/disabled、非 independent_suffix | 合法 H+ 保留，target=null；写声明错误，不再调用 LLM 修复 |
| search/dev 未通过 | 不接受候选，不预检、不训练 |
| H+ 被接受但无目标或 executable preflight 失败 | 保持旧模型 + H+，accepted_without_internalization |
| A/B 未通过 | 保持旧模型 + H+，attribution_failed |

evidence_refs 只能引用实际提供的代表轨迹决策 step；仅有 score 的其余任务不能作为逐步引用。有代表决策时要求至少一条引用；若 prepare 已完成、输入没有任何学生决策，允许空引用数组，仍提供全部公开环境事件。不能通过伪造学生 response 补引用。

`public_history()` 只读取在 dev 前写入的 `proposer_search_feedback`，其 status 仅为 search_positive/no_search_gain/search_failed。过去 code_candidate 的 eligible/no_gain 同时包含 dev 判定，不能继续传给 proposer。完整 dev/attribution/retirement/test 记录留在受保护归档，新的反馈中没有这些逐题结果或判定。当前父源码当然反映已经接受的实际 Agent；这不等于向优化模型开放 held-out 数据。

候选按文本精确扫描本轮所有 search task ID（包括只提供分数的任务）；该检查不能检测所有答案/实体过拟合或语义伪装。提示词明确禁止实例记忆，效果泛化仍需既有独立评价，不能声称扫描提供了泛化证明。

executable preflight 不是第二个目标发现阶段，没有代码设计模型参与。沿用原 search 任务检查 H− rollout、同一状态的目标执行、增强上下文及 response IDs/logprob/mask；它也不是任意代码的静态等价性证明。控制尝试获取新观察或执行环境动作会被现有隔离 broker 拒绝。评分按部署位置插入目标，复用非目标实际输出；同 batch old-policy、detach、mask 和优化器不改。

## 文件产物与复现

| 路径（相对周期目录） | 内容 |
| --- | --- |
| `proposals/proposer_prompt.json`, `proposer_response.json`, `cost.json` | 单次 API 输入、完整原始响应、实际调用 token/成本 |
| `proposals/candidate_i.json` | 完整候选、引用、宿主绑定 patch 和可选可执行 target |
| `proposals/candidate_i/internalization_declaration_error.json` | 声明错误、full hash、harness_only 降级、repair_model_calls=0 |
| `proposals/candidate_i_invalid.json` | 无效 patch/引用/重复候选原文与拒绝原因 |
| `candidate_i/candidate.json`, `search_screen.json`, `dev_acceptance.json` | 实际评价对应的候选、search 初筛、dev 判定 |
| `harness_acceptance.json`, `internalization_target.json`, `compatibility/` | 正式接受、使用的内置目标、可执行预检 |

省略前缀的 proposer_response.json/cost.json 与 proposer_prompt.json 位于同一 proposals 目录。没有 target_proposal 请求目录。RevisionStore 保存 parent/full/reduced 的实际只读代码树；被拒绝候选也不改变父树。

```bash
cd /data/miyapeng/harness-internalization
PYTHONPATH=src:tests python3.12 -m unittest test_code_proposer_binding -v
PYTHONPATH=src python3.12 -m unittest discover -s tests -v
PYTHONPATH=src python3.12 -m internalization.training.entrypoint --help
```

正式实验继续使用 [BUDGET_V1.md](BUDGET_V1.md) 中三个明确绑定 H0 的命令。五份 backend 配置只移除 target stage，execution 对象和全部有效数值不变；不存在 target 的其余 benchmark 配置未改。

## 测试迁移与验证边界

旧测试中的独立 target request、目标选择 decline、无 hook 零 target API 调用已退役，改为同次响应的 null/合法/错误声明测试。旧 mock Proposer 在 propose 中嵌入现成目标，不保留 propose_target 方法。两周期 mock 候选原来只改 rationale、源码完全相同，现提供不同 fixture 文件避免违反新增去重规则；分数、任务和统计阈值不改。无效目标的新断言是保留 H+ 且 target=null，原 unsupported 原因字段改为 no_embedded_internalization_target。search 历史失败状态改为 search_failed，不删除失败隔离断言。

仍保留真实 CPU 同策略两批更新、no-op、mask、梯度隔离、非目标上下文复用、单控制撤除、工具保留、版本隔离、环境事件、状态恢复、A/B、retirement、rollback 回归。新增真实 JSON proposer 子进程往返，其 API transport 为 scripted mock；沙箱工具与预检实际执行。

逐文件不变性证据见 [invariants.json](validation/structured-proposals/invariants.json)。本轮不运行真实 proposer API、官方 benchmark 数据或 GPU 训练；CPU/mock 成功不代表任务改进或可学习性已经验证。环境安装与真实训练预检问题不在本次候选协议修改范围内。

## 最终运行结果

全量 **224 项 CPU 回归通过**（250.012 秒，0 skip）；其中 19 项单次 proposer 测试及 4 个 CLI help 通过。沙箱执行和 JSON 子进程为实际运行，API/任务模型使用 scripted fixture，既有小模型 CPU 更新测试保留。`git diff --check` 与当前文档链接检查通过。完整结果、首轮失败日志和不变性检查见 [validation/structured-proposals/results.json](validation/structured-proposals/results.json)。真实 API、官方 benchmark 与 GPU 训练未执行。

可执行双候选示例及实际生成的 full/reduced hash 见 [example.json](validation/structured-proposals/example.json)，其 artifact_directory 指向本机保留的完整快照与轨迹。GitHub 中保留候选 JSON、源代码片段与验证断言；本机绝对快照路径不可直接作为另一台机器的加载路径，需用上述测试命令重新生成。
