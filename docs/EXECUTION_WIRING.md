# 运行入口、有效配置与记账（任务一）

本轮只修复运行接线、公开观察呈现及计数；不迁移训练框架、不新增 benchmark，不改变具名控制组合协议、任务优势或归因/退役统计门槛。

## 入口与模式

`configs/alfworld_backend.json`和 `configs/hotpotqa_backend.json` 均配置 propose、check_internalization、evaluate、train。`{python}` 使用主 CLI 的 Python 解释器，不再因子进程 PATH 选到另一个 Python。

- `execution.mode=internalization`：启动时缺少任一入口就报错。候选没有可撤除控制、监督不兼容、接受/归因 gate 不通过，仍按原选择性内化规则跳过；这是候选级结果，不是配置缺失的静默降级。
- `execution.mode=evolution_only`：只要求 propose/evaluate；版本化循环接受 H+ 后不请求 check/train，记录 `accepted_without_internalization` / `detail=evolution_only`。其他尚未配置完整入口的后端明确采用此模式。
- WebShop 本轮没有扩展。

## 配置流

主 CLI `--experiment-config` → `CommandBackend.execution_config` → 每个 `request.json` 的 `effective_config` / `effective_config_hash` → 子进程 → runner、ModuleTrainer、ModuleTeacherScorer、HF 后端与 VerlPolicy。

配置例子：`configs/alfworld_execution.json`、`configs/hotpotqa_execution.json`。可只提供部分字段：先继承 `core/execution_config.py` 的默认值，若后端配置了环境 JSON 则继承其模型/步数上限，最后应用显式实验设置；子进程只使用完整的解析结果，不再以本地默认值或设备环境变量覆盖。CLI 文件替换 backend 中的 execution 设置，不做两个用户配置间的隐式合并。

严格拒绝未知 backend 字段、实验字段（含嵌套字段）、子进程 request 字段，以及非法值或不匹配哈希。`docs/history/experiment_protocol.json` 是历史研究预注册文档，**不是**本次运行配置 schema，不能直接传给 `--experiment-config`。

| 配置字段 | 消费位置 / 含义 |
| --- | --- |
| `advantage.module_weight` | λ；显式 0 保留为 0，进入 AdvantageConfig/combine_advantages |
| `supervision` | `targeted` 或 `all`，训练 scorer 与版本化兼容性 checker 一致 |
| `tasks_per_batch` | 每批选取的任务数，上限为 train allowlist 的实际任务数 |
| `rollouts_per_task` | 每任务独立 on-policy rollout 数，训练 replica ID 保留 |
| `optimizer.learning_rate`, `weight_decay` | VerlPolicy 的 AdamW 构造参数 |
| `max_steps` | runner 决策上限；HotpotQA 等环境也使用相同上限 |
| `model.max_context/max_prompt_tokens/max_action_tokens/max_new_tokens` | 学生、H+ 评分用同 batch policy；HF 评价/兼容性及 KL reference 同样接收模型上限，不截断历史 |
| `device`, `reference_device` | 学生/评价及阶段 KL reference 设备；reference 不是 H+ scorer |

veRL actor 的 PPO 配置与优化算法保持原样，不支持的优化器/actor 字段会被拒绝。`HI_DEVICE/HI_TEACHER_DEVICE` 不再覆盖已解析的实验设备。旧单独 worker 请求没有 effective_config 时，使用并记录明确的兼容默认值；新 CommandBackend 请求始终含完整配置。已配置请求的 `--device` 若与配置冲突立即报错。

运行根目录和每个子进程目录保存 `effective_config.json`、`effective_config.sha256`。已有同名文件只允许内容一致，不能覆盖。版本化 accepted state/protocol 绑定该 hash；续跑配置不一致，或历史状态缺少可核验配置身份，拒绝恢复。最终通用 benchmark 评价读取 accepted state 的配置。预算、统计门槛、数据 manifest 身份继续单独保存在 protocol 中，不并入执行参数默认值。

## 公开观察契约

`EnvironmentStep.observation_kind`：

- `context`：当前完整的环境公开视图，替换上一次环境视图。ALFWorld、HotpotQA、AppWorld、TB2、SWE-bench Pro、LawBench 已有 adapter 的完整上下文返回显式标记此值。
- `delta`：这一次新事件的观察，按出现顺序追加。旧合成增量环境默认采用此值。

CodeRuntime 分别保留环境视图和本地工具结果；新的完整环境视图不会递归嵌套旧历史，也不会抹去之前本地工具结果。一次 dispatch 若原样回显本次环境事件的 observation，不再把它作为 Harness tool result 再写一遍。比较范围仅是本次事件，**不会对历史字符串做全局去重**。多次调用返回完全相同错误，仍有多条事件/观察。候选主动在自己 prompt 或自定义字符串中复述内容不在通用 host 的语义去重范围内；host 不猜测任意候选改写文本的来源。

EnvironmentEvent 增加 observation_kind；完整事件、reward、cost、原始 capability 调用参数/结果都保留。模型视图整理不影响事件账本累计回报，不伪造学生 response 或 loss mask。环境 context 使用 adapter 本来的公开历史窗口；不扩大或改变原窗口规则。

## 训练计数

| 字段 | 定义 |
| --- | --- |
| `planned_update_batches` | 该阶段申请的 collection/update batch 数 |
| `attempted_update_batches` | 实际进入的批次；中途失败可以小于 planned |
| `actor_update_calls` | 实际调用 student.update 的次数，失败的调用也算一次调用 |
| `optimizer_steps` | 可直接观测的 optimizer.step 次数；VerlPolicy 用 AdamW post-step hook 统计；不具备计数能力的替身返回 null |

`--planned-update-batches` 是主 CLI 的明确名称；`--train-steps` 继续作为相同参数的历史别名。LoopConfig 的 `total_train_steps` 为兼容既有 protocol 保留，但含义始终是计划批次，仍平均分配到周期。旧 worker 的 `optimizer_steps` request 字段只作历史批次预算别名；新请求使用 planned_update_batches。不再输出 `optimizer_steps_completed` 或含混的 `training_batches_completed`。

无学生决策的 batch 保留全部环境回报与成本，计入 attempted，不调用 actor。整阶段 actor_update_calls=0：保存 `status=no_actor_updates` summary，不保存新 checkpoint，不做 C/D，不宣称训练成功。外层保留 old_model + 已接受的 H+，记录 reason=no_actor_updates、model_decision=unchanged、module_decision=retain。真实训练异常保持原 rollback 路径，另有 failed summary 记录实际尝试数；无法返回的中途失败调用成本只能从已落盘的原始日志恢复，不冒充完整消费报告。

## 命令

以下从工程根目录运行。真实训练需要已经安装项目及外部 HF/veRL/ALFWorld 依赖、本地模型权重、真实任务数据、隔离支持和 [Claude proposer 配置](CLAUDE_PROPOSER.md)；本轮未启动真实模型训练。

```bash
# CPU/合成回归：实际子进程、隔离 worker 和 tensor 更新；无 GPU 声称
PYTHONPATH=src python3.12 -m unittest discover -s tests -p 'test_execution_wiring.py' -v
PYTHONPATH=src python3.12 -m unittest discover -s tests -v

# 已准备好完整 train/search/dev/retirement 分区的 ALFWorld manifest
PYTHONPATH=src python3.12 -m internalization.cli run \
  --manifest data/alfworld/manifest.json \
  --backend configs/alfworld_backend.json \
  --experiment-config configs/alfworld_execution.json \
  --checkpoint /absolute/path/to/local-hf-checkpoint \
  --harness-workspace seed_harnesses/alfworld \
  --revision-store runs/alfworld-revisions \
  --cycles 3 --planned-update-batches 300 --output runs/alfworld-task1

# HotpotQA：先使用原 build_benchmark_manifest.py 导入真实 train/heldout 源
export HI_BENCHMARK_CATALOG="$PWD/data/hotpotqa/catalog.json"
export HI_BENCHMARK_PYTHON="$(command -v python3.12)"
PYTHONPATH=src python3.12 -m internalization.cli run \
  --manifest data/hotpotqa/manifest.json --backend configs/hotpotqa_backend.json \
  --experiment-config configs/hotpotqa_execution.json \
  --checkpoint /absolute/path/to/local-hf-checkpoint \
  --harness-workspace seed_harnesses/hotpotqa \
  --revision-store runs/hotpotqa-revisions \
  --cycles 3 --planned-update-batches 300 --output runs/hotpotqa-task1
```

输出目录必须尚不存在；不覆盖旧实验。proposer 没有提出有效贡献或兼容目标时，internalization 模式也不保证发生训练；入口齐备与候选是否应训练是两件事。

## 修改清单

| 文件 / 边界 | 本轮修改 |
| --- | --- |
| `core/execution_config.py` | 严格 schema、唯一默认解析、配置 hash、不可覆盖的有效配置产物、NoActorUpdates |
| `cli.py`, `command_backend.py`, `core/interfaces.py` | CLI 配置输入、四入口 gate、跨进程完整配置、明确批次预算和响应校验 |
| `training/entrypoint.py` | 参数真实传给 runner/ModuleTrainer/HF/VerlPolicy，HotpotQA 环境上限同步、返回实际计数 |
| `training/trainer.py`, `revision_scoring.py` | 监督选择模式、批次/actor 计数、失败 summary、零更新不保存模型 |
| `training/teacher_backend.py`, `verl_backend.py` | 公共 prompt 上限、AdamW 学习率/权重衰减入参、实际 step hook |
| `training/rollout.py`, `core/trajectory.py`, `harness/code_runtime.py` | context/delta 公开观察契约和按事件呈现；旧/新版 runner 一致处理 |
| `benchmarks/{alfworld,hotpotqa,appworld,terminalbench,swebench,lawbench}.py` | 仅标注已有完整上下文返回；未新增环境功能 |
| `outer_loop.py`, `revision_loop.py` | 显式模式、配置身份与续跑校验、零更新保留旧模型和已接受 Harness |
| `scripts/evaluate_benchmark.py` | 最终评价使用 accepted state 中的执行配置 |
| `configs/*_backend.json`, `{alfworld,hotpotqa}_execution.json` | ALFWorld/HotpotQA 四入口、其他后端显式 evolution_only、完整运行配置示例 |
| `tests/test_execution_wiring.py`, `tests/fixtures/execution_worker.py` | 13 项新回归及真实子进程/CPU 测试 worker |
| `tests/test_environment_events.py`, `tests/test_appworld.py` | 仅调整零更新检查点和旧计数名的必要断言 |

## 回归与边界

新增测试覆盖：启动 gate、未知字段/hash、真实 subprocess 传参、λ=0 有非零 module signal 时不影响 outcome advantage、all/targeted、任务/rollout 数、max_steps、真实 CPU optimizer 调用统计、失败时 attempted<planned、零更新回滚保留 H+、显式 evolution_only、HotpotQA 版本化 check/evaluate/train、完整视图和重复事件呈现。

两处旧断言按本轮要求更新：`test_environment_events` 不再期待零更新阶段保存 step=0 新 checkpoint；`test_appworld` 将含混旧字段改查 attempted_update_batches。其他原测试断言保留。实际完整结果见 `validation/execution-wiring-report.json`，其中 CPU/mock 与未运行真实 GPU 项分开列出。
