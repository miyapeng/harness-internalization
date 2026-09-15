# 数据准备 → 版本化演化 → 最终评价

此次修复解决两个接口断点：旧导入器缺少 acceptance 分区；最终评价只读旧模块字段，未指定 state 时默认空 Harness。统一入口为 `core/accepted_state.py`，不改变训练公式、统计阈值、候选执行隔离或官方评分器。

## 已接受 Agent 配对

`AcceptedAgentState` 将以下字段一起保存、加载和校验：

```text
agent_state_schema: 1
checkpoint: 已接受模型目录
harness_revision: 可执行代码 revision，或兼容的完整模块表示
manifest_hash: 原完整任务 manifest 的身份
protocol: 循环预算、统计策略和 manifest 身份
protocol_hash: protocol 内容哈希
next_cycle: 下一未完成周期；中间接受状态为 null
```

新版 `cycle_XX/state.json`、`deployment.json` 都包含这组字段；`initial_agent.json` 保存起点。`accepted_harness.json` 尚未完成本周期内部决策，允许评价，但不能用于周期级恢复。

统一加载函数 `load_accepted_state(path, manifest, checkpoint=None, protocol_path=None)` 校验实际 revision 文件哈希、checkpoint 目录及 manifest/protocol 身份。最终评价只需 `--state`，模型路径从状态中读取；额外 `--checkpoint` 只能做一致性检查，不能覆盖配对。损坏、未知或缺字段的状态不会回退到空 Harness。

历史版本化 state 和含源码的旧 cycle state 可以加载，但必须保留实验 `protocol.json`（相邻目录发现，搬迁后可显式 `--protocol`）。只有模块名字的旧 deployment 不能恢复实现，应使用含 source 的 cycle state。历史模块恢复还会读取原 `attribution_policy.json`，不能重置门槛。缺少身份的孤立旧文件直接报错。

## 数据分区和启动检查

通用导入器通过 `--cycles N` 生成互斥的 train/search/dev、retirement_0..N-1 和 test。dev 的既有配对评价同时用于正式接受 Harness，不再生成或要求 acceptance 分区；旧 manifest 中的 acceptance 分区可保留，始终闲置，不挪入 train/dev/retirement/test。

通用导入器每 cohort 默认 30 个任务，需要至少 `(3 + N) * cohort_size` 个训练源任务；三周期为 **180**，两种模式的数据分区要求一致。只有 test 源的导入仍可供最终评价，但不能进入演化循环。`--legacy-modules` 仍显式标记旧模块模式。ALFWorld/AppWorld 导入入口也提供 cycles/模式参数；AppWorld 保留 scenario 三变体整组，样本不足直接报错。它的 search/dev 大小不同，不能直接套用通用导入器总量公式。

`TaskManifest.validate_loop` 在任何 rollout/proposer 调用前一次性检查整个计划；CLI 在创建代码工作区前也会校验。已存在的 manifest 不自动重切。若文件声明 manifest_hash，还要与实际内容一致。

## 运行命令

以下真实任务路径须替换为实际已准备资源；本次未运行真实模型/GPU/官方任务。

```bash
cd harness-internalization
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/build_benchmark_manifest.py --benchmark hotpotqa \
  --source /absolute/heldout-data.json --train-source /absolute/train-data.json \
  --revision PINNED_DATA_REVISION --cycles 3 --output data/hotpot-versioned

# configs/hotpotqa.json 的 catalog 必须指向此次导入的 catalog.json。
python -m internalization.cli run --manifest data/hotpot-versioned/manifest.json \
  --backend configs/hotpotqa_backend.json --checkpoint /absolute/model \
  --harness-workspace seed_harnesses/hotpotqa \
  --cycles 3 --train-steps 300 --output runs/hotpot-versioned

python scripts/evaluate_benchmark.py --manifest data/hotpot-versioned/manifest.json \
  --backend configs/hotpotqa_backend.json \
  --state runs/hotpot-versioned/deployment.json \
  --partition test --output runs/hotpot-final
```

ALFWorld/HotpotQA 的 internalization 后端已配齐五入口，缺入口会启动失败；其他未配齐的后端采用显式 evolution_only。配置与命令细节见 [运行接线](EXECUTION_WIRING.md)。

初始基线必须显式选择：

```bash
python scripts/evaluate_benchmark.py --manifest data/hotpot-versioned/manifest.json \
  --backend configs/hotpotqa_backend.json --state runs/hotpot-versioned/initial_agent.json \
  --partition test --output runs/hotpot-baseline
```

三个正式 benchmark 不接受空 Harness 的 `--baseline`；必须使用 `initial_agent.json` 评价 H0。不能仅传 checkpoint 而省略 `--state`。最终评价使用**原完整 manifest**，通过 partition 选测试集；不能另建一个只有 test 的不同 manifest 绕过配对身份。

## 周期级恢复

```bash
# 原计划三周期，cycle_00 完成后中断，后两个周期尚未使用。
python -m internalization.cli run --manifest data/hotpot-versioned/manifest.json \
  --backend configs/hotpotqa_backend.json \
  --state runs/interrupted/cycle_00/state.json --output runs/resumed
```

checkpoint、Harness、总周期数、总预算、seeds、统计策略从状态恢复。只运行 next_cycle..cycles-1，每周期仍按原总预算/原总周期数分配；不重新分配剩余预算。不同 manifest、改变循环/统计设置、无剩余周期或中间 acceptance 状态都会拒绝，原输出不覆盖。

这是周期边界恢复，不是训练 batch 续训或全局实验调度器。调用者必须使用实际中断后的最新完整状态，不应从较早状态重放已完成的评价 cohort。完成的 deployment 用于最终评价，不能借其追加周期或替换 manifest。

## 验证与限制

`tests/test_agent_state_pipeline.py` 覆盖周期分区、AppWorld 整组隔离、启动失败前零 backend 调用、状态身份与 checkpoint 拒绝、旧格式、显式基线、CLI 恢复/导入、下一周期配对传播和最终评价真实调用新工具。组合测试使用 mock 搜索分数，最终工具在真实 Linux 隔离进程执行；环境和模型为合成 fixture。

AppWorld 最终评价也使用统一加载器。LawBench 专用整类原生评价器尚无代码 revision 执行路径，本次明确报 unsupported，并在模型加载前停止；不回退为空 Harness，不用逐样本入口冒充官方完整整类评价。ALFWorld 请求式 evaluate 进程本身已显式接收 Harness，沿用原协议。

两个旧划分测试按新语义调整：通用训练池从 180 增至 270，并检查 acceptance；AppWorld 原规模 fixture 显式选 legacy，另加较大合成 fixture 验证新版分区。原 20 项测试文件未修改。验证记录见 `docs/validation/agent-pipeline-report.json`。

旧独立接受协议若从未完成周期继续，保留原 manifest 与 next_cycle，并在新输出协议中记录 acceptance_transition/from_protocol_hash；不修改旧 state/protocol，不再次使用已消费的 retirement cohort。新 dev 协议续跑保持原协议身份。
