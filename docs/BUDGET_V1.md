# budget_v1 配置与运行

本配置改变搜索/采样预算，不改变具名控制、同 batch old-policy 自蒸馏、A/B gate、C/D 退役或模型回滚。没有独立 acceptance 集合。官方数据及真实 GPU 训练的当前状态见 [数据报告](BUDGET_V1_DATA.md) 与 [验收报告](validation/budget-v1-report.json)。

## 配置和调度

三个完整 execution 配置为 `configs/budget_v1/{alfworld,webshop,hotpotqa}.json`；对应 `_backend.json` 配齐 propose/target/check_internalization/evaluate/train。非 ALFWorld 的 `_environment.json` 配置隔离 worker。budget_v1 主CLI必须显式给出可执行Harness工作区/revision或接受状态，缺少时拒绝进入旧模块循环。所有 execution 字段严格校验并记录 effective_config.json/SHA256；固定的 veRL PPO 参数也记录在 algorithm.ppo 中，不能通过本配置修改。

| 参数 | ALFWorld | WebShop | HotpotQA |
|---|---:|---:|---:|
| cycles / total planned update batches | 3 / 300 | 3 / 300 | 3 / 300 |
| tasks per batch / fresh rollouts per task | 4 / 4 | 4 / 4 | 4 / 4 |
| max_steps | 30 | 15 | 8 |
| context / prompt tokens | 8192 / 4096 | 8192 / 4096 | 16384 / 12288 |
| action / internal auxiliary output tokens | 512 / 256 | 512 / 256 | 512 / 256 |
| learning_rate / module_weight | 1e-6 / .001 | 1e-6 / .001 | 1e-6 / .001 |
| supervision | targeted | targeted | targeted |

优化器准确名称：外部 `verl==0.5.0` 的 **vanilla PPO actor**，AdamW；outcome 是 **step-weighted within-task episode-outcome normalization**，不是重新实现或换用一个新的 GRPO optimizer。公式仍为同一 old policy 的 H+ action-token logprob 减去 H− rollout cached old_log_prob，再乘 λ 和目标 mask，加到 outcome advantage。PPO clipping、KL、熵项、mini/micro batch 等保留原 ACTOR_CONFIG。

96 题 search 池先按 run_seed 打乱，三周期取连续、不重叠的 8 题。两个候选和父版本使用相同任务、相同环境 seed。只按逐题配对**平均收益严格 >0**初筛，不对 8 题做显著性接受。正收益候选按 search 平均收益、较少 token、较小候选序号排序，最多一个进入 dev；dev 失败不补选第二个。只有出现 finalist 才执行父/候选的 32 题 dev；仍要求既有 task-cluster bootstrap 区间下界 >0。首周期筛选最多 88 次任务评价（24 search + 64 dev），无 search 收益则仅 24 次；兼容性检查、A/B/C/D、训练另记成本。

proposer 得到全部 8 题分数，以及按最差/最好交替选取的最多 4 条完整代表轨迹；全部原始轨迹保存在各 runner 输出，未删除。API 代码提案的生成上限独立显式记录为 proposer.max_tokens=8192，temperature=0；256 是执行期内部控制输出上限，不能用它隐式截断多文件代码提案。

训练队列以 run_seed 打乱，保存 task hash、seed、epoch、order、position、draws。每批四个不同任务、每题四次 fresh stochastic rollout；跨 epoch 边界推迟本批已出现任务，不丢任务。每次取样先持久化不可变 sampling 快照和原子最新游标；跨 cycle、无更新、失败、模型 rollback 后都不回到列表头。cycle state/deployment 包含游标；恢复时校验任务池和 seed。每阶段计划 100 批，无训练资格时不把该预算转移到其他周期。

split_seed=42；run_seed=17，另可显式选 29/43。environment_seed=0 与 model_sampling_seed=17 是独立字段。`--run-seed 29` 同时设置队列和模型基础 seed，环境 seed 不变。每次 rollout 的模型 seed 由基础 seed、全局 batch 位置、task ID、replica 派生；子进程设置 PYTHONHASHSEED，实际模型采样前设置 Python/NumPy/Torch seed。环境在 reset 时单独设 seed。训练动作 sampling（temperature=1, top_p=1, top_k=0）；评价和内部控制 greedy，每评价任务一次。seed 不构成跨硬件逐位一致的承诺。

## 数据准备

所有命令从仓库根目录运行，使用**新的输出目录**。先准备官方原始数据及固定 revision；当前仓库不附带这些数据，也不自动下载、换源或启动实验。

```bash
cd /data/miyapeng/harness-internalization
export PYTHONPATH="$PWD/src"

# ALFWORLD_DATA 指向包含 logic/ 和 json_2.1.1/ 的外部官方数据目录。
python3.12 scripts/prepare_budget_v1.py --benchmark alfworld \
  --data-root "$ALFWORLD_DATA/json_2.1.1" --revision "$ALFWORLD_REVISION" \
  --output data/budget_v1/alfworld

python3.12 scripts/prepare_budget_v1.py --benchmark hotpotqa \
  --train "$HOTPOT_TRAIN_JSON" --distractor-dev "$HOTPOT_DISTRACTOR_DEV_JSON" \
  --revision "$HOTPOT_REVISION" --output data/budget_v1/hotpotqa
```

默认精确数量 train=2048/search=96/dev=32/三个 retirement 各128。导入器也有各用途独立的 `--train-size`、`--search-size`、`--dev-size`、`--retirement-0-size` 等参数；这用于显式的其他数据计划，正式 budget_v1 启动检查仍拒绝数量不符的 manifest。不会补入 retirement/test、减少数量或拆同源 group 来凑数。不存在足够独立任务时直接失败。旧 manifest 不修改。

WebShop 需要外部干净、固定 commit 的官方仓库、完整百万商品文件、完整 attributes/human instructions、完整 `search_engine/indexes` 及匹配独立 Python。`HI_WEBSHOP_*` 变量对应 `webshop_environment.json` 中所有路径和 SHA256；`index_hash` 用本项目 `benchmarks.common.tree_hash` 计算。`products_sha256/attributes_sha256/human_attributes_sha256` 是真实文件 SHA256，不是文件名或占位值。内部初始化 seed 固定42，导入与运行保持一致，官方自身的 goal shuffle 不修改。

```bash
export HI_BENCHMARK_CATALOG="$PWD/data/budget_v1/webshop/catalog.json"
# HI_BENCHMARK_PYTHON 指向已装官方 WebShop 依赖的独立 Python。
# 先设置 webshop_environment.json 中其余 HI_WEBSHOP_* 变量，再导出已解析资源配置。
python3.12 -c 'import json; from pathlib import Path; from internalization.benchmarks.common import BenchmarkConfig; c=BenchmarkConfig.load(Path("configs/budget_v1/webshop_environment.json")); p=Path("webshop-assets.json"); f=p.open("x"); json.dump(c.options,f,indent=2); f.close()'
"$HI_BENCHMARK_PYTHON" scripts/prepare_budget_v1.py --benchmark webshop \
  --webshop-assets webshop-assets.json --revision "$HI_WEBSHOP_COMMIT" \
  --output data/budget_v1/webshop
```

WebShop worker 使用官方 WebAgentTextEnv 的 text 页面、search/click、整数 session reset、官方 reward。完整商品配置禁止 num_products/goal filter/goal limit；goal hash 校验 task ID 映射。Agent 不接收 catalog row、goal 对象或 reward debug HTML，终局页面用提交回执表示。所有完整 reward 和事件进入受保护账本。当前每个 episode 单独加载模拟器，成本会较大，尚未做吞吐优化。适配实现及替身契约测试不等于官方完整环境已验证。

## 独立真实 smoke（先运行这一项）

`HI_TRAIN_PYTHON` 必须指向具备 external verl==0.5.0、兼容 torch/transformers 和所用环境依赖的 Python。当前已有 `/data/miyapeng/miniconda3/envs/verl/bin/python` 是 verl 0.9.0.dev0，**不符合本后端要求**，没有自动修改它。`HI_CHECKPOINT` 是本地 HF checkpoint 目录。单进程后端只使用配置指定的一张 GPU。

```bash
"$HI_TRAIN_PYTHON" -m internalization.training.smoke --benchmark alfworld \
  --checkpoint "$HI_CHECKPOINT" --manifest data/budget_v1/alfworld/manifest.json \
  --env-config configs/alfworld.yaml --experiment-config configs/budget_v1/alfworld.json \
  --output runs/budget-v1-alfworld-smoke
```

只做检查时追加 `--preflight-only`。HotpotQA/WebShop 替换 benchmark、manifest、experiment-config，并使用对应 `_environment.json`；先设置 HI_BENCHMARK_CATALOG 与 HI_BENCHMARK_PYTHON。smoke 只从 train 选8题，不运行 proposer、dev、A/B、retirement/test；固定 `examples/budget_v1_smoke` 的 public_review_v1 H+/H−，最多执行两批、要求两次真实 actor 更新。它不要求提升或退役，不降低任何正式门槛。

smoke 验证更新前同 batch 缓存 logprob、进入 actor 的 advantage/mask、参数哈希变化、KL reference 不变、完整事件 return、固定入口的逐事件上下文复建、checkpoint 重载并继续 rollout。overflow/非有限值直接失败。输出逐批审计、planned/attempted/actor/实际 optimizer 计数、设备/显存、generation/scoring tokens 与耗时、重载成本。smoke 的额外复评分探针计入成本。结果分 not_run、failed_real_attempt、passed_real_gpu；不会用 CPU/mock 替代实际运行。

## 正式运行与最终评价（本轮未启动）

以下一次只启动一个 benchmark、一个 seed。API proposer 的 HI_PROPOSER_MODEL/BASE_URL/API_KEY 沿用已授权配置，密钥不写入 effective_config。

```bash
"$HI_TRAIN_PYTHON" -m internalization.cli run \
  --manifest data/budget_v1/alfworld/manifest.json \
  --backend configs/budget_v1/alfworld_backend.json \
  --experiment-config configs/budget_v1/alfworld.json --run-seed 17 \
  --checkpoint "$HI_CHECKPOINT" --harness-workspace examples/versioned_harness/base \
  --output runs/budget-v1-alfworld-seed17
```

WebShop/HotpotQA 将上述三个文件路径及输出名称的 alfworld 替换为对应 benchmark，配置其环境变量即可。另提供 `--run-seed 29` 和 `--run-seed 43`，使用各自新输出；不能改变已记录实验的 seed 来续跑。继续未完成周期用 `--state runs/.../cycle_00/state.json` 代替 checkpoint/workspace，输出仍用新目录。

```bash
"$HI_TRAIN_PYTHON" scripts/evaluate_benchmark.py \
  --manifest data/budget_v1/alfworld/manifest.json \
  --backend configs/budget_v1/alfworld_backend.json \
  --state runs/budget-v1-alfworld-seed17/deployment.json \
  --partition test_valid_seen --output runs/budget-v1-alfworld-final-seen
```

valid_unseen 单独用 test_valid_unseen；HotpotQA/WebShop 用 test。最终脚本从接受状态读取实际 checkpoint、revision 和 effective_config，budget_v1 默认每题一次，不退回空 Harness。只有显式 `--baseline --checkpoint ...` 才评价初始基线。

## 接口修改位置

| 文件 | 增量接口/行为 |
|---|---|
| core/execution_config.py、cli.py | schedule/seeds/proposer/固定 algorithm 参数 schema；预算和完整配置校验 |
| core/sampling.py | search_schedule、TaskQueue、模型/进程 seed；精确 manifest 大小检查 |
| evolution/revision_search.py、revision_loop.py | budget_v1 初筛、单 dev finalist、代表轨迹、真实 task ID、游标状态 |
| core/accepted_state.py、core/interfaces.py、command_backend.py | sampling_state 跨子进程/周期/续跑，失败/超时也保留已取样位置 |
| training/entrypoint.py、trainer.py、rollout.py、revision_rollout.py | 参数与种子执行；随机队列和每批四题四 rollout |
| training/revision_scoring.py | 兼容性探测采用配置环境 seed；评分公式不变 |
| training/teacher_backend.py | 分开记录生成/评分 token 和时间；NaN/Inf 检查 |
| benchmarks/budget_data.py、scripts/prepare_budget_v1.py | 独立分区数量、来源/类型/同源 group、不可覆盖导入 |
| benchmarks/webshop.py、worker.py、common.py | 真实官方环境接线、隔离 worker、原 task_id/split helper 保留 |
| training/smoke.py、examples/budget_v1_smoke/ | 独立真实更新验收入口与固定合法版本 |
| scripts/evaluate_benchmark.py | ALFWorld 也能从接受状态最终评价；budget seed 一题一次 |

真实数据分布、GPU 更新和官方 WebShop 完整运行仍未执行；CPU 合成测试只证明上述调度/接线，不构成实验性能证据。
