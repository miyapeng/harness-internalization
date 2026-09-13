# 运行说明

从项目根目录操作。核心CPU demo不需要模型、上游仓库或网络。以下真实训练命令是准备入口，尚未完成GPU端到端验证。

## CPU检查

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3.12 -m unittest discover -s tests -v
python3.12 -m internalization.cli demo --output runs/new-demo
python3.12 scripts/verify_migration.py --before runs/migration-before --after runs/new-demo --allow-model-acceptance-fields
```

测试缺torch/PyYAML时明确skip；当前机器上71项全部执行。每次使用新output目录，不覆盖旧实验。`runs/migration-before`是本地保存的迁移前基准，本轮新增接受字段的对照必须有此基准原文件；只有checksum不能做新增字段的语义比较。不带该选项仍保持严格字节对照。当前77文件字节相同，8文件仅新增结果字段；脚本仍拒绝原分数、轨迹或决策变化。

## 可选依赖

在自己的独立Python环境安装：

```bash
python -m pip install -e '.[training,alfworld]'
```

`training`依赖普通外部 `verl==0.5.0`、torch、transformers与OmegaConf；`teacher`只装HF教师所需依赖。没有相对目录PYTHONPATH、补丁安装或嵌套仓库步骤。依赖安装、模型下载和真实训练本次均未运行。

默认adapter使用单进程标准veRL PPO actor与HF模型，独立KL reference默认CPU，可设置 `HI_TEACHER_DEVICE`，学生设置 `HI_DEVICE`。该入口没有旧Ray/FSDP/vLLM集群启动功能；扩展分布式时注入标准veRL TrainingBackend，不能把本入口当成原八卡启动脚本的等价替代。

## ALFWorld准备与外层入口（未运行）

使用外部ALFWorld发布的数据，设置 `ALFWORLD_DATA` 指向包含logic与json_2.1.1的目录。`configs/alfworld.yaml` 保留原配置。记录环境包版本、数据revision、tokenizer与模型权重checksum。

```bash
python scripts/build_alfworld_manifest.py \
  --data-root "$ALFWORLD_DATA/json_2.1.1" \
  --revision YOUR_DATA_REVISION --output configs/local-alfworld-manifest.json
```

manifest从train划出独立search/dev和三份retirement cohort；官方valid_seen/unseen仅作最终评价，外层不会访问。真实task ID为game.tw-pddl路径，不使用parquet占位索引。

为独立API proposer在环境变量设置 `HI_PROPOSER_MODEL`、`HI_PROPOSER_BASE_URL`（兼容chat/completions的API前缀）、`HI_PROPOSER_API_KEY`。密钥不写入配置或日志。然后：

```bash
python -m internalization.cli run \
  --manifest configs/local-alfworld-manifest.json \
  --backend configs/alfworld_backend.json \
  --checkpoint /absolute/local/hf-checkpoint \
  --attribution-policy configs/attribution.json \
  --train-steps 300 --output runs/new-alfworld-run
```

配置中三个子进程使用当前环境的 `python`，应从根目录运行。直接阶段入口：

```bash
python scripts/propose.py --request /path/request.json --response /path/new-response.json
python scripts/train.py --request /path/request.json --response /path/new-response.json
python scripts/evaluate_alfworld.py --request /path/request.json --response /path/new-response.json
```

不要手写私有OPID Hydra参数。请求由 `CommandBackend` 根据 `core/interfaces.py` 生成；提案包含typed trajectories/search scores/history，训练包含student/teacher checkpoints、H+/H−、train allowlist、target和预算。旧teacher_checkpoint字段现在仅指定阶段冻结KL reference；H+ scorer直接复用本批student策略，在update前完成评分。trainer自行采集新on-policy轨迹；它不会把proposer的旧轨迹当训练数据。

## 训练前归因门槛

`--attribution-policy`读取固定JSON，省略时使用默认值：至少30个独立task，配对A−B的95% bootstrap区间下界严格大于0；2000次重采样，seed=42。可预先提高`min_external_gain`，不能根据本次评价结果调整配置。准入策略在任何rollout前保存为`attribution_policy.json`，判定见每轮`attribution.json`。

拒绝候选归档为`no_external_contribution`，不调用trainer、不生成C/D或retirement结论；原checkpoint与已有模块不变，本轮训练预算保持未花费。下一轮继续；归因判定不提供给proposer。通过门槛只表示允许训练。

## 模型接受与回滚

模型接受使用outer loop的固定`RetirementPolicy`，与撤除共用ε和配对bootstrap设置：默认ε=0.02、95%区间、2000次重采样、30个独立task、seed42，在运行前写入`protocol.json.retirement`。当C−A区间上界严格小于−ε时，`model_decision=rollback`、`module_decision=retain`；任务不足也拒绝新模型并记录独立原因。没有显著退化时接受新模型，再由原能力/成本检查决定retire/retain。没有显著退化不等于统计上证明非劣。

结果见`retirement.json`：`model_acceptance`包含原因与区间；`capability_preserved`、`cost_improved`、`retirement_eligible`和`retirement_accepted`可分别检查。state/events/deployment的逐轮记录也包含模型与模块双字段，以及before/proposed/accepted checkpoint。旧`decision`仅是模块决策兼容别名。

回滚恢复原checkpoint供下一周期使用，并保留H+中本轮与已有模块；拒绝的checkpoint与训练记录留作排查，训练费用仍计入账本。自定义评估器必须返回双字段，只返回retire/retain会报错。

## 输出与边界

`protocol.json`、`events.jsonl`、`candidate_archive.jsonl`、每轮A/B/C/D、retirement与state、deployment记录外层决策。进程costs journal记录搜索/评价/提案/训练成本，训练子目录保存逐步公开状态、教师指导与评分、精确token IDs、更新metrics、checkpoint manifest。旧预算键optimizer_steps实际表示采样/更新批次数，见MIGRATION。

缺依赖、信息不对称、checkpoint变化、wrong task/seed、stale rollout或预算不足均报错，不静默退化为SFT。WebShop/Search-QA仅有任务身份约束。AppWorld和新增四项benchmark均有独立backend/config，不能用ALFWorld配置运行。

## AppWorld独立入口

安装固定AppWorld包到单独Python环境，设置HI_APPWORLD_PYTHON与APPWORLD_ROOT后，通过scripts/check_appworld.py核验真实engine/数据，再使用configs/appworld_backend.json运行。完整命令、task分区和费用口径见 [AppWorld说明](benchmarks/APPWORLD.md)。当前只完成代码与CPU/mock验收，真实包下载被网络代理阻断；没有官方task或GPU运行结果。

## TB2 / Pro / HotpotQA / LawBench

配置分别是`configs/terminalbench2_backend.json`、`swebench_pro_backend.json`、`hotpotqa_backend.json`、`lawbench_backend.json`。先通过`scripts/build_benchmark_manifest.py`导入实际数据并固定hash，再配置独立worker解释器与外部grader commit。用`scripts/check_benchmark.py`做无模型环境验收；最终评价使用`scripts/evaluate_benchmark.py`，完整LawBench原生最终评价使用`scripts/evaluate_lawbench.py`。详细命令、官方来源、训练数据隔离与验证限制见 [ADAPTERS.md](benchmarks/ADAPTERS.md)。
