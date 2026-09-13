# 运行说明

## 1. CPU 核心

```bash
cd /data/miyapeng/harness-internalization
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3.12 -m unittest discover -s tests -v
python3.12 -m internalization.cli demo --output runs/new-smoke
```

检查 `events.jsonl`、每轮 `A/B/C/D/episodes.json`、`retirement.json`、`state.json` 和最终 `deployment.json`。`DISCLAIMER.json` 标记合成运行。真实运行也使用相同记录结构，外部进程增加 request/response/stdout/stderr 与成本账本。

## 2. 上游版本与补丁

两个仓库已存在 `upstream/`，版本在 `upstream.lock.json`。OPID 的本地改动由 `scripts/install_opid_patch.py` 管理；它验证 commit 和完整待改文件，遇到额外修改会拒绝覆盖。

```bash
python3.12 scripts/install_opid_patch.py --check
# 新 clone 且匹配 lock 后应用：
python3.12 scripts/install_opid_patch.py
```

补丁涉及 trainer、rollout collector、ALFWorld/WebShop/Search 环境元数据，不覆盖多套 verl。未创建远端 fork，后续可按实际 GitHub 账号添加 remote。

## 3. 单独准备 OPID 环境

按固定 checkout 的 `upstream/OPID/README.md` 创建 Python 3.12 主环境；其中 vLLM/torch/flash-attn 版本须按硬件验证。WebShop 单独 Python 3.10 环境，检索服务也独立，不能把默认 Python 3.14 当成 OPID 环境。

本项目只需在 OPID 环境中增加：

```bash
python -m pip install -e /data/miyapeng/harness-internalization
```

需要可用 GPU、对应驱动、HF 模型目录、ALFWorld 数据；当前机器未具备完整运行条件。模型建议先 Qwen2.5-3B-Instruct，不自动用已有 Qwen3 模型替换。

## 4. 准备真实任务划分

下载并配置 `ALFWORLD_DATA` 后，使用实际 `game.tw-pddl` 路径划分：

```bash
python scripts/build_alfworld_manifest.py \
  --data-root /absolute/alfworld/json_2.1.1 \
  --revision actual-dataset-version-or-checksum \
  --output configs/alfworld.tasks.json
```

search/dev/retirement_0/1/2 各默认 60 个 train 内任务，剩余用于训练。真实正式规模应预先确定；脚本不生成虚假 task ID。官方 valid_seen/valid_unseen 全部映射为最终 test 分区。

按 OPID `examples/data_preprocess/prepare.py` 生成占位 parquet。它只控制 rollout 批规模，不能作为真实数据划分。ALFWorld patch 在环境初始化时把训练游戏限制到阶段 task allowlist；每个 rollout 再检查实际游戏 ID。

## 5. 单阶段训练入口

```bash
python -m internalization.cli prepare-phase \
  --checkpoint /absolute/models/Qwen2.5-3B-Instruct \
  --module harness_modules/recovery.py \
  --manifest configs/alfworld.tasks.json \
  --output runs/alf-first/opid_phase.json \
  --device cpu

export HI_PHASE_FILE=/absolute/path/to/opid_phase.json
export MODEL_PATH=/absolute/models/Qwen2.5-3B-Instruct
export TRAIN_PARQUET=/absolute/path/to/train.parquet
export VAL_PARQUET=/absolute/path/to/test.parquet
export OUTPUT_DIR=/absolute/path/to/new-checkpoints
export NUM_GPUS=1
export TRAIN_STEPS=100
bash scripts/launch_opid_alfworld.sh
```

`NUM_GPUS=1` 是可覆盖的启动值，不代表测得单卡可运行。教师 `cpu` 是节省学生显存的正确性后端，可能很慢；有独立 GPU 时在阶段配置中设置。教师与学生 vocab 必须一致。max_prompt_length=4096 且 truncation=error，越界会失败，不能让教师偷偷看更多历史。

默认关闭 OPID 自动 validation，防止官方测试任务参与中间选择。episode teacher 权重为 0，step teacher 权重为 0.001，任务奖励路径保留。所有 baseline 需相同解码、任务/种子和独立教师开销口径。

## 6. 完整外层的真实入口（尚未 GPU 验证）

在 OPID 环境配置好上述 parquet、GPU、模型和 ALFWorld 后，再提供 proposer 的 API 配置：`HI_PROPOSER_BASE_URL`（包含 `/v1`）、`HI_PROPOSER_MODEL`、`HI_PROPOSER_API_KEY`。密钥不写入仓库或日志。默认不做任何外部 API 调用，只有执行此入口才使用这些变量。

```bash
python -m internalization.cli run \
  --manifest configs/alfworld.tasks.json \
  --backend configs/alfworld_backend.json \
  --checkpoint /absolute/models/Qwen2.5-3B-Instruct \
  --output runs/alfworld-001 \
  --train-steps 300
```

流程为先采集 baseline_search 轨迹、调用 proposer、两个候选 search/dev 评价、冻结阶段、A/B 评价、OPID 更新及 HF 合并、C/D 评价、退役/保留。每个阶段运行独立命令。更换机器后修改 backend JSON 的 cwd，并保证 `python` 来自正确环境。环境服务有自己的依赖时单独替换对应命令的解释器。

模块未退役时，下一阶段 H− 的保留模块通过当前 actor 执行，只将最终主动作送到环境。辅助生成独立计费，不加入学生 response 训练目标。该路径要求同步 vLLM；已做 CPU 接口与保留后继续周期测试，真实 Ray/vLLM 行为仍待验证。每阶段的决定和 checkpoint 均保存在 `state.json`。

## 7. 复核已有四格结果

```bash
python -m internalization.cli retirement \
  --A runs/experiment/cycle_00/A/episodes.json \
  --B runs/experiment/cycle_00/B/episodes.json \
  --C runs/experiment/cycle_00/C/episodes.json \
  --D runs/experiment/cycle_00/D/episodes.json \
  --output runs/experiment/recheck.json --margin 0.02
```

缺任务、不同 seed、重复结果或 NaN 都会拒绝。置信区间不足则保留，不通过调整评测队列追求退役。

## 成本记录口径

- evaluator 每 episode 记录输入/输出 token、总/辅助模型调用、环境工具调用和实测 wall latency。
- teacher score 中 response token 算输入计算量，不算生成输出；模块生成和 draft 都计入辅助成本。
- OPID rollout 记录所有实际生成行，包含 done 后可能浪费的计算；工具数表示调用环境 step 接口的行数。
- stage wall time 包含反向传播、优化、环境与 checkpoint 工作；CPU demo 所有成本为合成单位。
- 外层另记录每个进程总 wall time（包含模型加载），不能把进程时长与 episode 时长相加两次。正式 GPU 小时账本还需记录资源分配；baseline 不能仅凭相同 optimizer steps 声称等预算。
