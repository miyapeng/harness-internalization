# TB2 / SWE-bench Pro / HotpotQA / LawBench

2026-09-13。四项自有adapter已接入现有训练与评价入口；新增29项CPU/协议mock测试通过。实际Harbor、Docker、Pro测试、LawBench评分依赖、官方数据和GPU尚未运行。**代码接入不等于真实benchmark验收通过。**

| Benchmark | 环境与动作 | 评分 | 验证边界 |
| --- | --- | --- | --- |
| Terminal-Bench 2 | Harbor 0.23.0 `Trial.create/run`管理容器；自有`BaseAgent`转发student JSON exec/final；保留官方agent user、task.toml时间/资源约束 | agent返回后才执行官方verifier，读取`verifier_result.rewards["reward"]` | 队列/lifecycle/异常mock通过；真实Harbor、Docker、TB2任务未运行 |
| SWE-bench Pro | 官方实例镜像，在`/app`的base commit工作；JSON exec/final；提交实际working tree diff | 固定外部checkout的`swe_bench_pro_eval.py --use_local_docker`；读取resolved和实际test artifact | 镜像/命令/补丁/CLI mock通过；官方镜像与测试未运行 |
| HotpotQA | **distractor context agent wrapper**；固定题目paragraph集合，search/lookup/final | 自写官方数学定义：answer、supporting facts、joint EM/F1；terminal reward固定joint F1 | 实际自有worker子进程、合成检索与指标例子、两批CPU更新通过；官方数据和官方脚本逐样本parity未运行 |
| LawBench | 原生zero-shot单次文本响应，20题型；没有额外检索工具 | 调用外部官方`evaluation/main.py`；单例训练分数与完整题型官方聚合分别记录 | 20题型dispatcher、评分尺度、聚合mock通过；官方规则/中文依赖/ChERRANT未运行 |

## 代码位置与算法边界

| 功能 | 自有实现 |
| --- | --- |
| 环境配置、catalog哈希、split guard、外部grader commit检查 | `src/internalization/benchmarks/common.py`: `BenchmarkConfig`, `Catalog`, `checked_repo` |
| 环境进程隔离、超时、关闭自己启动的worker | `benchmarks/isolated.py`: `WorkerProcess`, `IsolatedEnvironment` |
| native worker分派 | `benchmarks/worker.py`: `native_environment`, `run` |
| TB2官方生命周期、自有Harbor agent | `benchmarks/terminalbench.py`: `HarborSession`, `TerminalBench2Environment`；`harbor_agent.py`: `InternalizationAgent`, `serve` |
| Pro工作区、patch、官方grader | `benchmarks/swebench.py`: `DockerWorkspace`, `OfficialProScorer`, `SWEBenchProEnvironment` |
| HotpotQA检索/终答/指标 | `benchmarks/hotpotqa.py`: `HotpotQAEnvironment`, `metrics` |
| LawBench官方dispatcher | `benchmarks/lawbench.py`: `LawBenchEnvironment`, `OfficialLawBenchScorer` |
| 官方题型聚合 | `benchmarks/aggregate.py`: `aggregate`；`lawbench_scoring.py` |
| 原始数据→实际任务ID/catalog/manifest | `benchmarks/importers.py`；`scripts/build_benchmark_manifest.py` |
| 训练/评价命令 | `training/entrypoint.py --benchmark {terminalbench2,swebench_pro,hotpotqa,lawbench}` |
| 独立最终评价 | `scripts/evaluate_benchmark.py` |
| 无模型engine/verifier smoke | `scripts/check_benchmark.py` |

`outer_loop.py`、优势公式、归因门槛、retirement判据和accept/rollback没有修改。新的worker只接受环境动作，教师评分不接触环境worker，因此H+评分不会偷偷执行真实工具。学生和teacher仍使用学生当前可见历史；不截断后再给teacher提供完整存档。

环境保留公开任务、动作和工具输出，target module指导只由原有Harness runtime管理。训练仍由`ModuleTrainer`在H−生成动作，使用同一batch behavior policy在H+对相同token重评分，再进入原有更新路径。教师指导和工具输出不是student response tokens。新环境没有新增hindsight skills。

## 准备依赖和数据

主模型环境仍按原RUNBOOK安装。四个`configs/*_backend.json`中的`python`是主模型环境解释器；`HI_BENCHMARK_PYTHON`是独立环境解释器，可以为每项使用不同venv。worker路径保留venv符号链接，不解析成系统Python。

共同变量：

```bash
export PYTHONPATH="$PWD/src"
export HI_BENCHMARK_PYTHON=/absolute/path/to/benchmark-venv/bin/python
export HI_BENCHMARK_CATALOG=/absolute/path/to/imported/catalog.json
```

未设变量、错误benchmark、无效context预算、未知task ID、stage中catalog哈希变化均报错。默认context=32768、student prompt=28672、action=2048、generation=1024；交互环境最多30步，LawBench固定1步。长历史超过模型预算时明确报错，不能把截断后的信息差归为module effect。

### Terminal-Bench 2

独立Python>=3.12，依赖`configs/terminalbench2-worker-requirements.txt`固定Harbor 0.23.0，还需要Docker Engine/Compose和相应镜像。数据是从固定TB2发布版本取得的本地Harbor task bundles；每项必须有`task.toml`、`instruction.md`、`tests/test.sh`，整个task目录记录hash并在reset复验。

```bash
"$HI_BENCHMARK_PYTHON" -m pip install -r configs/terminalbench2-worker-requirements.txt
python scripts/build_benchmark_manifest.py --benchmark terminalbench2 \
  --source /absolute/path/to/terminal-bench-2-tasks \
  --revision terminal-bench-2.0 --output data/tb2-import
```

不将task目录挂到模型或agent工作区；由Harbor负责构建、工具执行和terminal verifier。源码/隐藏tests只有host grader能访问。没有把solution脚本运行成“agent”。底层任务环境没有通用seed参数，记录的seed用于模型replica，不能声称多个seed就是多个独立任务。首次安装后的API兼容性必须用smoke实测。

### SWE-bench Pro

使用[官方Pro开源评测仓库](https://github.com/scaleapi/SWE-bench_Pro-os)，**不是SWE-bench Verified**。外部checkout固定完整commit；运行时检查HEAD和tracked working tree。依赖按其`requirements.txt`安装到独立worker环境，并准备官方`run_scripts`、`dockerfiles`与实例镜像。

```bash
export SWE_PRO_EVALUATOR_ROOT=/absolute/path/to/SWE-bench_Pro-os
export SWE_PRO_EVALUATOR_COMMIT=<exact-40-character-commit>
export SWE_PRO_DOCKERHUB_USERNAME=<official-image-publisher>
"$HI_BENCHMARK_PYTHON" -m pip install -r "$SWE_PRO_EVALUATOR_ROOT/requirements.txt"
python scripts/build_benchmark_manifest.py --benchmark swebench_pro \
  --source /absolute/path/to/public-instances.jsonl \
  --image-lock /absolute/path/to/pro-images.json \
  --revision <pinned-dataset-revision> --output data/pro-import
```

`pro-images.json`将每个`instance_id`映射为`{"tag":"official/repository:instance-tag","digest":"official/repository@sha256:..."}`。tag必须匹配官方`helper_code.image_uri`生成的实例映射，digest来自实际准备的镜像，不提供伪造示例digest。agent使用digest；官方grader调用前后检查tag是否仍指向同一镜像，变化则拒绝整个结果。官方grader本身可能pull其tag，拉取过程中版本变化也不能默认为有效比较。

任务导入保留官方字段，列表列按官方CSV/JSONL evaluator要求序列化。公开prompt只有problem statement、requirements、interface；gold patch、测试名单、安装/评分脚本不进入prompt。工作区无host mount，默认无网络。显式`final`或步数上限后捕获已跟踪和未忽略新文件的diff；官方脚本对binary patch的处理沿用其实现。grader缺少实际test artifact时报错，不能仅信CLI写出的False。

### HotpotQA

本轮接入[官方HotpotQA](https://hotpotqa.github.io/)的**有标签distractor**数据。`--source`放官方dev distractor，保留为最终test；可选`--train-source`放官方train。固定context中的词项重叠search（配置`top_k=3`）和按title查全文的lookup是本项目的交互包装，**不能称作fullwiki Search-R1检索配置或与fullwiki排行榜同协议**。

```bash
python scripts/build_benchmark_manifest.py --benchmark hotpotqa \
  --source /absolute/path/to/hotpot_dev_distractor_v1.json \
  --train-source /absolute/path/to/hotpot_train_v1.1.json \
  --revision hotpot-train-v1.1-dev-distractor-v1 --output data/hotpot-import
```

数据与语料文件SHA256入catalog。初始观察给question和context titles，正文由search/lookup返回；训练label仅在terminal计算指标。保留官方answer normalization、yes/no/noanswer特殊处理、supporting-fact集合去重和joint precision/recall组合；joint F1不是answer F1×support F1。没有标签的官方test拒绝本地评分，fullwiki与官方提交服务器流程不在本轮支持范围。

### LawBench

使用[open-compass/LawBench](https://github.com/open-compass/LawBench)外部固定checkout。worker需要该版本评分依赖，包括pandas、cn2an、jieba、rouge-chinese、nltk以及2-1题型所需ChERRANT脚本/资源；具体依赖随锁定checkout验收。本轮没有完成这些依赖安装。

```bash
export LAWBENCH_EVALUATOR_ROOT=/absolute/path/to/LawBench
export LAWBENCH_EVALUATOR_COMMIT=<exact-40-character-commit>
python scripts/build_benchmark_manifest.py --benchmark lawbench \
  --source "$LAWBENCH_EVALUATOR_ROOT/data" \
  --revision "$LAWBENCH_EVALUATOR_COMMIT" --output data/lawbench-import
```

原始数据没有统一实例ID，因此ID为category、原始文件hash前缀、原始row index，完整hash另存。20题型全部分派给官方脚本，不统一替换成EM。官方脚本返回fraction，不再除以100。训练/四格的episode score是官方单样本分数；最终报告另按seed聚合同题型的全部预测，重新运行官方脚本，不能将singleton均值冒充官方整体指标。缺少题型时`official_20_category_macro=null`，不足500条/题型时`full_dataset_verified=false`。

官方3-4/3-5会跳过部分死刑/无期样本，单例评分可能无定义。完整最终评价使用专用 `scripts/evaluate_lawbench.py` / `benchmarks/lawbench_final.py`：先生成所有预测，再交给官方函数按题型评分，保留其跳过样本的规则，不编造逐样本reward。逐episode训练/四格入口遇到无定义或越界分数仍明确报错；这些样本不能未经新协议就进入现有四格统计。2-1使用官方固定临时文件，本项目以checkout级文件锁串行化调用，子脚本使用同一worker环境的python3。

```bash
python scripts/evaluate_lawbench.py --manifest data/lawbench-import/manifest.json \
  --config configs/lawbench.json --baseline --checkpoint /absolute/model \
  --output runs/lawbench-final
```

LawBench是非交互机制实验，不能当成多步Agent能力证据；本轮未引入法律检索或参考答案特权。

## 划分、运行与记录

TB2、Pro、LawBench导入默认只有`test`，不能直接启动outer loop。只有显式提供**独立**`--train-source`，新版默认三周期至少270个任务，建立train/search/dev、三个retirement与三个acceptance cohort；每个cohort至少30任务。`--cycles`预先调整周期数，`--legacy-modules`显式生成旧模式（三周期180任务）。不能用公开测试数据再次导入冒充train：跨源ID与题面重复会拒绝。独立训练源的合法来源仍需研究者核实，本工具不证明外部数据没有语义近重复。

search、dev、acceptance和retirement来自训练源的互斥保留子集；官方测试源完全隔离。主入口对catalog的test行默认拒绝，独立最终评价脚本才增加`--final-evaluation`，train/propose不能使用此flag。bootstrap仍以task ID为cluster，多个seed不是额外任务。

```bash
# 先单任务engine/verifier验收；无模型smoke不是benchmark成绩。
python scripts/check_benchmark.py --config configs/terminalbench2.json \
  --task-id <imported-task-id> --output runs/tb2-engine-smoke

# 显式初始基线；评价演化结果时改用 --state .../deployment.json。
python scripts/evaluate_benchmark.py --manifest data/tb2-import/manifest.json \
  --backend configs/terminalbench2_backend.json --baseline --checkpoint /absolute/model \
  --partition test --output runs/tb2-final

# 版本化模式要求独立 train/search/dev/acceptance/retirement 分区。
python -m internalization.cli run --manifest data/hotpot-import/manifest.json \
  --backend configs/hotpotqa_backend.json --checkpoint /absolute/model \
  --harness-workspace examples/versioned_harness/base \
  --cycles 3 --output runs/hotpot-cycle --train-steps 300
```

以上命令中的依赖/数据路径必须替换为实际准备路径；本轮未运行真实任务命令。`environment/identity.json`保存task/seed/catalog与镜像或task bundle hash，`grade.json`保存terminal指标，worker/grader日志留在episode目录。训练和模型成本沿用原ledger。tool_calls计学生search/lookup/exec（Pro还计patch capture）；不把内部测试用例数当agent工具调用。总wall latency含环境启动、官方评分和关闭；最终LawBench重新聚合的额外时间进入stage latency。镜像准备/下载、环境安装和人工准备费用尚未纳入在线token/tool账目，论文总预算需另外记录。

容器运行结束停止本次环境并保留诊断产物，不执行全局docker prune。长任务的容器资源、镜像体积、真实延迟、Harbor API兼容性均需首次真实smoke验证。

## 验证记录与来源

新增回归见`tests/test_benchmark_adapters.py`，全量日志见`docs/validation/benchmark-adapters-tests.txt`，机器报告见`docs/validation/benchmark-adapters-report.json`。旧P0和AppWorld测试继续保留，CPU三周期demo与上一版90个文件逐字节一致。原20项测试文件不变。

本轮直接检查了[Harbor Trial](https://github.com/laude-institute/harbor/blob/main/src/harbor/trial/trial.py)、[BaseAgent](https://github.com/laude-institute/harbor/blob/main/src/harbor/agents/base.py)、[Harbor 0.23.0发布](https://pypi.org/project/harbor/0.23.0/)、[Pro evaluator](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/swe_bench_pro_eval.py)、[HotpotQA evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)、[LawBench dispatcher](https://github.com/open-compass/LawBench/blob/main/evaluation/main.py)与相应评分函数。没有复制这些上游源码到src；来源及尚未解析的commit见THIRD_PARTY_NOTICES。来源可读不代表这些依赖已在本机运行。

统一状态加载与恢复见 [ACCEPTED_AGENT_PIPELINE.md](../ACCEPTED_AGENT_PIPELINE.md)。LawBench 专用整类原生评价暂不支持代码 revision，会在模型加载前明确拒绝；不会回退为空 Harness。
