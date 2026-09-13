# AppWorld 接入与运行记录

状态：**自有adapter与命令入口已实现，CPU/mock验证通过；真实AppWorld engine/data与GPU训练NOT YET VERIFIED。** 本轮只接AppWorld，下一项为Terminal-Bench 2；SWE-bench Pro、HotpotQA、LawBench尚未开工，不把文件占位当作接入。

## 官方接口与来源

固定外部发布包`appworld==0.1.3.post1`，支持Python>=3.11。环境代码不放进本仓库，模型进程只通过JSON-lines与独立worker通信。发布版约束与训练依赖分开安装，尤其避免其pydantic等版本约束以及环境时间冻结影响训练进程。[PyPI发布元数据](https://pypi.org/project/appworld/0.1.3.post1/)。

接口依据版本化的 [AppWorld环境源码](https://github.com/StonyBrookNLP/appworld/blob/v0.1.3.post1/src/appworld/environment.py)、[TestTracker与官方指标实现](https://github.com/StonyBrookNLP/appworld/blob/v0.1.3.post1/src/appworld/evaluator.py)、[task ID加载器](https://github.com/StonyBrookNLP/appworld/blob/v0.1.3.post1/src/appworld/task.py)。官方规定由Agent生成API调用，不应把登录等调用流程写死在框架中；任务参考答案、难度和评测细节不进入Agent输入。[官方开发协议](https://github.com/StonyBrookNLP/appworld#no_entry_sign-agent-development-restrictions)。本项目只实现接口适配，没有复制官方任务、参考解或环境实现。

## 执行、监督与评分

每个task/seed（包括重复seed的训练replica）启动独立worker，创建独立experiment_name，避免world状态和输出覆盖。调用`AppWorld(task_id=..., random_seed=...)`初始化，提示只包含公开instruction、supervisor、datetime、app描述、api_docs/supervisor帮助文档。框架不会先执行登录或complete_task。

模型每步生成Python；支持原始代码或一个完整Python代码围栏，代码交给官方`world.execute`。原始action token不替换；仅执行接口移除围栏。执行输出与全部已提交action追加到公开历史，不截断后暗中给教师额外历史。发生context overflow直接报错。

`world.task_completed()`决定任务是否已声明结束；达到step上限也终止。只在终止时调用`world.evaluate()`，从完整执行的`TestTracker.success`取得0/1任务成功和terminal reward。complete_task成功不等于任务评分通过。测试数为零/不完整、worker崩溃、超时或依赖错误都报错，不转成普通失败轨迹。正常Python/API错误则作为REPL观察留给Agent修复。

worker为了官方评测加载`ground_truth_mode=minimal`，但只向主进程回传上述公开字段和数值reward/success；从不读取或转发参考解、答案、required APIs、difficulty或grader failures。评分结果不追加到下一步prompt。当前未采用AppWorld中间步骤的测试通过率作为reward。

训练仍走现有`ModuleTrainer`：学生H−真实执行；同batch behavior policy在H+对相同response IDs评分；保留outcome + λg×module advantage及P0-1/2/3门槛，不改优化器。默认AppWorld上下文32768、student prompt上限28672、action上限1024、模块生成上限256、30步、每步API上限1000、执行超时100秒、RPC超时180秒；固定在configs/appworld.json，并记录于每个训练/评价阶段的benchmark_protocol.json。ALFWorld的原有默认值不变。

## 成本与输出

`Cost.tool_calls`对AppWorld明确为“world.execute调用数 + 其间官方api_calls.jsonl记录的API请求数”，不是代码行数。前者包括失败的执行尝试；后者包含API文档检索等真实请求。environment.jsonl另列execute_calls与api_calls，便于分开比较。reset、task_completed检查和grader调用不是Agent工具动作，耗时包括在episode墙钟时间内。输入/输出token、模型/辅助调用沿用真实模型backend计数；没有用字符数估token。

逐episode的identity.json记录task/seed、包版本、DB版本、official输出目录、实验名与限制；原始AppWorld日志和最终DB继续由官方engine保存。训练/评价轨迹仍在本项目trajectories.jsonl，包含student prompt、精确response IDs、task/model/Harness身份、reward与cost。文件标识使用随机UUID以防覆盖，这些UUID不进入Agent prompt。

最终评价脚本独立于outer loop，输出TGC（平均严格任务成功率）和完整scenario triples上的SGC（同scenario三个变体都成功），多seed先各自计算再平均。指标单位为百分数；子集缺变体时SGC=null并明确标记，不能冒充完整官方SGC。test评价只输出aggregate到终端，任务轨迹留在本地，不回流proposer/训练。

## Manifest协议

build_appworld_manifest.py通过官方`load_task_ids`读取四个split，不靠伪造parquet输入或猜任务ID。训练/search只取官方train；dev与retirement仅用剩余train/dev。所有同scenario的三个变体整组划分，官方test_normal/test_challenge原样保留供最终评价。

默认3周期，search=15、dev至少15、每个retirement cohort=30；先用未选为dev的官方dev填充retirement，不足时从官方train预留，其余官方train用于参数训练。对90个train、57个dev的输入，这产生train=27/search=15/dev=15/retirement=30×3；这是已测试的划分形状，当前没有声称已在本机下载并验证真实任务数量。任务不足或变体不全则报错，不复用held-out cohort、不借test、不降低P0 min_tasks。

manifest保存数据revision、原官方ID列表/文件SHA256、split seed与完整fingerprint；worker每次训练reset还验证task属于官方train。P0配对bootstrap仍按原协议的task ID聚类，不把多个seed视为独立任务；没有改成scenario级bootstrap。三个变体存在相关性，不能把30 tasks解释为30独立scenarios；正式实验需要另外报告scenario层面的不确定性敏感性分析，这项尚未实现。

## 安装与无模型smoke（本机未完成）

以下是准备好的真实入口，当前环境没有appworld包/数据，PyPI下载连接经代理返回403，且本机无Docker。没有实际安装或跑官方task。

在独立环境中安装worker，不要把AppWorld依赖装进veRL环境：

```bash
python3.12 -m venv /absolute/new-appworld-env
/absolute/new-appworld-env/bin/python -m pip install -r configs/appworld-worker-requirements.txt
/absolute/new-appworld-env/bin/appworld install
export HI_APPWORLD_PYTHON=/absolute/new-appworld-env/bin/python
export APPWORLD_ROOT=/absolute/appworld-data-root
mkdir -p "$APPWORLD_ROOT"
/absolute/new-appworld-env/bin/appworld download data
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/check_appworld.py --output runs/new-appworld-engine-smoke
```

smoke仅在第一条官方train任务执行print并调用官方grader，成功不代表Agent完成任务。保留官方发布版的raise_on_unsafe_syntax和null_patch_unsafe_execution安全选项，不启用login shortcut或修改时间等实验特权。当前使用独立进程隔离依赖；这不等于容器级文件系统隔离，真实模型执行应在专用工作环境运行。

## 接入演化与训练

从训练环境运行，worker仍由HI_APPWORLD_PYTHON独立启动：

```bash
python scripts/build_appworld_manifest.py \
  --revision YOUR_DOWNLOADED_DATA_REVISION \
  --output configs/local-appworld-manifest.json
python -m internalization.cli run \
  --manifest configs/local-appworld-manifest.json \
  --backend configs/appworld_backend.json \
  --attribution-policy configs/attribution.json \
  --checkpoint /absolute/local/hf-checkpoint \
  --train-steps 300 --output runs/new-appworld-run
```

proposer使用原HI_PROPOSER_MODEL/HI_PROPOSER_BASE_URL/HI_PROPOSER_API_KEY。模型使用本地HF checkpoint；真实veRL/GPU仍未验证。搜索、A/B/C/D、训练都经同一AppWorld配置，outer_loop不导入AppWorld类型。

最终评价必须显式传入正式接受的checkpoint；有residual模块时同时传最后周期state：

```bash
python scripts/evaluate_appworld.py \
  --manifest configs/local-appworld-manifest.json \
  --checkpoint /absolute/accepted-checkpoint \
  --state runs/new-appworld-run/cycle_02/state.json \
  --partition test_normal --output runs/new-appworld-test-normal
```

另跑test_challenge时使用新的输出目录。脚本检查state的checkpoint与传入模型一致，载入其中active_modules及其Harness hash。Hcore-only基线可省略state。最终test不参与候选选择或P0判定。

## 实现定位与验收

| 功能 | 状态 | 代码与证据 |
| --- | --- | --- |
| 独立环境、reset/step/close、完整公开历史 | IMPLEMENTED | [AppWorldEnvironment](/data/miyapeng/harness-internalization/src/internalization/benchmarks/appworld.py:117) |
| 官方execute、terminal grader、版本/任务校验 | IMPLEMENTED | [NativeAppWorld](/data/miyapeng/harness-internalization/src/internalization/benchmarks/appworld_worker.py:23) |
| 官方split、scenario分组、test隔离 | IMPLEMENTED | [build_manifest](/data/miyapeng/harness-internalization/src/internalization/benchmarks/appworld_manifest.py:22) |
| 评价与训练进程分派 | IMPLEMENTED | [entrypoint](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:46) |
| 模型接受/退役/evolution接口复用 | IMPLEMENTED | outer_loop.py未修改；现有P0回归继续通过 |
| 实际子进程通信＋两批CPU mock参数更新 | IMPLEMENTED / MOCK VERIFIED | [集成测试](/data/miyapeng/harness-internalization/tests/test_appworld.py:178) |
| 真实AppWorld、数据、API proposer、HF/veRL/GPU实验 | NOT YET VERIFIED | 未安装/未运行；不能用stub成功率作为benchmark结果 |

71项测试通过，其中AppWorld新增13项；原20测试文件未改。P0-3之后的三周期demo全部90文件逐字节一致。完整日志和机器记录见 [appworld-tests.txt](../validation/appworld-tests.txt)、[appworld-report.json](../validation/appworld-report.json)。测试使用项目自写test double，不包含官方数据或参考解；持久化mock目录为runs/appworld-proof。
