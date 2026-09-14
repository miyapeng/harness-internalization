# Harness Internalization

版本化 Harness 代码演化与选择性内化：先用可执行代码改进当前 Agent，再尝试把其中的控制计算蒸馏进模型，通过独立四格评价决定保留还是撤除。

工程是独立实现，不需要 OPID 或 Meta-Harness 的嵌套运行仓库。训练优化器通过可选外部依赖 `verl==0.5.0` 使用。原 planning/review/recovery 模块模式仍兼容。

**验证状态：182 项测试通过，包含完整环境事件回报、无学生决策 batch、具名控制及既有数据接线、训练与生命周期回归。真实 API proposer、HF 模型、外部 veRL、GPU 和官方 benchmark 实验均未运行。** CPU/mock 结果用于验证工程流程，不能作为模型内化效果或论文性能证据。

## 方法

```text
当前 checkpoint + Harness revision
    → 两个代码候选 → 实际 search/dev 运行 → 独立 Harness 接受门槛
    → 可选 InternalizationTarget → 训练前 contribution gate
    → H− on-policy rollout + 同一 old policy 的 H+ 评分
    → 模型更新 → 四格评价 → accept/retire、accept/retain 或 rollback/retain
```

`HarnessCandidate` 表示完整代码改进；`InternalizationTarget` 表示本次准备旁路的行为。H− 不必等于父版本，例如撤除诊断控制后仍可保留新增的日志工具。有效改进没有目标或无法接入当前监督桥时，保留 H+、模型不变、不训练。

schema 2 使用具名控制 ID；例如只关闭 review_v1，保留 recovery_v1 和工具。独立控制读取相同基础上下文，教师复用学生实际生成的非目标内容，按部署顺序插入目标结果。顺序依赖组合仍能运行，但暂不支持内化；旧单 hook 格式保持兼容。见 [具名控制协议与示例](docs/NAMED_CONTROLS.md)。

模块信号仍是同一 batch 的 old policy 对同一 response IDs 的概率差：`logp_old(response | enhanced_context) - old_log_prob(response | student_context)`，与原任务优势相加。模型回滚与行为退役分别判定，统计门槛没有为了演示而放宽。

环境事件与学生决策分开记录：prepare 中完成任务也计入完整 return，没有学生生成 token 就没有对应 actor loss。公开工具/模型调用参数与结果进入允许的 search 提案轨迹。见 [环境事件与回报](docs/ENVIRONMENT_EVENTS.md)。

任意候选代码必须在独立 Linux Landlock/seccomp 进程中运行；缺少隔离会报错。当前内化桥支持显式内部计算 hook 的旁路，不能把任意代码修改强行转换成文本指导。详细边界见 [版本化 Harness 说明](docs/VERSIONED_HARNESS.md)。

## 工程结构

```text
src/internalization/
  core/          数据结构、真实任务manifest、四种独立backend接口
  harness/       代码revision、隔离runtime、工具broker与兼容模块
  evolution/     proposer、候选lineage、archive、配对搜索选择
  training/      on-policy rollout、同批次策略H+评分、优势、更新、checkpoint
  evaluation/    训练前归因、模型接受/回滚、四格退役、bootstrap与成本
  benchmarks/    ALFWorld、AppWorld、TB2、SWE-bench Pro、HotpotQA、LawBench适配
  outer_loop.py  只通过本项目接口编排三个周期
configs/         保留的实验协议与进程配置
scripts/         提案、训练、评价、manifest与迁移验证入口
tests/           训练、版本化演化、数据/状态/最终评价接线回归
docs/            方法、运行、迁移、来源与验证记录
licenses/        保留的第三方许可证和NOTICE
```

## 快速开始

推荐 Python 3.12。纯兼容模式 demo 无额外依赖；完整测试和版本化 demo 需要安装 torch。代码执行要求 Linux Landlock ABI ≥ 3 和系统 `libseccomp.so.2`，本机验证环境为 ABI 4。

```bash
git clone https://github.com/miyapeng/harness-internalization.git
cd harness-internalization
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3.12 -m pip install -e .
# 若当前环境未安装 torch：
python3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3.12 -m unittest discover -s tests -v
python3.12 scripts/demo_versioned_harness.py \
  --config configs/versioned_demo.json --output runs/my-versioned-demo
python3.12 scripts/demo_named_controls.py --output runs/my-named-controls-demo
```

输出目录必须不存在。版本化 demo 实际执行代码、工具和隔离；模型行为与 trainer checkpoint 转换是 scripted mock，实际 optimizer updates 为 0。另有 CPU 小模型 SGD 测试验证训练桥。运行产物、模型权重、凭据及官方任务数据不随仓库发布；执行上述命令会在 `runs/` 生成完整候选、精简版本、目标和评价记录。

原兼容模式仍可运行：

```bash
python3.12 -m internalization.cli validate-module harness_modules/recovery.py
python3.12 -m internalization.cli demo --output runs/my-legacy-demo
```

真实训练和环境依赖通过 `.[teacher]`、`.[training]`、`.[alfworld]` 安装；真实运行前必须准备模型、授权资源及独立任务 manifest。新版导入器按 `--cycles` 生成独立 acceptance/retirement；三周期通用导入默认需至少270个非测试任务。演化恢复使用 `--state`，最终评价必须显式选择 `--state` 或 `--baseline`，不能复用 retirement/test。

## 文档与实验边界

- [方法](docs/METHOD.md)、[当前状态](docs/STATUS.md)、[工作记录](docs/WORKLOG.md)
- [版本化接口、配置和运行命令](docs/VERSIONED_HARNESS.md)
- [数据准备到最终评价的运行说明](docs/ACCEPTED_AGENT_PIPELINE.md)
- [144 项测试与接线验收记录](docs/validation/agent-pipeline-report.json)
- [154 项测试与 proposer 接口简化验收记录](docs/validation/proposer-binding-report.json)
- [171 项测试与具名控制验收记录](docs/validation/named-controls-report.json)
- [182 项测试与环境事件回报验收记录](docs/validation/environment-events-report.json)

完整迁移表和验证边界见 [docs/MIGRATION.md](docs/MIGRATION.md)，操作见 [docs/RUNBOOK.md](docs/RUNBOOK.md)，版权见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。源commit记录在 `upstream.lock.json`，仅作来源记录，不用于加载代码。历史flat模块和旧脚本名只是本项目代码的兼容入口。

AppWorld已增加独立环境worker、官方评分接口、manifest和训练/评价入口，真实包与数据尚未运行。见 [AppWorld运行说明](docs/benchmarks/APPWORLD.md)；后续接入顺序见 [Benchmark排期](docs/BENCHMARK_ROADMAP.md)。

TB2、SWE-bench Pro、HotpotQA和LawBench也已增加自有适配、配置、数据导入与评价入口，见 [四项适配运行说明](docs/benchmarks/ADAPTERS.md)。HotpotQA当前为distractor交互包装；LawBench保留原生单次响应与各题型官方评分。真实官方环境尚未验收，不能用CPU/mock测试结果代替benchmark成绩。

## 许可证

项目自有代码按 [MIT License](LICENSE) 发布。保留或修改的第三方文件仍适用其原许可证，包括 Apache-2.0；来源、版权声明和完整许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 `licenses/`。MIT 声明不替代这些第三方条款。
