# 算法实现一致性审计：批次策略、归因与模型接受

2026-09-13更新。**P1-1先接AppWorld：环境/manifest/评价/训练接口已实现并通过CPU/mock测试，真实环境NOT YET VERIFIED。** **P0-3已实现：模型accept/rollback与模块retire/retain分别判定；明显退化时回滚旧checkpoint并保留模块。** **P0-2已实现：A/B配对归因通过预先固定门槛才训练，否则discard并保留原模型与residual Harness。** **P0-1正式定义为“H+ scorer与当前batch behavior-policy snapshot的同步”，当前已在默认单进程训练路径实现并通过CPU回归。** 不再要求增加阶段冻结的teacher_minus评分，`teacher_log_prob - old_log_prob`公式保留。

先前按“整个阶段冻结teacher、双路冻结评分”目标生成的完整只读审计保存在 [history/IMPLEMENTATION_AUDIT-stage-frozen.md](history/IMPLEMENTATION_AUDIT-stage-frozen.md)。其中的源码行号、反例、缺口结论对应修改前实现和当时目标；本页以用户最新确定的batch-aligned self-distillation与训练前Attribution Gate为准。P0-2修改前审计另存 [history/IMPLEMENTATION_AUDIT-before-attribution.md](history/IMPLEMENTATION_AUDIT-before-attribution.md)。

P0-3修改前审计保存在 [history/IMPLEMENTATION_AUDIT-before-acceptance.md](history/IMPLEMENTATION_AUDIT-before-acceptance.md)，旧文中“无条件接受新模型”的结论仅对应修改前代码。

## 最终训练定义与真实执行顺序

对第u个训练batch：

\[
\theta_{old,u}=\text{生成该batch rollout的策略},
\qquad
\ell^-_{t,j}=\log\pi_{\theta_{old,u}}(a_{t,j}\mid a_{t,<j},s_t,H^-).
\]

H− rollout保存原始response IDs与`old_log_probs=ell_minus`。同一次参数更新前，用**同一个policy对象、同一组权重**运行H+模块，并只对这些既有response IDs评分：

\[
\ell^+_{t,j}=\log\pi_{\theta_{old,u}}(a_{t,j}\mid a_{t,<j},s_t,H^+),
\qquad
A^{module}_{t,j}=\operatorname{stopgrad}(\ell^+_{t,j}-\ell^-_{t,j}).
\]

\[
A_{t,j}=A^{outcome}_{t,j}+0.001\,g_tM_{t,j}A^{module}_{t,j}.
\]

构造完整batch后才update，下一batch重新绑定更新后的策略。H+可以按模块定义生成内部规划/审核/恢复指导（review含未执行草案），但不重新生成用于监督的最终action，也不执行teacher环境动作。监督token始终来自H− rollout。

```text
batch u:
  with BehaviorPolicySnapshot(student):   # 同一对象；eval + inference_mode
      H− rollout → action IDs、old_log_probs
      H+ module execution → 对同一action IDs评分
      task advantage + lambda * gate * (teacher_log_prob − old_log_prob)
  student.update(batch)                  # 只在只读上下文结束后执行
batch u+1:
  绑定更新后的student，再执行相同流程
```

生产调用链定位：[training/trainer.py:85](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:85) `ModuleTrainer.train` → [training/behavior_policy.py:10](/data/miyapeng/harness-internalization/src/internalization/training/behavior_policy.py:10) `BehaviorPolicySnapshot` → [training/rollout.py:50](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:50) `InteractionTaskRunner.rollout` → [training/teacher_scoring.py:30](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:30) `ModuleTeacherScorer.score` → [training/trainer.py:19](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:19) `build_update_batch` → [training/trainer.py:105](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:105) `student.update`。

这里的snapshot是串行训练中的只读策略视图，并未复制完整权重。作用域内不能进行优化器更新；常规torch参数/缓冲区in-place版本变化、模型替换、snapshot ID变化、train-mode变化均拒绝。它不是用于任意并发写入的互斥锁，也不是权重内容checksum；分布式backend必须提供自己的同步/冻结保障。

## 逐项实现状态

| Stage | Requirement | Status | Code location | Evidence | Gap / Action |
| ----- | ----------- | ------ | ------------- | -------- | ------------ |
| 1 Evolve | 同一当前checkpoint、residual Harness、多候选、search/dev评分与无收益跳训 | IMPLEMENTED | [outer_loop.py:50](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:50)；[evolution/search.py:17](/data/miyapeng/harness-internalization/src/internalization/evolution/search.py:17) | 搜索评分不变；返回含parent/hash的Candidate供归因归档；原20测试与外层回归通过 | 真实API/benchmark搜索仍NOT YET VERIFIED |
| 2 Attribute | 同checkpoint、同任务/seed的A/B，H+/H−只差target | IMPLEMENTED | [outer_loop.py:61](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:61)；[tests/test_attribution.py:127](/data/miyapeng/harness-internalization/tests/test_attribution.py:127) | 同checkpoint、任务和seed调用；只增加本轮module；测试验证两侧参数 | 真实benchmark评价NOT YET VERIFIED |
| 2 Attribute | A−B不达预先固定统计门槛则discard，禁止internalization | IMPLEMENTED | [evaluation/attribution.py:34](/data/miyapeng/harness-internalization/src/internalization/evaluation/attribution.py:34)；[outer_loop.py:69](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:69)；[tests/test_attribution.py:105](/data/miyapeng/harness-internalization/tests/test_attribution.py:105) | task-cluster收益区间下界严格超过min_external_gain；A=B时trainer零调用、不跑C/D、不接受新module | P0-2 CPU回归通过；通过表示允许训练，尚不表示内化成功 |
| 3 Rollout | 当前batch的theta_old在H−实际生成，保存原response IDs与old LP | IMPLEMENTED | [training/trainer.py:88](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:88)–104；[training/rollout.py:50](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:50)–62 | 同一BehaviorPolicySnapshot传runner与scorer；三批次测试检查缓存概率、ID与采样次数 | 未复制第二份H−模型，也未重新采样监督action |
| 3 Rollout | 不让target guidance进入student，保留公共历史与retained模块 | IMPLEMENTED | [training/rollout.py:44](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:44)；[harness/runtime.py:53](/data/miyapeng/harness-internalization/src/internalization/harness/runtime.py:53)；[tests/test_behavior_policy.py:152](/data/miyapeng/harness-internalization/tests/test_behavior_policy.py:152) | 原隔离测试与新增retained模块同策略/同原始prompt重算测试通过 | 默认HF aux为greedy，非target指导重算条件一致；自定义随机aux后端需控制其随机性 |
| 4 Scorer / P0-1 | H+ scorer每批与behavior policy同步，不能停在cycle初始权重 | IMPLEMENTED | [training/trainer.py:88](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:88)–104；[training/behavior_policy.py:16](/data/miyapeng/harness-internalization/src/internalization/training/behavior_policy.py:16)–50 | 同一policy对象顺序运行，无权重copy延迟；[tests/test_behavior_policy.py:88](/data/miyapeng/harness-internalization/tests/test_behavior_policy.py:88)验证policy-0/1/2逐批对齐 | 默认单进程已修复；不是“补一个阶段冻结teacher_minus” |
| 4 Scorer | 对同state、同student token序列进行H+重评分 | IMPLEMENTED | [training/teacher_scoring.py:35](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:35)–58 | scorer逐trajectory验证model_version；读取既有response_ids，只额外生成模块指导；三批次每batch两次H− score、两次H+ score，ID均相同 | 不给scorer环境句柄，不以重新生成的action代替学生token |
| 4 Scorer | rollout/scoring期间策略不更新，所有推理no-grad | IMPLEMENTED | [training/behavior_policy.py:31](/data/miyapeng/harness-internalization/src/internalization/training/behavior_policy.py:31)–50；[training/trainer.py:105](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:105) | 参数/缓冲区版本与snapshot守卫；测试拒绝中途参数变化，检查eval/inference，退出只读作用域后才能梯度更新 | GPU/分布式冻结与数值路径仍NOT YET VERIFIED |
| 4 Scorer | 无module评分效应时差分为零，即使参数跨batch变化 | IMPLEMENTED | [tests/test_behavior_policy.py:125](/data/miyapeng/harness-internalization/tests/test_behavior_policy.py:125) | effect=0但每批仍执行真实CPU参数更新，三批H+ LP均与缓存H− LP逐值相等，module term严格为零 | 这是受控“无评分效应”回归，不宣称任意自然语言“do nothing”提示都零效应；空advice标记仍可能改变context |
| 5 Advantage | outcome + lambda*g*(teacher−old)，公式不需更换 | IMPLEMENTED | [training/module_advantage.py:43](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:43)–69；[training/trainer.py:45](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:45)–62 | 减数与被减数现在来自同批权重；原优势fixture保持通过；新增测试期望0.001×0.5=0.0005 | outcome仍为原step-weighted组归一化，未改成trajectory均权GRPO |
| 5 Advantage | snapshot、mask、同token校验；旧teacher/旧rollout不混用 | IMPLEMENTED | [training/trainer.py:47](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:47)；[training/trainer.py:99](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:99)；[training/teacher_scoring.py:36](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:36) | scorer与batch builder均拒绝不匹配snapshot；生成response外的prompt/tool文本不进入response mask；g=False term为零 | 旧TensorTeacherScorer为兼容测试接口，非本轮生产入口；其调用方仍须自行提供正确snapshot |
| 5 Update | advantage真正传actor，更新后下一batch使用新权重 | PARTIAL | [training/trainer.py:105](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:105)–114；[training/verl_backend.py:66](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:66)–84 | CPU mock执行SGD并推进snapshot；真实adapter在update前校验batch behavior_snapshot，再传external veRL | 接线已实现；真实veRL PPO、GPU和模型重载未运行 |
| 5 Reference | 阶段KL reference与batch module scorer分离 | IMPLEMENTED | [training/trainer.py:77](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:77)–81；[training/verl_backend.py:61](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:61)–77 | H+ scorer使用behavior，teacher_loader返回对象只交set_reference；日志分开记录两个snapshot；测试禁止phase reference生成模块指导 | 保留原KL正则设定；其实际GPU数值路径仍未验证 |
| 6 Retirement | 四格配对、task cluster、能力/成本与最终退役字段 | IMPLEMENTED | [evaluate_retirement](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:70) | 原退役能力/成本检查保留；新增capability_preserved、cost_improved、retirement_eligible、retirement_accepted | 实际HF部署成本NOT YET VERIFIED；模型rollback优先于退役 |
| 7 Acceptance | 模型accept/rollback与模块retire/retain三分支 | IMPLEMENTED | [evaluate_model_acceptance](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:51)；[outer loop接受分支](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:98)；[三分支集成测试](/data/miyapeng/harness-internalization/tests/test_model_acceptance.py:79) | A=1/C=0回滚；C不退化但D不足则接受并保留；D达标则接受并退役；完整记录旧/候选/采用checkpoint | 采用“拒绝显著退化”，并非统计上证明新模型非劣；独立任务不足保守回滚 |
| 8 Repeat | 三分支后下一周期使用正式采用的checkpoint与residual Harness | IMPLEMENTED | [checkpoint/Harness分支](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:104)；[跨周期测试](/data/miyapeng/harness-internalization/tests/test_model_acceptance.py:106) | accept/retain→rollback/retain→accept/retire→归因discard，检查下轮proposer、search、A/B、trainer与KL reference输入 | 真实GPU checkpoint重载NOT YET VERIFIED；旧retained模块后续单独再审计仍MISSING |

## P1-1：AppWorld接入审计

| Stage | Requirement | Status | Code location | Evidence | Gap / Action |
| ----- | ----------- | ------ | ------------- | -------- | ------------ |
| Benchmark | 官方AppWorld reset/execute/terminal grader，进程隔离 | IMPLEMENTED | [NativeAppWorld](/data/miyapeng/harness-internalization/src/internalization/benchmarks/appworld_worker.py:23) | 外部包API、自写adapter；不转发hidden ground truth；13项mock测试覆盖 | 包/数据下载被网络代理阻断，真实engine NOT YET VERIFIED |
| 1 / 2 / 6 | 官方task ID、scenario分组、3个retirement cohort及test隔离 | IMPLEMENTED | [build_manifest](/data/miyapeng/harness-internalization/src/internalization/benchmarks/appworld_manifest.py:22) | 训练/search只用train，cohort>=30；task×seed统计沿用原P0协议 | scenario级不确定性分析未实现；真实manifest未生成 |
| 3 / 4 / 5 | AppWorld H− rollout→同batch H+评分→原trainer更新 | IMPLEMENTED / MOCK VERIFIED | [entrypoint分派](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:46)；[两批更新测试](/data/miyapeng/harness-internalization/tests/test_appworld.py:178) | 实际环境子进程通信、自写stub、CPU torch参数更新与response mask检查通过 | 真实HF/veRL/GPU NOT YET VERIFIED；非官方任务成绩 |
| Final evaluation | TGC/完整scenario SGC，test评价不回流outer loop | IMPLEMENTED | [aggregate_results](/data/miyapeng/harness-internalization/src/internalization/benchmarks/appworld_manifest.py:59)；[最终评价脚本](/data/miyapeng/harness-internalization/scripts/evaluate_appworld.py:16) | 不完整scenario不报SGC；checkpoint/state配对；只打印汇总 | 官方test未运行 |

完整协议、配置字段、源码和来源见 [AppWorld说明](benchmarks/APPWORLD.md)，逐项排期见 [BENCHMARK_ROADMAP.md](BENCHMARK_ROADMAP.md)。本轮没有同时接TB2/SWE-bench Pro/HotpotQA/LawBench，也没有重写P0算法。之前的审计另存 [history/IMPLEMENTATION_AUDIT-before-appworld.md](history/IMPLEMENTATION_AUDIT-before-appworld.md)。

## P0-2：训练前归因准入

在search/dev选出一个候选后，使用当前checkpoint和本轮`retirement_k` cohort分别采集A/B，进入trainer前执行：

```text
delta_external = paired_task_bootstrap(A - B)
passed = (independent_tasks >= min_tasks
          and delta_external.low > min_external_gain)
if not passed:
    archive(candidate, reason="no_external_contribution")
    keep(current_checkpoint, current_residual_harness)
    continue
train(...)
```

| 字段 | 固定默认值 | 实际实现 |
| --- | --- | --- |
| `min_external_gain` | 0.0；区间下界必须严格大于它 | `evaluation/attribution.py:15,39` |
| `confidence` | 0.95；双侧百分位bootstrap区间 | `evaluation/attribution.py:16`、`evaluation/retirement.py:34` |
| `bootstrap_samples` / `seed` | 2000 / 42 | `evaluation/attribution.py:17,19` |
| `min_tasks` | 30个独立task ID；同任务多seed先取均值 | `evaluation/attribution.py:18,38`、`evaluation/retirement.py:39` |
| CLI加载 | `--attribution-policy configs/attribution.json`；省略则使用上述默认值 | `cli.py:27,71` |

`AttributionPolicy`不可变，运行开始、任何rollout之前写入`attribution_policy.json`；不能依据本轮A/B结果调整门槛。均值大于零但区间包含零仍不准入；配对缺失、重复或seed不一致会报错，不能绕过训练门槛。`no_external_contribution`表示没有通过预设贡献证据门槛，不等于证明真实效应恰为零；`checks`区分收益不足与独立任务不足。

`cycle_XX/attribution.json`保存判定、区间、策略、checkpoint、candidate ID、parent、module版本、H+/H−版本、manifest hash、partition、seeds和A/B证据路径。`CandidateArchive.record_attribution`另存候选源码与content hash；拒绝路径还保存不变的`state.json`及`training_batches_spent=0`。不调用trainer，不生成C/D或retirement结论，不重分配本轮未花费训练预算；下一周期从原模型和原residual Harness继续。归档中的`decision=internalize`只表示准许训练，不是“内化成功”。

源码入口：[AttributionPolicy / evaluate_attribution](../src/internalization/evaluation/attribution.py)、[outer loop准入分支](../src/internalization/outer_loop.py)、[候选归档和proposer过滤](../src/internalization/evolution/archive.py)。归因事件整条排除在`proposer_history()`之外，防止通过事件数量或状态泄漏retirement标签；候选search历史仍保留。当前沿用每轮独立cohort：A/B用于准入后供同轮四格审计复用，C/D为训练后评价；它们并非四份互不相关的数据，也不能把已用于准入的cohort称为完全未参与选择的最终test。最终test仍不进入outer loop。

## P0-3：模型接受与模块退役分离

用相同task/seed的A/C，在每个task内先对seed差分取均值，再按task ID做配对bootstrap。模型接受复用运行前已写入`protocol.json.retirement`的固定`RetirementPolicy`：`performance_margin=0.02`、`confidence=0.95`、`bootstrap_samples=2000`、`min_tasks=30`、`seed=42`。CLI外层默认即此策略，Python调用可在运行前传入policy；描述性`experiment_protocol.json`本身不驱动额外配置加载。

```text
full_degraded = upper_CI(C - A) < -performance_margin
model_decision = rollback if insufficient_tasks or full_degraded else accept
retirement_eligible = all(existing_capability_and_cost_checks)
retirement_accepted = model_decision == accept and retirement_eligible
module_decision = retire if retirement_accepted else retain
```

| 条件 | model_decision | module_decision | 下一轮采用系统 |
| --- | --- | --- | --- |
| C没有显著退化，D能力/成本满足原退役要求 | accept | retire | 新checkpoint + H− |
| C没有显著退化，D不满足退役要求 | accept | retain | 新checkpoint + H+ |
| C显著退化 | rollback | retain | 训练前checkpoint + H+ |

区间上界恰等于−ε时不算“明显退化”；平均C−A为负但区间跨越门槛时也不回滚。这是用户指定的“没有明显退化即可接受”，不能报告成通过非劣检验。任务不足时以`insufficient_independent_tasks`拒绝模型，不把多seed当独立任务，也不伪称检测到了退化；缺失/重复/不配对数据直接报错。

`model_acceptance`单独保存reason、C−A区间、checks和policy；`capability_preserved`汇总原有五项能力/归因/样本量检查，`cost_improved`汇总原成本检查。`retirement_eligible`表示忽略模型接受时原退役条件是否满足；`retirement_accepted`还要求模型被接受。即使A=1、C=0、D=1导致原退役条件通过，也必须rollback+retain，不能部署被拒绝的新模型。

`retirement.json`、`state.json`、`events.jsonl`的cycle_complete与deployment的逐轮archive分别记录`model_decision`、`module_decision`，以及`before_checkpoint`、`proposed_checkpoint`、`accepted_checkpoint`。旧`decision`保留为模块决策的兼容别名，outer loop不再用它接受模型。归因失败的周期没有训练、没有模型接受判定，仍按P0-2记录discard。

回滚是恢复正式采用的checkpoint引用；本轮新模块和已有模块都保留。被拒绝的checkpoint与训练/评价产物保留供排查，不清除已花费预算。下轮runner/proposer/trainer/KL reference接收正式采用的旧checkpoint，默认trainer由该路径重新加载policy。`CandidateArchive.record_outcome`存完整候选source/parent/hash与结果，outcome事件不传给proposer。自定义RetirementEvaluator必须提供合法双字段，只返回旧decision或rollback+retire会报错，不能默认为接受。

## 角色、配置与日志

| 名称 | 当前含义 | 来源/位置 |
| --- | --- | --- |
| `student` / `theta_old` | 同一个可训练policy在本次update之前的参数状态 | `BehaviorPolicySnapshot(student)`，[training/trainer.py:88](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:88) |
| `teacher_log_prob` / `ModuleSignal.log_probs` | 本batch behavior policy在H+ context对原student response的打分 | [training/teacher_scoring.py:50](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:50) |
| `old_log_probs` | 该behavior policy在H−生成后、更新前评分的缓存；即ell_minus | [training/rollout.py:55](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:55)–62 |
| `teacher` / `teacher_checkpoint`旧接口参数 | 仅作为本阶段冻结KL reference，不再生成module监督 | [training/trainer.py:66](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:66)、[training/entrypoint.py:72](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:72) |
| `reference` / `ref_log_prob` | PPO KL正则参考，仍采用阶段开始checkpoint | [training/verl_backend.py:61](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:61)–77 |
| `teacher_refresh` | `each_batch_behavior_policy_before_update` | [configs/experiment_protocol.json:11](/data/miyapeng/harness-internalization/configs/experiment_protocol.json:11) |
| `module_signal` | `same_behavior_policy_H_plus_logprob_minus_cached_H_minus_old_logprob` | [configs/experiment_protocol.json:12](/data/miyapeng/harness-internalization/configs/experiment_protocol.json:12) |
| `kl_reference_refresh` | `between_cycles_only` | [configs/experiment_protocol.json:13](/data/miyapeng/harness-internalization/configs/experiment_protocol.json:13) |
| update日志 | behavior_snapshot=teacher_snapshot；student_snapshot为更新后；kl_reference_snapshot单列 | [training/trainer.py:113](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:113) |
| checkpoint manifest | last_behavior_snapshot、student_snapshot、kl_reference_snapshot；teacher_snapshot兼容名指最后评分批次 | [training/checkpoint.py:9](/data/miyapeng/harness-internalization/src/internalization/training/checkpoint.py:9)–17 |

`experiment_protocol.json`仍是描述性协议文件，当前CLI未自动读取整份JSON；本次行为由实际trainer调用链保证，同时将描述改到一致。归因策略已单独通过CLI加载并传入outer loop；其余统一配置加载仍是P1，不将修改JSON本身当作实现。lambda=0.001、g=trigger/persistence、outcome/PPO/KL/entropy配置未变。

原`FrozenHFBackend`仍用于只读评价和KL reference。模块评分改用behavior只读视图后不需要额外H+权重副本；代价仍包括H+模块调用与对原action的评分，这些成本照常记账。

## 验证记录与范围

本轮完整日志：[validation/acceptance-tests.txt](validation/acceptance-tests.txt)。机器可读记录：[validation/acceptance-report.json](validation/acceptance-report.json)。P0-2历史验证另见 [validation/attribution-report.json](validation/attribution-report.json)。P0-1的历史验证另见 [validation/batch-policy-report.json](validation/batch-policy-report.json)。

- 最新71项测试通过，无skip：原20、迁移12、batch对齐7、归因9、模型接受10、AppWorld13。P0历史记录原样保留；本轮日志与报告见 [appworld-tests.txt](validation/appworld-tests.txt)、[appworld-report.json](validation/appworld-report.json)。
- 三分支、严格容差、统计不确定性、独立任务不足、成本失败不拒绝模型、回滚优先、拒绝checkpoint产物保留、跨周期模型/Harness传播和自定义评估器拒绝旧单字段均有测试。持久化mock证据见`runs/acceptance-proof/summary.json`，摘要写入本轮报告。
- 归因覆盖零/负/不确定收益、严格固定阈值、task与seed区别、缺失/重复配对、实际配置加载、零trainer调用、正例训练/退役、已有模块和模型保留、proposer隔离。持久化mock证据：`runs/attribution-proof/summary.json`，摘要包含于上述报告。
- 原迁移trainer mock调整为由同一当前policy产生模块context差异；不再把teacher/student固定偏差当作学习收益。原优势算术fixture未修改。
- 三个训练batch均发生实际CPU torch参数更新：H−、H+权重依次为(policy-0,policy-0)、(policy-1,policy-1)、(policy-2,policy-2)，更新后为policy-3。
- 检查相同response IDs、无额外最终action采样、模块效应为零时无参数漂移污染、未触发时零term、retained模块同策略重算、snapshot过期及中途参数修改拒绝。
- CPU三周期toy demo的原有轨迹/分数/决策与迁移前一致：77文件逐字节相同，8文件仅增加接受/回滚字段与C−A区间。显式`--allow-model-acceptance-fields`只移除约定新增字段后检查旧字段全部相等；原字节检查默认仍严格，反例测试确认旧分数或decision变化不会被放过。toy不执行神经模型优势路径，不作为GPU或真实训练证据。
- 未运行：真实HF模型、external veRL optimizer、GPU、真实benchmark、API proposer；分布式快照同步未实现于默认串行adapter。没有把CPU mock的SGD测试称为真实PPO/GPU复现。

## A. 已完整实现

P0-1在单进程生产调用链中已实现：每batch同behavior策略的H−采样与H+评分、原action token重用、缓存old LP相减、推理作用域内不更新、更新后下一batch重新绑定、可审计snapshot日志。原有outcome、mask、门控、独立KL reference、四格评价继续存在。P0-2已实现固定统计门槛、训练前拦截、候选discard归档、residual状态保留及proposer标签隔离；9项专门测试通过。P0-3的模型接受/模块退役分离、回滚保留模块、正式checkpoint跨周期传播和结果归档已实现，10项新增回归通过。

## B. 已有框架但算法语义不一致

按最终定义，`teacher_log_prob-old_log_prob`不再是错误公式；之前真正的问题是H+ scorer停留在cycle初始权重。P0-1修复同步，旧审计“必须增加阶段冻结H−评分”的修复建议已撤销。

其余已审计的语义缺口仍存在：模型接受与退役已分开；outcome采用step-weighted组归一化，论文需准确说明；toy训练仍为表格更新，不能代表目标算法。

## C. 后续优先级

| Priority | 项目 | 当前状态 |
| --- | --- | --- |
| P0-1 | H+ scorer与当前batch behavior-policy snapshot同步 | IMPLEMENTED；真实GPU验证NOT YET VERIFIED |
| P0-2 | A/B收益不达标禁止训练、discard模块 | IMPLEMENTED；CPU反例与正例已验证 |
| P0-3 | C−A明显退化时rollback模型并保留模块 | IMPLEMENTED；三分支与跨周期CPU反例通过 |
| P0验证 | 真实veRL最小更新、token概率/mask、checkpoint再加载、真实成本验证 | NOT YET VERIFIED |
| P1 | retained模块后续再审计；统一配置加载；真实模型端到端接受/回滚验证 | MISSING/PARTIAL |
| P2 | 生命周期归档统一关联、恢复运行入口、真实成本口径、候选采样覆盖 | PARTIAL |

**现在启动真实GPU实验会测什么？** 若依赖和adapter正常运行，模块监督将测试用户确定的“same batch policy, different scaffolding”，不会再混入cycle初始教师与当前学生的参数差异；训练前Attribute准入也已实现；本轮也已补齐模型accept/rollback与模块retire/retain三分支。因此默认调用链已实现这三项P0修复，已不再是无条件接受新模型的近似版；但真实veRL/GPU运行仍NOT YET VERIFIED，旧retained模块后续再审计等P1未完成，不能据CPU测试宣称真实训练有效或整套研究协议均已验证。
