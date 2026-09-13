# 算法实现一致性审计：批次策略对齐与训练前归因门槛

2026-09-13更新。**P0-2已实现：A/B配对归因通过预先固定门槛才训练，否则discard并保留原模型与residual Harness。** **P0-1正式定义为“H+ scorer与当前batch behavior-policy snapshot的同步”，当前已在默认单进程训练路径实现并通过CPU回归。** 不再要求增加阶段冻结的teacher_minus评分，`teacher_log_prob - old_log_prob`公式保留。

先前按“整个阶段冻结teacher、双路冻结评分”目标生成的完整只读审计保存在 [history/IMPLEMENTATION_AUDIT-stage-frozen.md](history/IMPLEMENTATION_AUDIT-stage-frozen.md)。其中的源码行号、反例、缺口结论对应修改前实现和当时目标；本页以用户最新确定的batch-aligned self-distillation与训练前Attribution Gate为准。P0-2修改前审计另存 [history/IMPLEMENTATION_AUDIT-before-attribution.md](history/IMPLEMENTATION_AUDIT-before-attribution.md)。

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

生产调用链定位：[training/trainer.py:85](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:85) `ModuleTrainer.train` → [training/behavior_policy.py:10](/data/miyapeng/harness-internalization/src/internalization/training/behavior_policy.py:10) `BehaviorPolicySnapshot` → [training/rollout.py:49](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:49) `InteractionTaskRunner.rollout` → [training/teacher_scoring.py:30](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:30) `ModuleTeacherScorer.score` → [training/trainer.py:19](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:19) `build_update_batch` → [training/trainer.py:105](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:105) `student.update`。

这里的snapshot是串行训练中的只读策略视图，并未复制完整权重。作用域内不能进行优化器更新；常规torch参数/缓冲区in-place版本变化、模型替换、snapshot ID变化、train-mode变化均拒绝。它不是用于任意并发写入的互斥锁，也不是权重内容checksum；分布式backend必须提供自己的同步/冻结保障。

## 逐项实现状态

| Stage | Requirement | Status | Code location | Evidence | Gap / Action |
| ----- | ----------- | ------ | ------------- | -------- | ------------ |
| 1 Evolve | 同一当前checkpoint、residual Harness、多候选、search/dev评分与无收益跳训 | IMPLEMENTED | [outer_loop.py:50](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:50)；[evolution/search.py:17](/data/miyapeng/harness-internalization/src/internalization/evolution/search.py:17) | 搜索评分不变；返回含parent/hash的Candidate供归因归档；原20测试与外层回归通过 | 真实API/benchmark搜索仍NOT YET VERIFIED |
| 2 Attribute | 同checkpoint、同任务/seed的A/B，H+/H−只差target | IMPLEMENTED | [outer_loop.py:61](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:61)；[tests/test_attribution.py:127](/data/miyapeng/harness-internalization/tests/test_attribution.py:127) | 同checkpoint、任务和seed调用；只增加本轮module；测试验证两侧参数 | 真实benchmark评价NOT YET VERIFIED |
| 2 Attribute | A−B不达预先固定统计门槛则discard，禁止internalization | IMPLEMENTED | [evaluation/attribution.py:34](/data/miyapeng/harness-internalization/src/internalization/evaluation/attribution.py:34)；[outer_loop.py:69](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:69)；[tests/test_attribution.py:105](/data/miyapeng/harness-internalization/tests/test_attribution.py:105) | task-cluster收益区间下界严格超过min_external_gain；A=B时trainer零调用、不跑C/D、不接受新module | P0-2 CPU回归通过；通过表示允许训练，尚不表示内化成功 |
| 3 Rollout | 当前batch的theta_old在H−实际生成，保存原response IDs与old LP | IMPLEMENTED | [training/trainer.py:88](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:88)–104；[training/rollout.py:49](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:49)–62 | 同一BehaviorPolicySnapshot传runner与scorer；三批次测试检查缓存概率、ID与采样次数 | 未复制第二份H−模型，也未重新采样监督action |
| 3 Rollout | 不让target guidance进入student，保留公共历史与retained模块 | IMPLEMENTED | [training/rollout.py:43](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:43)；[harness/runtime.py:53](/data/miyapeng/harness-internalization/src/internalization/harness/runtime.py:53)；[tests/test_behavior_policy.py:152](/data/miyapeng/harness-internalization/tests/test_behavior_policy.py:152) | 原隔离测试与新增retained模块同策略/同原始prompt重算测试通过 | 默认HF aux为greedy，非target指导重算条件一致；自定义随机aux后端需控制其随机性 |
| 4 Scorer / P0-1 | H+ scorer每批与behavior policy同步，不能停在cycle初始权重 | IMPLEMENTED | [training/trainer.py:88](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:88)–104；[training/behavior_policy.py:16](/data/miyapeng/harness-internalization/src/internalization/training/behavior_policy.py:16)–50 | 同一policy对象顺序运行，无权重copy延迟；[tests/test_behavior_policy.py:88](/data/miyapeng/harness-internalization/tests/test_behavior_policy.py:88)验证policy-0/1/2逐批对齐 | 默认单进程已修复；不是“补一个阶段冻结teacher_minus” |
| 4 Scorer | 对同state、同student token序列进行H+重评分 | IMPLEMENTED | [training/teacher_scoring.py:35](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:35)–58 | scorer逐trajectory验证model_version；读取既有response_ids，只额外生成模块指导；三批次每batch两次H− score、两次H+ score，ID均相同 | 不给scorer环境句柄，不以重新生成的action代替学生token |
| 4 Scorer | rollout/scoring期间策略不更新，所有推理no-grad | IMPLEMENTED | [training/behavior_policy.py:31](/data/miyapeng/harness-internalization/src/internalization/training/behavior_policy.py:31)–50；[training/trainer.py:105](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:105) | 参数/缓冲区版本与snapshot守卫；测试拒绝中途参数变化，检查eval/inference，退出只读作用域后才能梯度更新 | GPU/分布式冻结与数值路径仍NOT YET VERIFIED |
| 4 Scorer | 无module评分效应时差分为零，即使参数跨batch变化 | IMPLEMENTED | [tests/test_behavior_policy.py:125](/data/miyapeng/harness-internalization/tests/test_behavior_policy.py:125) | effect=0但每批仍执行真实CPU参数更新，三批H+ LP均与缓存H− LP逐值相等，module term严格为零 | 这是受控“无评分效应”回归，不宣称任意自然语言“do nothing”提示都零效应；空advice标记仍可能改变context |
| 5 Advantage | outcome + lambda*g*(teacher−old)，公式不需更换 | IMPLEMENTED | [training/module_advantage.py:43](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:43)–69；[training/trainer.py:45](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:45)–62 | 减数与被减数现在来自同批权重；原优势fixture保持通过；新增测试期望0.001×0.5=0.0005 | outcome仍为原step-weighted组归一化，未改成trajectory均权GRPO |
| 5 Advantage | snapshot、mask、同token校验；旧teacher/旧rollout不混用 | IMPLEMENTED | [training/trainer.py:47](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:47)；[training/trainer.py:99](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:99)；[training/teacher_scoring.py:36](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:36) | scorer与batch builder均拒绝不匹配snapshot；生成response外的prompt/tool文本不进入response mask；g=False term为零 | 旧TensorTeacherScorer为兼容测试接口，非本轮生产入口；其调用方仍须自行提供正确snapshot |
| 5 Update | advantage真正传actor，更新后下一batch使用新权重 | PARTIAL | [training/trainer.py:105](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:105)–114；[training/verl_backend.py:63](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:63)–84 | CPU mock执行SGD并推进snapshot；真实adapter在update前校验batch behavior_snapshot，再传external veRL | 接线已实现；真实veRL PPO、GPU和模型重载未运行 |
| 5 Reference | 阶段KL reference与batch module scorer分离 | IMPLEMENTED | [training/trainer.py:77](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:77)–81；[training/verl_backend.py:58](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:58)–77 | H+ scorer使用behavior，teacher_loader返回对象只交set_reference；日志分开记录两个snapshot；测试禁止phase reference生成模块指导 | 保留原KL正则设定；其实际GPU数值路径仍未验证 |
| 6 Retirement | 四格配对、task cluster bootstrap、能力/成本门槛 | IMPLEMENTED | [evaluation/retirement.py:34](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:34)–86 | 原统计/成本/冗余/共同退化测试全部通过，本轮共享bootstrap计算，不改退役规则 | 实际HF部署成本NOT YET VERIFIED；统一capability/cost/model verdict字段仍不齐 |
| 7 Acceptance | full H+训练明显退化时rollback | INCORRECT / SEMANTIC MISMATCH | [outer_loop.py:99](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:99)；[evaluation/retirement.py:58](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:58) | 仍无C−A模型接受门槛；retirement失败仍无条件接受new checkpoint | **P0-3仍待实现**，独立于已完成的P0-1/P0-2 |
| 8 Repeat | 正常retire/retain后传递residual Harness和模型 | PARTIAL | [outer_loop.py:99](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:99)–106 | 正常跨周期留存及归因失败后保留旧模型/旧模块测试通过 | 缺rollback路径；已retained旧模块后续单独再审计仍MISSING |

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

## 角色、配置与日志

| 名称 | 当前含义 | 来源/位置 |
| --- | --- | --- |
| `student` / `theta_old` | 同一个可训练policy在本次update之前的参数状态 | `BehaviorPolicySnapshot(student)`，[training/trainer.py:88](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:88) |
| `teacher_log_prob` / `ModuleSignal.log_probs` | 本batch behavior policy在H+ context对原student response的打分 | [training/teacher_scoring.py:50](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:50) |
| `old_log_probs` | 该behavior policy在H−生成后、更新前评分的缓存；即ell_minus | [training/rollout.py:54](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:54)–62 |
| `teacher` / `teacher_checkpoint`旧接口参数 | 仅作为本阶段冻结KL reference，不再生成module监督 | [training/trainer.py:66](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:66)、[training/entrypoint.py:54](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:54) |
| `reference` / `ref_log_prob` | PPO KL正则参考，仍采用阶段开始checkpoint | [training/verl_backend.py:58](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:58)–77 |
| `teacher_refresh` | `each_batch_behavior_policy_before_update` | [configs/experiment_protocol.json:11](/data/miyapeng/harness-internalization/configs/experiment_protocol.json:11) |
| `module_signal` | `same_behavior_policy_H_plus_logprob_minus_cached_H_minus_old_logprob` | [configs/experiment_protocol.json:12](/data/miyapeng/harness-internalization/configs/experiment_protocol.json:12) |
| `kl_reference_refresh` | `between_cycles_only` | [configs/experiment_protocol.json:13](/data/miyapeng/harness-internalization/configs/experiment_protocol.json:13) |
| update日志 | behavior_snapshot=teacher_snapshot；student_snapshot为更新后；kl_reference_snapshot单列 | [training/trainer.py:113](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:113) |
| checkpoint manifest | last_behavior_snapshot、student_snapshot、kl_reference_snapshot；teacher_snapshot兼容名指最后评分批次 | [training/checkpoint.py:9](/data/miyapeng/harness-internalization/src/internalization/training/checkpoint.py:9)–17 |

`experiment_protocol.json`仍是描述性协议文件，当前CLI未自动读取整份JSON；本次行为由实际trainer调用链保证，同时将描述改到一致。归因策略已单独通过CLI加载并传入outer loop；其余统一配置加载仍是P1，不将修改JSON本身当作实现。lambda=0.001、g=trigger/persistence、outcome/PPO/KL/entropy配置未变。

原`FrozenHFBackend`仍用于只读评价和KL reference。模块评分改用behavior只读视图后不需要额外H+权重副本；代价仍包括H+模块调用与对原action的评分，这些成本照常记账。

## 验证记录与范围

本轮完整日志：[validation/attribution-tests.txt](validation/attribution-tests.txt)。机器可读记录：[validation/attribution-report.json](validation/attribution-report.json)。P0-1的历史验证另见 [validation/batch-policy-report.json](validation/batch-policy-report.json)。

- 48项测试全部通过，无skip：原20测试文件保持原SHA256，迁移12项，batch对齐7项，新增归因9项。
- 归因覆盖零/负/不确定收益、严格固定阈值、task与seed区别、缺失/重复配对、实际配置加载、零trainer调用、正例训练/退役、已有模块和模型保留、proposer隔离。持久化mock证据：`runs/attribution-proof/summary.json`，摘要包含于上述报告。
- 原迁移trainer mock调整为由同一当前policy产生模块context差异；不再把teacher/student固定偏差当作学习收益。原优势算术fixture未修改。
- 三个训练batch均发生实际CPU torch参数更新：H−、H+权重依次为(policy-0,policy-0)、(policy-1,policy-1)、(policy-2,policy-2)，更新后为policy-3。
- 检查相同response IDs、无额外最终action采样、模块效应为零时无参数漂移污染、未触发时零term、retained模块同策略重算、snapshot过期及中途参数修改拒绝。
- CPU三周期toy demo原85个输出文件仍与迁移前逐字节一致；toy不执行神经模型优势路径，因此只作外层回归，不作本次P0-1的主要证据。
- 未运行：真实HF模型、external veRL optimizer、GPU、真实benchmark、API proposer；分布式快照同步未实现于默认串行adapter。没有把CPU mock的SGD测试称为真实PPO/GPU复现。

## A. 已完整实现

P0-1在单进程生产调用链中已实现：每batch同behavior策略的H−采样与H+评分、原action token重用、缓存old LP相减、推理作用域内不更新、更新后下一batch重新绑定、可审计snapshot日志。原有outcome、mask、门控、独立KL reference、四格评价继续存在。P0-2已实现固定统计门槛、训练前拦截、候选discard归档、residual状态保留及proposer标签隔离；9项专门测试通过。

## B. 已有框架但算法语义不一致

按最终定义，`teacher_log_prob-old_log_prob`不再是错误公式；之前真正的问题是H+ scorer停留在cycle初始权重。P0-1修复同步，旧审计“必须增加阶段冻结H−评分”的修复建议已撤销。

其余已审计的语义缺口仍存在：退役失败与训练退化没有分开，缺C−A模型回滚；outcome采用step-weighted组归一化，论文需准确说明；toy训练仍为表格更新，不能代表目标算法。

## C. 后续优先级

| Priority | 项目 | 当前状态 |
| --- | --- | --- |
| P0-1 | H+ scorer与当前batch behavior-policy snapshot同步 | IMPLEMENTED；真实GPU验证NOT YET VERIFIED |
| P0-2 | A/B收益不达标禁止训练、discard模块 | IMPLEMENTED；CPU反例与正例已验证 |
| P0-3 | C−A明显退化时rollback模型并保留模块 | MISSING / 当前接受逻辑SEMANTIC MISMATCH |
| P0验证 | 真实veRL最小更新、token概率/mask、checkpoint再加载、真实成本验证 | NOT YET VERIFIED |
| P1 | retained模块后续再审计；统一配置加载；完整模型接受测试 | MISSING/PARTIAL |
| P2 | 生命周期归档统一关联、恢复运行入口、真实成本口径、候选采样覆盖 | PARTIAL |

**现在启动真实GPU实验会测什么？** 若依赖和adapter正常运行，模块监督将测试用户确定的“same batch policy, different scaffolding”，不会再混入cycle初始教师与当前学生的参数差异；训练前Attribute准入也已实现；但Rollback与旧模块再审计尚缺，完整Evolve→Attribute→Internalize→Retire/Retain/Rollback仍未完成，不能宣称整套方法已经实现或已经GPU验证。
