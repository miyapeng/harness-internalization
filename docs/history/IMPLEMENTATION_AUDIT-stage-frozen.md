# 算法实现一致性审计

审计日期：2026-09-13。项目：`/data/miyapeng/harness-internalization`。

**结论：当前代码是“可执行 Harness 增强教师的定向 on-policy 蒸馏＋四格退役”的近似实现，尚不完整对应本轮确定的算法。三个 P0 缺口是：没有 frozen-teacher H+/H− 模块差分、没有训练前 A/B contribution gate、没有训练退化后的 checkpoint rollback。** 已保留模块的后续再退役审计也未实现。

本轮只新增本文档。未修改源代码、测试、配置、运行脚本，未重构或新增 benchmark。依据为实际入口、调用链、函数体、配置读取及测试；没有用 README 的描述替代实现证据。当前工程文件尚未被 Git 跟踪，不能用 commit 作为本次源码定位依据；以下行号对应本次读取的工作区，审计前后以文件 SHA256 比较确认原文件未变。

## 范围、状态和执行证据

| Status | 本文含义 |
| --- | --- |
| IMPLEMENTED | 默认实际路径有对应实现；证据栏注明现有测试或本轮 CPU 探针。只代表所列功能，不代表 GPU 有效性 |
| PARTIAL | 有部分必要环节，但目标约束、完整接口或证据不齐 |
| MISSING | 默认调用链中没有实现该机制 |
| INCORRECT / SEMANTIC MISMATCH | 当前可执行逻辑与目标定义直接不同；不是简单缺少运行验证 |
| NOT YET VERIFIED | 有接线或预期行为，但未在所要求的真实模型、环境或 GPU 后端验证 |

本轮使用 Python 3.12.3、`PYTHONDONTWRITEBYTECODE=1` 运行原有测试，**32/32 通过，0 skip**，耗时 3.074 s。测试代码未改；输出与临时反例探针仅写入 `/tmp/hi-implementation-audit-qy1_s5k9/`。探针结果在本文完整说明，不依赖临时目录长期保存。

该解释器可导入 torch/numpy/PyYAML，不能导入 transformers/verl/alfworld/ray。本轮没有安装依赖、调用真实 proposer API、运行真实 benchmark 或 GPU，也没有重复把三周期 toy demo 当作算法验证。

实际生产调用链：

```text
cli.main(run)
  → CommandBackend.components()
  → run_outer_loop()
      → search_candidates() → APIProposer / TaskRunner
      → A / B evaluation
      → ModuleTrainer.train()
          → InteractionTaskRunner.rollout(student, H−)
          → ModuleTeacherScorer.score(frozen teacher, H+)
          → build_update_batch() → combine_advantages()
          → VerlPolicy.update() → external veRL.update_policy()
      → PairedRetirementEvaluator.evaluate(C / D; reuse already collected A / B)
      → checkpoint = new_checkpoint
      → H− if retire else H+
```

定位：[cli.py:61](/data/miyapeng/harness-internalization/src/internalization/cli.py:61) `main`；[command_backend.py:19](/data/miyapeng/harness-internalization/src/internalization/command_backend.py:19) `components`；[training/entrypoint.py:43](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:43) `main`；[training/trainer.py:68](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:68) `ModuleTrainer.train`。

## 逐项检查表

### Stage 1 — Harness Evolution

| Stage | Requirement | Status | Code location | Evidence | Gap / Action |
| ----- | ----------- | ------ | ------------- | -------- | ------------ |
| 1.1 | 固定当前模型进行 candidate search | IMPLEMENTED | [outer_loop.py:46](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:46) `run_outer_loop`；[evolution/search.py:33](/data/miyapeng/harness-internalization/src/internalization/evolution/search.py:33) `search_candidates`；[training/teacher_backend.py:35](/data/miyapeng/harness-internalization/src/internalization/training/teacher_backend.py:35) `FrozenHFBackend.__init__` | baseline/candidate search/dev传同一checkpoint；搜索阶段无update；默认HF模型eval且requires_grad=False。本轮P1–P3探针记录相同`before`模型 | 这是默认路径固定权重；没有跨全部评价子进程校验权重内容checksum，外部自行覆盖同一路径不受完整防护 |
| 1.2 | 每轮从当前 residual H_t 出发 | IMPLEMENTED | [outer_loop.py:42](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:42)、[outer_loop.py:57](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:57)、[outer_loop.py:73](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:73) | 循环外只初始化一次`Harness()`；随后传当前harness，追加本轮module。`OuterLoopTests.test_retained_module_survives_and_later_cycles_continue`，[tests/test_outer_loop.py:13](/data/miyapeng/harness-internalization/tests/test_outer_loop.py:13) | 正常连续调用成立；每次重新启动`run_outer_loop`都从空模块集开始，没有加载已存residual state恢复运行的入口 |
| 1.3 | 生成多个 executable candidates | IMPLEMENTED | [outer_loop.py:15](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:15) `LoopConfig.candidates_per_cycle=2`；[evolution/proposer.py:64](/data/miyapeng/harness-internalization/src/internalization/evolution/proposer.py:64) `APIProposer.propose`；[harness/module.py:70](/data/miyapeng/harness-internalization/src/internalization/harness/module.py:70) `HarnessModule.from_source` | 数量严格匹配，源码经受限AST解释器；mock proposer测试 [tests/test_migration.py:28](/data/miyapeng/harness-internalization/tests/test_migration.py:28) | 仅触发表达式、instruction、kind和persistence可演化，三类调用流程固定；不是任意Python控制程序搜索 |
| 1.4 | search/dev均以任务评分选择 | IMPLEMENTED | [evolution/search.py:8](/data/miyapeng/harness-internalization/src/internalization/evolution/search.py:8) `evaluate_tasks`；[evolution/search.py:33](/data/miyapeng/harness-internalization/src/internalization/evolution/search.py:33) | 校验完整task×seed；对`EpisodeResult.success`做配对bootstrap；search/dev的gain.low均>0；按dev gain.mean、较低token排序。P1–P3探针实际经过此函数 | 分数来自runner而非proposer自报；真实环境有效性单列为1.5 |
| 1.5 | 已用真实任务完成搜索验证 | NOT YET VERIFIED | [training/entrypoint.py:44](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:44)；[benchmarks/alfworld.py:59](/data/miyapeng/harness-internalization/src/internalization/benchmarks/alfworld.py:59) `reset`、[benchmarks/alfworld.py:78](/data/miyapeng/harness-internalization/src/internalization/benchmarks/alfworld.py:78) `step`；[demo.py:45](/data/miyapeng/harness-internalization/src/internalization/demo.py:45) | ALFWorld真实接口已接，reward/success来自env；现有测试是外部环境mock [tests/test_migration.py:77](/data/miyapeng/harness-internalization/tests/test_migration.py:77)。demo为确定性候选＋toy分数 | 尚无真实ALFWorld/API运行证据；不能把mock或demo判为真实搜索结果 |
| 1.6 | 无明确收益则本周期停止且不训练 | IMPLEMENTED | [evolution/search.py:39](/data/miyapeng/harness-internalization/src/internalization/evolution/search.py:39)；[outer_loop.py:54](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:54) | 无候选返回None，记录`cycle_skipped`后continue。本轮P1：train_calls=0、checkpoint保持before | 停止的是当前周期，后面的外层周期仍可继续；没有花掉该周期训练预算 |
| 1.7 | 保存源码、traces、score、parent/hash/version/history | IMPLEMENTED | [evolution/candidate.py:6](/data/miyapeng/harness-internalization/src/internalization/evolution/candidate.py:6) `Candidate`；[evolution/archive.py:12](/data/miyapeng/harness-internalization/src/internalization/evolution/archive.py:12) `CandidateArchive.record`；[training/rollout.py:65](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:65)；[outer_loop.py:59](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:59) | candidate源/parent/轮次version/index/content_hash/candidate_id、search/dev gain与status持久化；phase与四格文件保存module hash；归档测试 [tests/test_migration.py:44](/data/miyapeng/harness-internalization/tests/test_migration.py:44) | `parent`是父Harness hash；归档version是轮次，module.version才是源码hash。运行异常发生在search评价时没有专门写`execution_failed`归档；只覆盖语法invalid/评分rejected/eligible |
| 1.8 | proposer读取当前代码、成功/失败轨迹、分数和历史 | IMPLEMENTED | [outer_loop.py:47](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:47) `ProposalRequest`；[evolution/proposer.py:33](/data/miyapeng/harness-internalization/src/internalization/evolution/proposer.py:33)；[evolution/archive.py:21](/data/miyapeng/harness-internalization/src/internalization/evolution/archive.py:21) | 只传search任务；源码、轨迹、scores、历史候选进入API请求；历史不含dev/retirement分数；mock检查history到达prompt [tests/test_migration.py:28](/data/miyapeng/harness-internalization/tests/test_migration.py:28) | API只取前32个transition，未保证成功/失败均衡覆盖；raw完整trajectory存盘。proposer_history省略status/reason，失败诊断反馈不完整 |

### Stage 2 — Pre-training Attribution

| Stage | Requirement | Status | Code location | Evidence | Gap / Action |
| ----- | ----------- | ------ | ------------- | -------- | ------------ |
| 2.1 | A/B来自相同训练前checkpoint | IMPLEMENTED | [outer_loop.py:61](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:61) `run_outer_loop` | A与B均传`checkpoint`，发生在train前；本轮P2/P3记录before | 真实模型数值重现未验证，但调用链明确 |
| 2.2 | A/B相同任务、seed和评价协议 | IMPLEMENTED | [outer_loop.py:61](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:61)；[evolution/search.py:8](/data/miyapeng/harness-internalization/src/internalization/evolution/search.py:8)；[evaluation/retirement.py:53](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:53) | 同一`retirement_k`、`config.seeds`、runner；配对键异常测试 [tests/test_core.py:97](/data/miyapeng/harness-internalization/tests/test_core.py:97)，本轮P2/P3运行 | 没有独立attribution policy；已有参数只在search或最终retirement阶段使用 |
| 2.3 | H+/H−只差本轮target | IMPLEMENTED | [outer_loop.py:57](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:57)；[harness/module.py:118](/data/miyapeng/harness-internalization/src/internalization/harness/module.py:118) `Harness.without`；[training/trainer.py:69](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:69) | full=旧模块tuple+新module；reduced=full.without(target)；trainer校验hash相等；重复模块名被拒绝 | 该集合差异实现正确，不代表Stage4两个教师context已成对构造 |
| 2.4 | A−B不达标则discard且禁止训练 | MISSING | [outer_loop.py:62](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:62)–65；[evaluation/retirement.py:77](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:77) | A/B后立即调用trainer，既不算配对增益，也无if门槛。`module_was_useful`仅训练后参与退役。本轮P2：A=B=0.5仍train_calls=1 | P0：训练前增加预定义贡献门槛和discard路径；不能用search/dev收益替代指定A/B门槛 |
| 2.5 | 冗余模块不能记为内化成功 | PARTIAL | [evaluation/retirement.py:75](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:75)；[tests/test_core.py:89](/data/miyapeng/harness-internalization/tests/test_core.py:89) `test_preexisting_redundancy_is_not_internalization` | 最终retire要求`A-B.low>0`，因此冗余模块不会通过退役检查 | 但会先训练，之后retain而非discard；本轮P2实际接受新checkpoint并保留原本无贡献模块 |

### Stage 3 — Student On-policy Rollout

| Stage | Requirement | Status | Code location | Evidence | Gap / Action |
| ----- | ----------- | ------ | ------------- | -------- | ------------ |
| 3.1 | student实际在H−生成轨迹 | IMPLEMENTED | [training/trainer.py:81](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:81)；[training/rollout.py:43](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:43) `InteractionTaskRunner.rollout` | 每批调用`runner.rollout(student,h_minus,...,training=True)`；其runtime只有该harness；实际生成后才env.step。CPU参数更新mock [tests/test_migration.py:139](/data/miyapeng/harness-internalization/tests/test_migration.py:139) | 非静态成功轨迹模仿；真实HF/环境联跑未验证 |
| 3.2 | student prompt不含目标module输出 | IMPLEMENTED | [training/rollout.py:49](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:49)；[training/teacher_scoring.py:37](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:37)；[harness/runtime.py:53](/data/miyapeng/harness-internalization/src/internalization/harness/runtime.py:53) | reduced模块产生student advice；教师随后独立构造H+ prompt，不回写student；mock在student.generate中拒绝PRIVATE ADVICE [tests/test_migration.py:161](/data/miyapeng/harness-internalization/tests/test_migration.py:161) | retained模块指导允许存在；不是要求student prompt完全没有所有内部指导 |
| 3.3 | 环境、工具结果和公共历史保留 | IMPLEMENTED | [training/rollout.py:47](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:47)；[benchmarks/alfworld.py:35](/data/miyapeng/harness-internalization/src/internalization/benchmarks/alfworld.py:35)、[benchmarks/alfworld.py:78](/data/miyapeng/harness-internalization/src/internalization/benchmarks/alfworld.py:78) | 外部环境动作照常执行，结果进入下一state；ALFWorld保留task/current observation/admissible与最近5步；原格式fixture测试 [tests/test_migration.py:67](/data/miyapeng/harness-internalization/tests/test_migration.py:67) | “公开历史”是环境生成的当前可见窗口，并非无限历史。通用文件系统/沙箱工具不是已实现benchmark能力 |
| 3.4 | 只对student生成response建立loss mask | IMPLEMENTED | [training/rollout.py:52](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:52)；[training/trainer.py:28](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:28)–56 | prompt左padding、response右padding、`response_mask=mask[:,plen:]`；tool/advice在prompt侧。现有mask测试 [tests/test_core.py:61](/data/miyapeng/harness-internalization/tests/test_core.py:61)、[tests/test_migration.py:209](/data/miyapeng/harness-internalization/tests/test_migration.py:209)，tensor兼容测试 [tests/test_opid_bridge.py:57](/data/miyapeng/harness-internalization/tests/test_opid_bridge.py:57) | mask覆盖整个模型response（包括模型自己生成的think/action/EOS），不是仅XML action片段；真实veRL消费mask尚未验证 |
| 3.5 | 当前student权重采样，拒绝旧权重轨迹 | IMPLEMENTED | [training/trainer.py:87](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:87)–99；[training/verl_backend.py:79](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:79)–82 | 同一policy对象update后用于下一批；保存model_version并检查stale/harness错配。CPU mock连续更新2次；stale supplier测试 [tests/test_migration.py:196](/data/miyapeng/harness-internalization/tests/test_migration.py:196) | 默认单进程不存在另一个待同步rollout replica；多进程同步没有实现/验证，不能外推 |

### Stage 4 — Frozen Module Teacher（核心不一致）

| Stage | Requirement | Status | Code location | Evidence | Gap / Action |
| ----- | ----------- | ------ | ------------- | -------- | ------------ |
| 4.0 | 阶段teacher独立加载并冻结 | IMPLEMENTED | [training/entrypoint.py:54](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:54)；[training/trainer.py:72](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:72)–79；[training/teacher_backend.py:41](/data/miyapeng/harness-internalization/src/internalization/training/teacher_backend.py:41)、[training/teacher_backend.py:48](/data/miyapeng/harness-internalization/src/internalization/training/teacher_backend.py:48) | 单独HF实例，eval/no-grad，与student不alias；每批检查snapshot和frozen；CPU mock与runtime变更拒绝测试 [tests/test_migration.py:139](/data/miyapeng/harness-internalization/tests/test_migration.py:139)、[tests/test_core.py:49](/data/miyapeng/harness-internalization/tests/test_core.py:49) | 文件指纹为路径/大小/mtime，不是权重内容hash；实际HF冻结未运行 |
| 4.1 | 同时计算frozen teacher H+与H−两路logprob | INCORRECT / SEMANTIC MISMATCH | [training/teacher_scoring.py:29](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:29)–56 `ModuleTeacherScorer.score`；[training/module_advantage.py:41](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:41)–50 | scorer只创建一个H+ runtime，selected时只score一次；ModuleSignal只有一组log_probs。实际减数为old_student。本轮P4确认一次score和非零错误差分 | PARTIAL框架、错误核心量；P0增加成对teacher scoring及差分，不能将old_student重命名为teacher_minus |
| 4.2 | 两路使用完全相同冻结权重 | MISSING | [training/teacher_scoring.py:32](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:32)；[training/verl_backend.py:58](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:58)–75 | 只有单路module scorer。确有同一个frozen reference的额外KL打分，但没有连接为模块差分 | 不能把“存在KL reference”判为该要求已实现；详见后文KL辨析 |
| 4.3 | 对同一student-generated token序列评分 | PARTIAL | [training/teacher_scoring.py:47](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:47)；[training/rollout.py:52](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:52)、[training/teacher_backend.py:82](/data/miyapeng/harness-internalization/src/internalization/training/teacher_backend.py:82) | H+确实评分原始`step.response_ids`；old_student也评分这组IDs；本轮P4为[4,5] | 缺失H−教师分支，所以“双路同token”尚不成立 |
| 4.4 | 除target外H+/H−teacher输入一致 | MISSING | [training/teacher_scoring.py:42](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:42)；[harness/runtime.py:55](/data/miyapeng/harness-internalization/src/internalization/harness/runtime.py:55)；[training/verl_backend.py:69](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:69) | H+重算所有retained指导；old_student来自当前student生成的retained指导；KL reference也直接拿该student_prompt | 保留模块存在时，差异混入retained指导生成者变化。须独立构造或共享同一冻结模型的H−非target计算，并控制采样随机性 |
| 4.5 | 不读取未来obs/hidden answer/reference/学生不可见历史 | PARTIAL | [training/teacher_scoring.py:34](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:34)–47；[harness/runtime.py:53](/data/miyapeng/harness-internalization/src/internalization/harness/runtime.py:53)–71；[training/teacher_backend.py:64](/data/miyapeng/harness-internalization/src/internalization/training/teacher_backend.py:64)；[harness/module.py:78](/data/miyapeng/harness-internalization/src/internalization/harness/module.py:78) | 默认scorer仅使用当前step.state和response；不读trajectory.success/reward/未来state；prompt prefix/token校验，溢出报错不静默截断。兼容路径未来/历史隔离测试 [tests/test_opid_bridge.py:57](/data/miyapeng/harness-internalization/tests/test_opid_bridge.py:57)、[tests/test_opid_bridge.py:92](/data/miyapeng/harness-internalization/tests/test_opid_bridge.py:92) | 标准接线未发现reference solution/hidden-answer输入通道；但INSTRUCTION允许任意非空literal，禁止答案仅在proposer指令里，非语义检查。不能宣称所有候选绝无硬编码答案；H−成对信息对称仍未实现 |
| 4.6 | no-op模块应有近零H+−H−信号 | INCORRECT / SEMANTIC MISMATCH | [training/module_advantage.py:50](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:50)；[harness/runtime.py:69](/data/miyapeng/harness-internalization/src/internalization/harness/runtime.py:69)–71 | P4：同一个冻结mock在任意context均给−1，old_student给−2，当前信号+1，目标应为0；即使advice为空，runtime仍追加guidance标记 | 无no-op不变量测试；触发为False的模块因g=0而零，不是成对teacher差分验证。需真正context-preserving no-op定义与测试 |

### Stage 5 — GRPO outcome + Module-conditioned OPD

| Stage | Requirement | Status | Code location | Evidence | Gap / Action |
| ----- | ----------- | ------ | ------------- | -------- | ------------ |
| 5.1 | 保留真实任务outcome advantage | IMPLEMENTED | [benchmarks/alfworld.py:82](/data/miyapeng/harness-internalization/src/internalization/benchmarks/alfworld.py:82)；[core/trajectory.py:45](/data/miyapeng/harness-internalization/src/internalization/core/trajectory.py:45)；[training/trainer.py:43](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:43)；[training/module_advantage.py:22](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:22) | ALFWorld reward=10×won；episode reward求和，每个决策行附无效动作惩罚后组内normalize；优势fixture测试 [tests/test_migration.py:124](/data/miyapeng/harness-internalization/tests/test_migration.py:124)、环境mock [tests/test_migration.py:77](/data/miyapeng/harness-internalization/tests/test_migration.py:77) | 是按step行计权的OPID/GiGPO风格outcome变体，不宜未经说明声称严格trajectory均权GRPO；真实outcome训练未运行 |
| 5.2 | 模块term加到任务term，而非替代 | PARTIAL | [training/module_advantage.py:62](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:62) `combine_advantages` | 实际`task + module_weight*module`，有detach；数值fixture验证相加 | additive结构已实现；被加的量仍是teacher_plus−old_student，故目标A_ours没有实现 |
| 5.3 | 未触发/不在窗口时模块term严格为0 | IMPLEMENTED | [harness/runtime.py:57](/data/miyapeng/harness-internalization/src/internalization/harness/runtime.py:57)–62、[harness/runtime.py:80](/data/miyapeng/harness-internalization/src/internalization/harness/runtime.py:80)；[training/module_advantage.py:49](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:49) | 默认mode=targeted；g=target在active_modules；触发步s及s+1…s+PERSISTENCE均active。正常有限logprob下以bool mask乘零；本轮P4未selected结果[0,0]；窗口测试 [tests/test_core.py:33](/data/miyapeng/harness-internalization/tests/test_core.py:33) | 这是trigger/persistence gate，不是实测“修改了决策”的gate；NaN防护不由bool乘法保证，不能替代概率有限性检查 |
| 5.4 | 未意外启用hindsight/episode skill/analyzer | IMPLEMENTED | [training/entrypoint.py:52](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:52)；[training/trainer.py:79](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:79)；[training/teacher_scoring.py:149](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:149) | 生产入口仅ModuleTeacherScorer，无analyzer或episode skill调用。`step_skill_mask`仅TensorTeacherScorer兼容字段；现有迁移import检查 [tests/test_migration.py:54](/data/miyapeng/harness-internalization/tests/test_migration.py:54) | 不是仍在运行“文本skill analyzer”；错误在保留了旧logprob相减语义，二者应区别描述 |
| 5.5 | 实际lambda和module weight配置明确 | IMPLEMENTED | [training/module_advantage.py:13](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:13) `AdvantageConfig`；[training/trainer.py:62](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:62)；[training/entrypoint.py:54](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:54) | 默认module_weight=0.001；normalize_module=False；clip_module=None；task系数1；entrypoint未传自定义config，故这些默认值实际生效 | 不是从experiment_protocol.json加载的数值；CLI当前没有lambda覆盖参数；详见配置表 |
| 5.6 | advantage实际进入actor update | PARTIAL | [training/trainer.py:52](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:52)–56、[training/trainer.py:99](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:99)；[training/verl_backend.py:66](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:66)–79 | tensor_data含advantages，转DataProto后传engine.update_policy；CPU mock使用该tensor反传并改变参数 [tests/test_migration.py:166](/data/miyapeng/harness-internalization/tests/test_migration.py:166) | 接线与mock已实现；实际veRL actor未运行，不能宣称真实GPU loss/梯度接收正确；核心输入公式亦不符 |
| 5.7 | 教师/工具文本不会进入student response loss | IMPLEMENTED | [training/trainer.py:23](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:23)–56；[training/rollout.py:50](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:50)–62 | response仅generate返回IDs；指导在prompt；环境step在生成之后；兼容测试验证prompts/responses不变 [tests/test_opid_bridge.py:57](/data/miyapeng/harness-internalization/tests/test_opid_bridge.py:57) | 只声明项目batch构造与mock证据；真实veRL整链mask验证尚缺 |
| 5.8 | 阶段内teacher、target、harness、task allowlist固定 | IMPLEMENTED | [training/trainer.py:68](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:68)–101；[training/entrypoint.py:56](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:56)；[harness/module.py:49](/data/miyapeng/harness-internalization/src/internalization/harness/module.py:49)、[harness/module.py:106](/data/miyapeng/harness-internalization/src/internalization/harness/module.py:106) | 阶段teacher一次加载，target/harness不刷新；task tuple传入，scorer复制allowlist；每批拒绝非train/stale；frozen dataclass保存module/source。mock [tests/test_migration.py:139](/data/miyapeng/harness-internalization/tests/test_migration.py:139) | normal调用成立；生产路径没有逐批重读磁盘phase/manifest hash；旧`test_phase_cannot_change_midbatch`测试的是兼容TensorTeacherScorer，不应误当生产phase文件锁证据 |
| 5.9 | 更新后下一批rollout使用新权重 | IMPLEMENTED | [training/trainer.py:81](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:81)–99；[training/verl_backend.py:79](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:79)–82 | update后同student对象再次采样，snapshot_id递增，teacher不刷新；CPU mock连续2次更新和stale拒绝通过 | 仅默认单进程路径。分布式、vLLM权重广播不存在于此adapter，未验证 |

### Stage 6 — Post-training Four-cell Retirement Audit

| Stage | Requirement | Status | Code location | Evidence | Gap / Action |
| ----- | ----------- | ------ | ------------- | -------- | ------------ |
| 6.1 | 真正A/B/C/D四个条件各自执行、任务配对 | IMPLEMENTED | [outer_loop.py:62](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:62)–69；[evaluation/retirement.py:92](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:92) `PairedRetirementEvaluator.evaluate`；[training/rollout.py:41](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:41) | A/B训练前采集一次，C/D新checkpoint下各采集；before_cells重用已采集A/B，不是伪造C/D也不重复A/B；每次创建/重置环境。本轮P2/P3经过四格；配对测试 [tests/test_core.py:97](/data/miyapeng/harness-internalization/tests/test_core.py:97) | “独立执行”不等于四个统计独立样本，设计上要按同task/seed配对。真实环境4路执行尚未验证 |
| 6.2 | retirement任务不提供给proposer | IMPLEMENTED | [outer_loop.py:34](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:34)–49；[evolution/archive.py:21](/data/miyapeng/harness-internalization/src/internalization/evolution/archive.py:21)；[evolution/proposer.py:35](/data/miyapeng/harness-internalization/src/internalization/evolution/proposer.py:35) | ProposalRequest仅search IDs/traces/scores，直接allowlist检查；archive历史过滤dev/retirement。archive测试 [tests/test_migration.py:44](/data/miyapeng/harness-internalization/tests/test_migration.py:44) | 历史接受的模型/Harness自然反映先前决策，不等于向proposer直接泄漏retirement数据；扩展重复审计需另定数据复用协议 |
| 6.3 | test set不进入outer loop | IMPLEMENTED | [core/manifest.py:24](/data/miyapeng/harness-internalization/src/internalization/core/manifest.py:24)、[core/manifest.py:40](/data/miyapeng/harness-internalization/src/internalization/core/manifest.py:40)；[outer_loop.py:34](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:34) | 所有split无交叉，partition禁止test前缀，外层只取train/search/dev/retirement；测试 [tests/test_core.py:66](/data/miyapeng/harness-internalization/tests/test_core.py:66) | 依赖manifest真实ID声明；实际ALFWorld reset额外校验gamefile；泛化数据正确性尚未实跑 |
| 6.4 | bootstrap cluster是task ID，不是seed | IMPLEMENTED | [evaluation/retirement.py:34](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:34) `paired_interval` | 先按task把seed差值取均值，再对任务重采样；min_tasks=30；多seed单task不满足要求，测试 [tests/test_core.py:104](/data/miyapeng/harness-internalization/tests/test_core.py:104) | 统计口径正确；相同task内多个seed不提升独立task计数 |
| 6.5 | 退役性能条件A>B、D>B、D≈C、D≈A | IMPLEMENTED | [evaluation/retirement.py:57](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:57)–85 | gain.low>0；D−A、D−C的low≥−0.02；enough tasks；所有checks必须通过。正例/共同退化/冗余测试 [tests/test_core.py:83](/data/miyapeng/harness-internalization/tests/test_core.py:83) | 这些门槛只控制retire/retain，不控制训练前准入或模型rollback |
| 6.6 | 实际部署成本D优于C | PARTIAL | [evaluation/retirement.py:62](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:62)–74；[core/types.py:49](/data/miyapeng/harness-internalization/src/internalization/core/types.py:49)；[training/teacher_backend.py:63](/data/miyapeng/harness-internalization/src/internalization/training/teacher_backend.py:63)–98；[training/rollout.py:58](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:58)、[training/rollout.py:73](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:73) | 记录input/output token、模型/辅助/工具调用、latency；要求D对A、C均节约token超过5%、辅助调用下降、总模型/工具/latency不增。token反例测试 [tests/test_core.py:93](/data/miyapeng/harness-internalization/tests/test_core.py:93) | 实际HF计数与计时有代码，但现有demo成本合成；真实延迟/费用未验证。规则比只要求Cost(D)<Cost(C)更严格；加载时间不在episode latency内 |
| 6.7 | 能区分能力保留、成本改善、退役接受 | PARTIAL | [evaluation/retirement.py:75](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:75)–85 | `checks.preserves_original_capability`、`small_remaining_dependence`和各cost gate可逐项分辨；`decision`可判断retire是否接受 | 没有三个统一字段capability_preserved/cost_improved/retirement_accepted，也没有model_accepted/rollback字段。若只需分析可由checks派生，若要求稳定schema则需补全 |

### Stage 7 — Accept / Retain / Rollback

| Stage | Requirement | Status | Code location | Evidence | Gap / Action |
| ----- | ----------- | ------ | ------------- | -------- | ------------ |
| 7.A | 达标→接受新模型、retire目标模块 | IMPLEMENTED | [evaluation/retirement.py:83](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:83)；[outer_loop.py:73](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:73)–74 | decision=retire时new checkpoint＋reduced；正退役测试 [tests/test_core.py:83](/data/miyapeng/harness-internalization/tests/test_core.py:83)，已有toy运行结果仅为流程证据 | 只有retirement条件的接受规则；没有优先执行Case C的full-system退化审查 |
| 7.B | 未达退役但full系统未明显退化→接受新模型、retain | PARTIAL | [outer_loop.py:73](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:73)–74 | retain分支确实保留full并接受new；retained跨周期测试 [tests/test_outer_loop.py:13](/data/miyapeng/harness-internalization/tests/test_outer_loop.py:13) | 未验证“full未退化”这个必要前提；把B和C混成同一分支 |
| 7.C | C相对A明显退化→rollback旧模型、retain模块 | INCORRECT / SEMANTIC MISMATCH | [evaluation/retirement.py:57](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:57)–58；[outer_loop.py:73](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:73) | 不计算C−A interval；无rollback decision；`checkpoint=new_checkpoint`无条件执行。P3：A=1、C=0仍接受after。旧共同退化测试 [tests/test_core.py:85](/data/miyapeng/harness-internalization/tests/test_core.py:85) 只断言retain | P0：新增独立训练接受审查和旧checkpoint选择；epsilon/显著退化判定需预先定义，不能仅靠retirement失败 |

### Stage 8 — Continue Evolution

| Stage | Requirement | Status | Code location | Evidence | Gap / Action |
| ----- | ----------- | ------ | ------------- | -------- | ------------ |
| 8.1 | 下一轮沿用上一轮接受的模型和residual harness | PARTIAL | [outer_loop.py:43](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:43)、[outer_loop.py:73](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:73)–80 | retire后用new+旧H_t；retain后用new+H_t+m；正常残留模块测试 [tests/test_outer_loop.py:13](/data/miyapeng/harness-internalization/tests/test_outer_loop.py:13) | 只实现两种正常演进；由于没有model acceptance/rollback，可能沿用应被拒绝的模型 |
| 8.2 | retained旧模块在后续可重新接受retirement audit | MISSING | [outer_loop.py:57](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:57)–65、[outer_loop.py:71](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:71)–74 | 每轮target始终为本轮新module；没有遍历已保留模块，也不复用旧module的归因基准；旧模块保留后不会作为单独target再审计 | P1：设计再审计队列、所需参考模型/四格基准和独立任务预算；不是简单对旧module重跑C/D |
| 8.3 | archived/retired模块保留source/hash/parent/评价记录 | PARTIAL | [evolution/archive.py:12](/data/miyapeng/harness-internalization/src/internalization/evolution/archive.py:12)；[outer_loop.py:59](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:59)、[outer_loop.py:71](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:71)、[outer_loop.py:75](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:75) | source/module hash在deployment archive；parent/content hash/candidate ID在candidate_archive；四格score/gain在cycle文件。归档测试 [tests/test_migration.py:44](/data/miyapeng/harness-internalization/tests/test_migration.py:44) | 信息合计可恢复，但retirement archive未直接记录candidate_id/parent/评价路径/完整before-model身份；需跨文件按cycle/hash关联，未有统一完整链路测试 |
| 8.4 | retired模块不进入后续student runtime | IMPLEMENTED | [outer_loop.py:74](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:74)；[harness/module.py:118](/data/miyapeng/harness-internalization/src/internalization/harness/module.py:118)；[harness/runtime.py:55](/data/miyapeng/harness-internalization/src/internalization/harness/runtime.py:55)；[training/trainer.py:87](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:87) | runtime仅遍历active harness.modules，未读取archive自动启用模块；without返回新tuple；retained隔离测试 [tests/test_student_harness.py:12](/data/miyapeng/harness-internalization/tests/test_student_harness.py:12)和mock [tests/test_migration.py:139](/data/miyapeng/harness-internalization/tests/test_migration.py:139) | retired源码可作为proposer历史参考，这不等于进入student执行；未来显式重新提案是另一候选 |

## 当前实际训练公式

令 `u` 为本阶段内的更新批次，`i=(trajectory, decision step)` 为一个决策行，`j`为该行student response token。当前代码不是一个名为`compute_advantage`的函数，而是 `build_update_batch → combine_advantages → task_advantage/module_advantage`。

**outcome部分：**

\[
R_\tau=\sum_k r_{\tau,k},\qquad q_i=R_{\tau(i)}-0.1\,\mathbf1[\neg action\_valid_i].
\]

`q_i`仅写入该行最后一个有效response token，随后按行求和恢复成score。同一task/group的所有**决策行**参与均值和sample std，trajectory的长短会改变统计权重：

\[
A^{outcome}_{i,j}=M_{i,j}\frac{q_i-\mu_{G(i)}}{\sigma_{G(i)}+10^{-6}}.
\]

多行组 `mu=mean(q)`、`std(correction=1)`；单行组刻意使用`mu=0, sigma=1`。没有按trajectory先去重。定位：[training/trainer.py:35](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:35)–52，[training/module_advantage.py:22](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:22)–38。

因此outcome RL确实存在，但“GRPO”若指每个完整rollout等权的标准组归一化，需要说明或调整这里的step-weighted口径。本轮不擅自把用户尚未细化的GRPO口径判定为必须修改；将该区别明确列为协议决策项。

**模块部分及总优势（当前默认）：**

\[
\begin{aligned}
\ell^+_{i,j}&=\log p_{\bar\theta_t}(a_{i,j}\mid a_{i,<j},\mathrm{TeacherHarness}(s_i,H^+)),\\
\ell^{old}_{i,j}&=\log p_{\theta_{t,u}}(a_{i,j}\mid a_{i,<j},\mathrm{StudentHarness}(s_i,H^-)),\\
g_i&=\mathbf1[target\in guidance.active\_modules],\\
A^{current}_{i,j}&=\operatorname{stopgrad}\left[A^{outcome}_{i,j}+0.001\,g_iM_{i,j}(\ell^+_{i,j}-\ell^{old}_{i,j})\right].
\end{aligned}
\]

`old`指**当前采样批次、actor更新前**的student，不是整个阶段初始teacher。student更新后下一批old值也随之改变，teacher仍固定。当前实现是token-level dense advantage，覆盖整个student生成response；不是先对序列logprob求和再广播。`normalize_module=False`、`clip_module=None`时即上式；可选归一化是selected有效token的population std（correction=0），随后可选clip。定位：[training/rollout.py:54](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:54)–56，[training/module_advantage.py:41](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:41)–67。

**目标要求则是：**

\[
A^{target}_{i,j}=A^{GRPO}_{i,j}+\lambda g_iM_{i,j}\operatorname{stopgrad}(\ell^+_{i,j}-\ell^-_{i,j}),
\quad
\ell^-_{i,j}=\log p_{\bar\theta_t}(a_{i,j}\mid a_{i,<j},\mathrm{TeacherHarness}(s_i,H^-)).
\]

在能够定义相同H− context的条件下，现有信号与目标信号相差 `ell_minus_frozen − ell_old_student`；有retained模块时还会夹杂保留指导由不同权重重算的差异。第一批teacher/student同权重且H−指导完全相同时，两式可能暂时相等；这不构成整个阶段等价性。

### KL reference为何不是已经实现teacher_minus

[training/trainer.py:75](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:75)确实调用`student.set_reference(teacher)`。[training/verl_backend.py:69](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:69)–75也确实让同一个frozen reference对同一response IDs打分，保存为`ref_log_prob`。因此“全系统没有任何第二次冻结模型评分”是错误说法。

但该路径不满足目标：

1. 它发生在`combine_advantages`之后的`VerlPolicy.update`中，结果只作为KL reference传给actor，**没有参与module advantage相减**。
2. 它直接使用 `batch.metadata['student_prompts']`，没有用冻结teacher执行H−的retained模块；后续批次有retained指导时不等于所要求的teacher_minus context。
3. 它受KL loss配置控制，是额外正则项。`ref_log_prob`的存在不能修复`teacher_log_probs-old_log_probs`这条仍在执行的公式。

### g与no-op的准确边界

`PERSISTENCE=p`表示触发步及后续p步（总p+1步）处于active window；再次触发会刷新截止步。active时runtime基于当前公开历史重新生成指导，不把上一状态的teacher文本继续携带；持久状态仅有触发窗口。teacher运行时也不向环境执行draft/advice。

不触发的模块因g=0给零teacher term，是已实现的mask性质。一个始终active但语义上“什么也没做”的模块不同：runtime即使收到空advice也附加`[Internal ... guidance]`字符串，未定义context-preserving identity/no-op。因此需要分别验证“g=0→0”和“同冻结模型、同context无干预→difference≈0”；后者当前缺失，不能以前者代替。

## 实际生效配置与未接入配置

| 配置/运行量 | 实际值与来源 | 状态/说明 |
| --- | --- | --- |
| 外层轮数 / 候选数 / 总预算 / eval seeds | `LoopConfig`: 3 / 2 / 300 / (0,1,2)，[outer_loop.py:15](/data/miyapeng/harness-internalization/src/internalization/outer_loop.py:15) | CLI只覆盖total_train_steps；每轮预算=total/cycles |
| module weight / outcome系数 | 0.001 / 1，`AdvantageConfig.module_weight`与`combine_advantages`，[training/module_advantage.py:13](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:13)、[training/module_advantage.py:67](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:67) | 当前teacher term减数错误；值确实进入batch，不只用于metric |
| gating mode | `ModuleTeacherScorer(mode='targeted')`，[training/teacher_scoring.py:23](/data/miyapeng/harness-internalization/src/internalization/training/teacher_scoring.py:23)；trainer使用默认值，[training/trainer.py:79](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:79) | runtime支持all模式，但生产CLI未提供选择入口 |
| module normalize / clip | False / None，[training/module_advantage.py:16](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:16) | 无默认额外规范化 |
| outcome normalize / invalid penalty | mean_std_norm / 0.1，[training/module_advantage.py:15](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:15)、[training/module_advantage.py:18](/data/miyapeng/harness-internalization/src/internalization/training/module_advantage.py:18) | 按决策行组统计，并非trajectory均权 |
| train tasks / group rollouts / environment seed | 4 / 2 / 同任务两次seed均0，[training/trainer.py:63](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:63)、[training/trainer.py:83](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:83) | deterministic任务循环；replica ID区分重复，policy action随机采样 |
| action采样 / 上限 | train temperature1、top_p1、top_k0；eval和aux greedy；action512，aux192，[training/teacher_backend.py:30](/data/miyapeng/harness-internalization/src/internalization/training/teacher_backend.py:30)、[training/teacher_backend.py:63](/data/miyapeng/harness-internalization/src/internalization/training/teacher_backend.py:63) | 没有固定本阶段torch采样RNG的显式代码；环境seed不等于模型采样seed |
| PPO / KL / entropy / optimizer | clip0.2、dual clip3、KL low_var 0.01、entropy0.001；mini8/micro1/epoch1；AdamW lr1e-6 weight_decay0.01 grad_clip1，[training/verl_backend.py:16](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:16)、[training/verl_backend.py:41](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:41) | adapter配置存在，实际external veRL执行NOT YET VERIFIED |
| retirement标准 | margin0.02、confidence0.95、bootstrap2000、min_tasks30、token saving0.05、seed42，[evaluation/retirement.py:12](/data/miyapeng/harness-internalization/src/internalization/evaluation/retirement.py:12) | 总token要求upper CI严格小于负基线×5%；检查vs A与C |
| attribution标准 / rollback epsilon | 无独立配置或调用 | MISSING；不能假定retirement.margin已自动用于这两个缺失gate |
| experiment_protocol.json | [configs/experiment_protocol.json:7](/data/miyapeng/harness-internalization/configs/experiment_protocol.json:7)–20 | 描述性协议文件；`cli.main`只读取backend与manifest，[cli.py:66](/data/miyapeng/harness-internalization/src/internalization/cli.py:66)–69，未加载此JSON。当前重复默认值相同，不代表修改JSON会生效 |
| 预算单位 | 请求字段`optimizer_steps`，响应还含`training_batches_completed`，[training/entrypoint.py:59](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:59)–63 | 实际是采样/更新批次次数；一批内部可能多次AdamW.step，不可直接当优化器step等算力 |

## 成本、信息边界与验证限制

- `Cost`有input_tokens/output_tokens/model_calls/auxiliary_calls/tool_calls/latency_s；total_tokens为输入输出和，[core/types.py:49](/data/miyapeng/harness-internalization/src/internalization/core/types.py:49)。HF生成按tokenizer IDs计数，score消耗prompt+response输入token，score记作model/auxiliary call但不记生成output，[training/teacher_backend.py:78](/data/miyapeng/harness-internalization/src/internalization/training/teacher_backend.py:78)、[training/teacher_backend.py:96](/data/miyapeng/harness-internalization/src/internalization/training/teacher_backend.py:96)。
- runner逐步累加模块和action成本，每次真实`environment.step`计一次tool call。episode wall time包含reset/interaction/close，但模型加载发生在timer前；部署延迟采用何种冷热启动口径必须在真实实验报告明确。定位：[training/rollout.py:34](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:34)、[training/rollout.py:45](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:45)、[training/rollout.py:73](/data/miyapeng/harness-internalization/src/internalization/training/rollout.py:73)。
- `CommandBackend._call`保存process wall time和完整stage_cost；训练入口将阶段latency换成整阶段wall time。内部`training_summary`的累加latency不是完整优化器壁钟时间，PPO forward/backward token运算量也不由生成token计数覆盖。不要把同一次训练的process wall、stage_cost与逐步日志重复加总。定位：[command_backend.py:56](/data/miyapeng/harness-internalization/src/internalization/command_backend.py:56)–79，[training/entrypoint.py:60](/data/miyapeng/harness-internalization/src/internalization/training/entrypoint.py:60)，[training/trainer.py:102](/data/miyapeng/harness-internalization/src/internalization/training/trainer.py:102)–110。
- 默认state仅携带公共文本，teacher不读取reward/success或future observations作为输入。ALFWorld最近5步窗口对student和teacher相同；runtime保留触发截止步但不回填旧文本。显式token overflow报错，未发现静默截断后教师访问更长原始archive的路径。
- 然而模块instruction是任意字符串，触发器也能检查文字；AST限制保障不可执行任意I/O，不能证明没有任务答案硬编码。`APIProposer.CONTRACT`的“no answers”要求与自动语义审查不是一回事。此次未见默认初始模块携带reference solution，但不能据此给所有未来候选绝对保证。
- 默认实际环境入口只创建ALFWorld。WebShop/Search-QA文件仅为task ID/partition约束（[benchmarks/webshop.py:3](/data/miyapeng/harness-internalization/src/internalization/benchmarks/webshop.py:3)、[benchmarks/search_qa.py:3](/data/miyapeng/harness-internalization/src/internalization/benchmarks/search_qa.py:3)），AppWorld是未实现标记（[benchmarks/appworld.py:1](/data/miyapeng/harness-internalization/src/internalization/benchmarks/appworld.py:1)）。本轮未扩展它们，也不把预留目录算作实现。
- `VerlPolicy`依赖正常外部`verl==0.5.0`，默认单进程。存在world_size>1时直接要求注入另一个backend，[training/verl_backend.py:42](/data/miyapeng/harness-internalization/src/internalization/training/verl_backend.py:42)–48。adapter真实GPU调用、数值稳定性、mask消费、checkpoint再加载与跨轮teacher刷新均未经过真实模型验证。

## 本轮只读探针与现有测试的覆盖缺口

所有探针通过公开`Components`接口注入mock，调用未修改的`run_outer_loop`/scorer/advantage。外层探针使用1周期、2候选、30个互斥任务/partition、seed0、bootstrap100，原始模型名before、模拟训练返回after。A/B/C/D为mock分数，不是benchmark结果。

| 探针 | 可复现构造 | 实际输出 | 对目标的判断 |
| --- | --- | --- | --- |
| P1 无收益候选 | search/dev baseline和两个候选评分均0 | train_calls=0；checkpoint=before；active_modules=[] | 无收益跳过训练已实现 |
| P2 缺少训练前gate | search/dev上候选1、baseline0；attribution A=B=0.5；训练后C=D=0.7 | train_calls=1；checkpoint=after；decision=retain；module_was_useful=False；active_modules=[m0] | 冗余模块照常训练并保留；只有最终retire gate，缺少前置discard |
| P3 缺少rollback | search/dev正收益；A=1、B=0；训练后C=D=0 | train_calls=1；checkpoint=after；decision=retain；没有C−A interval | full系统明显退化仍接受新模型，Case C不成立 |
| P4 纯module effect应为0 | 单状态、response IDs=[4,5]；同一冻结mock在所有prompt上每token logprob=−1；student old=−2；active module、空advice | ModuleTeacherScorer只score一次；actual_module_signal=[1,1]；要求的冻结差分=[0,0]；空advice仍改变context | 当前信号混入teacher/student差异；没有no-op不变量 |
| P4附带mask检查 | 同上但selected=False | module term=[0,0] | g=0的屏蔽已实现，不能代替P4的差分不变量 |

核心反例公式可直接复算：

```python
# 使用现有 module_advantage；这里 old 的语义是student，不能当teacher_minus。
module_advantage(
    teacher_log_probs=torch.tensor([[-1., -1.]]),
    old_log_probs=torch.tensor([[-2., -2.]]),
    response_mask=torch.tensor([[1., 1.]]),
    selected=torch.tensor([True]),
)  # 实测 [[1., 1.]]；冻结teacher两种context均为-1时，目标结果为[[0., 0.]]。
```

32项测试通过与以上缺口并不矛盾：

- [tests/test_migration.py:124](/data/miyapeng/harness-internalization/tests/test_migration.py:124) `test_advantages_match_pre_migration_reference`验证的是保留的旧teacher−old公式，反而会固化与本轮新定义不同的量。
- [tests/test_core.py:89](/data/miyapeng/harness-internalization/tests/test_core.py:89)冗余模块测试只检查训练后的`module_was_useful=False`，没有断言`trainer.train`未调用。
- [tests/test_core.py:85](/data/miyapeng/harness-internalization/tests/test_core.py:85)共同退化测试只期望`retain`，没有检查旧checkpoint恢复。
- [tests/test_outer_loop.py:13](/data/miyapeng/harness-internalization/tests/test_outer_loop.py:13)验证旧模块留存并继续循环，没有测试旧模块后来重新退役。
- [tests/test_migration.py:139](/data/miyapeng/harness-internalization/tests/test_migration.py:139)的CPU trainer mock确实执行torch参数更新，但其loss是`-(weight*advantages).mean()`，不是实际veRL PPO。32项测试不能作为真实actor更新或GPU复现证据。
- 七个`OPIDBridgeTests`针对`TensorTeacherScorer`兼容路径，生产入口使用`ModuleTeacherScorer`；两者共享runtime不意味着所有旧phase文件锁测试覆盖了新的生产接口。

## A. 已完整实现

以下仅列有当前代码和已执行测试/探针支持的功能，不包括目标方法整体：

1. 多候选源码接口、受限可执行模块、search/dev配对选择、无收益不训练；proposer与archive可mock，P1验证无收益路径。
2. 单次连续外层运行保存并传递residual harness；retained模块继续运行，目标module可独立从runtime移除。证据：`OuterLoopTests`与`StudentHarnessTests`。
3. 同checkpoint、同task×seed的A/B采集，以及C/D的独立采集和配对检查；这只是采集能力，不包括Stage2准入和Stage7回滚。证据：P2/P3与配对异常测试。
4. student在H−访问自己的状态、精确response IDs与mask、teacher指导隔离、stale轨迹拒绝；CPU mock可连续更新参数并保存新checkpoint。
5. 真实环境reward到outcome advantage的代码接线、additive结构、trigger/persistence gating与padding屏蔽；只认可这些结构，不认可模块差分符合目标。
6. 生产路径移除hindsight/episode skill/analyzer；teacher阶段冻结、训练allowlist及公开状态输入约束有代码与mock层证据。
7. task-level bootstrap、四格能力条件和逐项成本门槛；冗余/共同退化/token增加/单task多seed等测试均覆盖最终退役判定。

## B. 已有框架但算法语义不一致

1. **核心模块监督量不同。** 当前为 `frozen H+ teacher − current rollout old student`；目标为 `same frozen teacher H+ − H−`。监督来源确实已换成executable module，但差分基准未改。KL reference没有参与纠正此量。
2. **归因目前是事后退役条件，非训练前准入。** 搜索候选虽通过search/dev，独立A/B不达标时仍训练，并以retain处理冗余module。
3. **retire/retain二分替代了accept/retain/rollback决策。** new_checkpoint无条件接受；full H+系统的C−A退化没有单独审查。
4. **outcome是step-weighted组归一化。** 如果最终论文使用标准trajectory均权GRPO表述，需要先明确两者的协议差别，不能仅凭变量名判断已一致。
5. **窗口mask与no-op消融不可互换。** 当前能验证g=0屏蔽，不能验证干预效应为零时frozen差分为零；空advice甚至仍改变输入context。
6. **toy连续退役不等于目标算法训练成功。** `ToyBackend.train`直接把teacher action写入表格，[demo.py:78](/data/miyapeng/harness-internalization/src/internalization/demo.py:78)–107；既不调用本项目优势函数，也不执行PPO或双教师差分。

## C. 还没有实现（后续行动，不在本轮修改）

| Priority | 未实现/未完成项 | 必要验收 |
| --- | --- | --- |
| P0 | 双路frozen-teacher模块评分与正确差分数据通路 | 同snapshot、同state、同response IDs、非target计算受控；ModuleSignal/UpdateBatch/advantage实际使用plus/minus；纯无效干预与teacher/student漂移测试；KL reference不能冒充模块minus |
| P0 | 前置A/B收益gate与discard | A−B未达预设统计阈值时train_calls=0，checkpoint不变，H_t不增加此module；记录attribution_failed，不能记内化 |
| P0 | 独立model acceptance与C−A rollback | 定义epsilon/显著退化标准；C显著低于A时恢复theta_t并保留m；下一轮接旧checkpoint＋H+；区分retirement decision和model decision |
| P0 验证门槛 | 目标公式修正后的真实veRL最小训练验证 | 当前代码有adapter但未验证，不能标MISSING optimizer。补齐依赖后确认优势/mask进入真实actor、teacher权重不变、学生更新后重新采样、checkpoint可重载；再开展论文GPU实验 |
| P1 | retained旧模块的后续再审计调度 | 新模型变强后能将旧模块设为审计target；为新旧module定义归因基准、任务独立性及重复检验预算；避免反复使用同held-out队列选择 |
| P1 | 统一算法配置的实际加载和快照 | 将lambda、监督模式、attribution/rollback参数与manifest/config hash记录并实际传递；测试改配置确实改变执行；目前experiment_protocol.json仅描述性 |
| P1 | 完整机制测试 | 在生产ModuleTeacherScorer路径测试双路信息对称、无未来/答案输入、no-op=0、A/B不达标禁训、C退化回滚、旧模块再退役；现有32测试不能替代 |
| P1 协议项 | 明确GRPO分组统计与模型采样随机性 | 确认采用step-weighted还是trajectory均权，分别给变长轨迹数值测试；记录模型RNG与环境seed，双路非target随机过程配对 |
| P2 | 完整生命周期归档schema | retirement条目直接链接candidate_id、parent、before/after checkpoint、A/B/C/D记录，归档runtime execution_failed原因；保留capability/cost/model/retirement独立判定 |
| P2 | 残留系统恢复运行入口 | 从已保存checkpoint＋residual harness恢复，而不是每次调用都Harness()；保持archive链路与继续运行预算 |
| P2 | 实际成本报告与候选数据质量补强 | 明确冷/热启动latency、训练优化器时间和各层cost去重；proposer前32步采样的成功/失败覆盖；防硬编码答案检查。不是本轮新增benchmark |

## 如果现在直接启动真实GPU实验，究竟测什么？

**明确回答：不是本轮定义的目标方法。** 当前审计解释器还缺transformers/verl/alfworld等依赖，直接启动先会遇到依赖问题；即使补齐依赖且adapter顺利运行，测到的仍是：

> 可执行Harness增强的冻结教师，对H−学生做teacher-plus减old-student的定向on-policy蒸馏，叠加step-weighted任务outcome优势和外部PPO/KL/entropy更新，再做只有retire/retain、没有模型rollback的外层循环。

它可以作为近似版本或基线，但不能以其结果声称验证“同一冻结模型H+/H−差分实现的module-conditioned effect”、训练前归因准入和训练退化回滚。本次32项通过及此前85个demo文件一致证明的是旧实现保持一致，**不能证明它已符合这次确定的新算法**。
