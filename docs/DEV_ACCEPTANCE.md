# 复用 dev 接受 Harness

本轮仅取消独立 Harness acceptance 数据集和重复评价。训练、具名内化目标、A/B contribution gate、C/D retirement、模型 rollback 的算法与门槛保持原样。

## 执行路径

```text
固定当前 checkpoint + H_parent
  → parent search/dev 各一次
  → 两个候选，各执行 search/dev
  → 复用选中候选的 dev 配对结果正式接受 H_plus
  → 可选目标与兼容性检查
  → 原 A/B → 训练 → 原 C/D → retire / retain / rollback
```

原 search/dev 接受标准不变：`search_gain.low > 0 AND dev_gain.low > 0`。均为原 `paired_interval` 按 task ID 聚类的配对 bootstrap 区间；置信度、bootstrap 次数及 seed 由同一个原 RetirementPolicy 提供。不改成只看均分，不改变现有候选排序（dev 平均收益优先，token 成本作 tie-break）。这里沿用原 dev 规则，不把原先额外 acceptance cohort 上的 AttributionPolicy 伪装成 dev 规则；A/B 的 AttributionPolicy 本身不变。

不再调用 `acceptance_plus` 或 `acceptance_parent`。两个正常候选时，接受前为 **6 次**批量评价（parent 2 次 + candidates 4 次），原来通过 search/dev 后还需额外 2 次。实际节约的 episode 数取决于旧 acceptance cohort 大小和 seeds；30题×1 seed 的合成验收少60个episode。兼容性检查、A/B、训练rollout、C/D仍按原分支执行、照常计费。

## 接口和证据

`evolution/revision_search.py::search_revisions` 现在返回 `RevisionSelection | None`：

- candidate、parent（可执行 revision）；
- checkpoint，以及可从轨迹验证的 model_version（不提供轨迹的 mock runner 为 null，不伪造权重指纹）；
- parent_dev、candidate_dev 逐题 EpisodeResult，含 task ID、seed、success、cost；
- search_gain、dev_gain 配对区间；
- dev_accepted、search_accepted 与联合 accepted；
- 候选 index、明确接受规则与 bootstrap 参数。

若没有有效可评候选返回 None；若候选可评价但均未通过，返回最高排名的**拒绝证据**，accepted=False，不能据此接受。异常候选仍隔离记录、继续评价兄弟候选。不同实际模型 snapshot 混入 search/dev 会报错。

外层在 `cycle_XX/harness_acceptance.json` 保存上述证据，标记 `decision_source="dev"`、`reused_evaluations=true`、`additional_evaluation_calls=0`，引用该 cycle 下原 `baseline_dev/episodes.json` 和 `candidate_N/dev/episodes.json`。逐题结果是既有证据的副本，不是新的调用；原子进程成本账本只记录实际 stage，不为接受记录再追加 stage_cost。候选目录另存 dev_acceptance.json 用于审核被拒绝候选。

dev 未通过保留旧模型和父 Harness，reason=no_useful_candidate；dev 通过即接受 H_plus。无目标、不兼容或 A/B 不通过继续保留 H_plus、不训练。训练退化仍回滚旧模型并保留 H_plus；通过退役才部署 H_minus。

## 数据与协议

新导入通过 `loop_cohort_names` 分配 train/search/dev、retirement_0..N-1 与 test，不再创建 acceptance 分区。通用导入最低训练源数量是 `(3+cycles)*cohort_size`，三周期、30题/cohort时为180。ALFWorld 使用相同共享列表；AppWorld 不再单独追加 acceptance cohorts，继续保持 scenario 三变体整组。既有 manifest **不重新导入、不重切**；其 acceptance 分区可保留不用，既有 retirement/test ID 和 manifest hash 原样保持。

启动仍检查所有 retirement cohort 及原任务数门槛，最终测试集不进入 outer loop。proposer 输入仍只有 search 轨迹/分数和允许的历史；不会提供 retirement/test 的轨迹或逐题分数，也不新增 dev 逐题结果到 proposer 输入。

新 protocol 的 harness_acceptance 标记来源 dev，并保存原 search/dev 规则。旧独立接受协议续跑时，只对未完成周期采用本轮授权的新流程，在**新输出**中记录 acceptance_transition 和原 protocol hash；旧文件不改写，已完成周期不重复，retirement cohort 不重排。其他 loop、A/B、retirement、执行配置身份检查继续执行。

## 修改与验证

| 范围 | 修改 |
| --- | --- |
| `evolution/revision_search.py` | 返回带门槛判定和完整 dev 证据的 RevisionSelection |
| `revision_loop.py` | 复用证据正式接受，去掉两次独立评价；下游训练/审计分支不变 |
| `core/accepted_state.py` | 校验新接受规则，保留旧协议的可追溯续跑 |
| `core/manifest.py`, `benchmarks/appworld_manifest.py` | 新分区与启动要求移除 acceptance；加载旧分区不变 |
| `revision_demo.py`, manifest CLI 说明 | 新示例不创建 acceptance；更新生成器元数据与帮助 |
| `tests/test_dev_acceptance.py` | 7项新测试：接受/拒绝、非正区间、成本复用、旧分区闲置、目标不兼容、A/B失败、协议迁移 |
| 既有测试 | 更新返回类型与新分区/接受来源的必要断言；保留训练和退役用例 |

命令保持原样，使用既有 manifest 即可；新导入的 manifest 无须 acceptance。测试：

```bash
PYTHONPATH=src python3.12 -m unittest discover -s tests -p 'test_dev_acceptance.py' -v
PYTHONPATH=src python3.12 -m unittest discover -s tests -v
```

完整结果见 `validation/dev-acceptance-report.json`。合成评分与 mock policy 不构成真实 GPU 或官方 benchmark 验证。
