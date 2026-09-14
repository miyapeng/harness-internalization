# budget_v1 数据分布报告

状态：**NOT YET VERIFIED（官方数据未准备；以下是要求数量，不是已导入数量）**。

| Benchmark | train | search 池 / 每周期 | dev | retirement_0/1/2 | 最终评价来源 | 已实际导入 |
|---|---:|---:|---:|---:|---|---|
| ALFWorld | 2048 | 96 / 8 | 32 | 各128 | 官方 valid_seen / valid_unseen | 未执行 |
| WebShop | 2048 | 96 / 8 | 32 | 各128 | 官方 test ID 0–499 | 未执行 |
| HotpotQA | 2048 | 96 / 8 | 32 | 各128 | 官方 distractor dev | 未执行 |

内部总计每 benchmark 2560 题，没有 acceptance 分区。search 跨三个周期使用其中24个不同 task ID，其余72题不自动进入其他用途。

来源约束：ALFWorld 内部全部 official train；WebShop train/search 来自 official train（session≥1500），dev/retirement 来自 official eval（500–1499）；HotpotQA 内部全来自 official train，绝不切换 fullwiki。

导入器由真实来源生成 task_type、source_hash 和 group_id：ALFWorld 按任务/对象/场景目录归并 trials；WebShop 按目标商品 ASIN 归并指令实例；HotpotQA 按规范化 question 与 supporting document identity 归并重编号副本。这些离线划分元信息和隐藏评分数据不传给 proposer/Agent。任务类型分层后，固定 seed 排序并以完整 group 做精确数量选择；如果无法精确满足，则报错，不拆 group、借测试或默默减少数量。

最终集合完整保留；内部池中与最终实例相同的别名明确隔离并列入 excluded_internal_aliases_of_final_instances。WebShop 先分配官方 eval 的 dev/retirement，再为 train/search 分配未被使用的同源商品 group，跨官方池的未选别名不再进入其他用途。分布报告记录所有选中任务 ID、实际 source/type/group 数、未使用内部数量、排除 ID、manifest hash 和 split seed。这些是预先定义的数据隔离操作，不依靠模型分数选择数据。

本地检查只发现其他项目的 HotpotQA simplified 文件，没有把它们当作用户要求的官方完整 train/distractor dev。未找到完整 ALFWorld/WebShop 数据，没有换用简化来源来凑数。正式 `distribution.json` 将在成功运行导入器后生成；资源不足时不伪造这一结果。

合成 CPU 回归验证了精确 2048/96/32/128×3、group 不跨分区和 WebShop 官方 ID 范围。合成数据验证不代表上述官方数据已经满足独立 group 容量；真实可用数量仍需导入时检查。命令见 [运行说明](BUDGET_V1.md)。
