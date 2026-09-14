# 环境回报与学生决策分开记录

版本化候选可以在 prepare、内部控制或 execute 中执行真实工具。因此事件发生的时间不能决定奖励是否被计入，学生是否生成 response 才决定是否存在 actor loss。

## 数据与回报

版本化 rollout 返回 `EventTrajectory`，是旧 `Trajectory` 的兼容子类：

| 字段 | 内容 |
| --- | --- |
| initial_observation | reset 返回的合法公开观察；reset 接口没有 reward，不伪造奖励 |
| environment_events | 每次已返回的环境调用：序号、step、phase、control_id、action、公开 observation、reward、done、success、action_valid、tool_calls |
| public_calls | broker model/environment 调用的公开参数、返回结果与成本；另记录没有环境调用的本地工具 dispatch |
| transitions | 只有实际学生决策：prompt、response IDs、old log-prob、原有效性字段；具名控制继续保存组合元数据 |

`EventTrajectory.total_reward = sum(event.reward)`。transition.reward 保留供旧接口和诊断读取，但不会再加到总回报上；显式空事件列表代表零环境奖励，不回退到 transition.reward。旧模块运行路径和旧 `Trajectory` JSON 仍按原规则读取，不重写历史文件。过去没有事件记录且已经漏记的奖励不能凭旧 transition 恢复，相关轨迹需要重新采集。

事件的 step 表示当前准备/执行所在的决策位置，不代表一定存在对应 response。例如已产生一个学生决策后，下一次 prepare 完成任务，终局事件的 step=1，但 transitions 仍只有 step=0。

例子：首次 execute 获得 0.5，下一次 prepare 获得终局 3.0。轨迹 return 为 3.5，actor batch 仍只有之前那条真实 response；不会增加第二行 token，也不会把奖励重复算为 4.0。

## 训练处理

保留现有任务优势和蒸馏公式。`build_update_batch` 原本读取 trajectory.total_reward，现在版本化轨迹的这一属性来自完整事件账本。仍按原协议把 episode return 放到每条真实响应末 token，并保留原 invalid-action penalty 和组内归一化方式。

* 混合 batch：无学生决策的 episode 保留结果/成本，既不产生 actor 行，也不把它的 reward 转移给另一个 episode。没有响应行的 episode 不进入现有按决策行构造的优势归一化。
* 整个 batch 无决策：跳过教师评分与 actor update，记录 batch_skipped/no_student_decisions、各 episode return、成功与成本。消耗本次已分配 rollout 批次，不另补训练预算；下一 batch 策略快照不漂移。
* checkpoint 的 step 使用实际 actor update 次数。summary 分别记录已处理 batch 数、actor_update_calls、batches_skipped_no_student_decisions。全部跳过时只保存未更新权重，不宣称发生了学习。

若 prepare 已终止环境，runtime 不再生成控制指导或学生动作。评分路径仍禁止新的环境交互；事件/公开调用日志不会被拼到 teacher context 或 response loss 中。具名控制的实际上下文复用、同批 old-policy、mask 和梯度隔离机制不变。

## 公开审计与 proposer

broker 每次环境返回时立即写入 environment_event，并写 capability 的 parameters/result/cost；因此候选拿到奖励后再抛异常，已发生的环境事实仍可在失败审计中检查。失败仍按原流程拒绝评价结果，不凭局部成功日志强行判任务通过。

只使用明确的公开字段：环境 action、adapter 已提供的 observation/done/action_valid，以及用于任务训练的 reward/success；模型调用只记录提交的 prompt 和返回文本。本地 dispatch 记录学生 action 与公开 observation/stop，不序列化环境对象、原始 evaluator state、隐藏答案、额外参考解或整个 memory 对象。

CodeProposer 的 search trace 增加 initial_observation、environment_events、public_calls 和 total_reward。即使 episode 没有任何主模型动作，proposer 也能看到 prepare 执行了什么。仍先检查任务 allowlist；acceptance/retirement/test 不因此获得提案可见性。目标选择接口只接收原公开源码/历史，没有把完整事件日志用作教师特权输入。

prepare 调过环境、execute 只执行本地工具的情况，成本现在同时包含前者的实际工具次数和后者的一次本地 dispatch；不会因为 broker 已有 public observation 就漏计后者。

## 验证

```bash
cd /data/miyapeng/harness-internalization
PYTHONPATH=src python3.12 -m unittest discover -s tests -p 'test_environment_events.py' -v
```

新增11项覆盖首个 prepare 终止、已有决策后的 prepare 终止、多次环境返回不重算、纯本地调用、空事件账本、序列化、异常后审计、proposer 可见性、全零决策 batch、混合 batch 和跳过后的策略同步。

持久化例子在 `runs/environment-events-proof/`：首个 prepare return=3、student_decisions=0；下一 prepare 终止例 return=3.5、只保留两个真实 response token；另保存两批 tool-only 训练的零 actor update summary。使用 scripted policy/environment 和真实隔离执行；CPU 小模型训练测试是实际 SGD，GPU/真实模型/官方任务未运行。机器记录见 `validation/environment-events-report.json`。
