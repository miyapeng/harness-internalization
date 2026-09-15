# 具名控制与选择性内化

版本化 Harness 可以保留多个控制，本轮只关闭明确指定的 ID。ID 是行为标识，不要求属于 planning/review/recovery，也不代表代码已经被自动分解为原子能力。模型与控制源码的版本仍由 checkpoint/revision hash 标识。

## 配置与目标

```json
{
  "schema": 2,
  "entrypoint": "agent/main.py:run",
  "composition": "independent_suffix",
  "controls": [
    {"id": "recovery_v1", "entrypoint": "controls/recovery.py:run", "enabled": true},
    {"id": "review_v1", "entrypoint": "controls/review.py:run", "enabled": true}
  ]
}
```

列表顺序就是部署时的组合顺序。ID 必须唯一；控制只返回 `suffix: str`、`selected: bool`，未触发时 suffix 必须为空。不同版本修改了同一 ID 的源码仍是不同 revision，不应只凭相同 ID 宣称行为没有变化。

同一次候选响应的 internalization 字段为（完整响应见 [结构化提案协议](STRUCTURED_PROPOSALS.md)）：

```json
{"internalization":{"target_control_id":"review_v1","removed_behavior":"旁路额外 review，保留 recovery 和日志工具"}}
```

也可以填入 `"internalization":null`，与可内化候选同等有效；没有第二次目标 API。宿主 `InternalizationTarget.from_control()` 从 full revision 查找实际入口，只将该 ID 的 enabled 改为 false。其他控制、顺序、入口、工具、prompt 和源码不变。产物仍保存 full/reduced revision、hash、target_control_id、removed_behavior、宿主解析的 supervision_adapter。不存在、已关闭或非独立组合的声明不能获得内化资格；合法 H+ 降为 harness-only 并保留声明错误记录。

此例 H+ 开启 recovery/review，H− 只开启 recovery。A−B 测量的是已有 recovery 条件下 review 的增量贡献，允许两种能力有协同作用；不把它描述为脱离其他控制的独立因果效应。

## 部署与评分使用同一个组合协议

`independent_suffix` 的规则：

1. prepare 生成基础上下文和工具描述。
2. 每个开启的控制都读取同一份基础上下文和 step；通过独立隔离进程运行，不传其他控制的输出或可变记忆。
3. 按注册顺序把输出拼接到基础上下文，生成实际学生 prompt。
4. H− rollout 记录 `ControlContext(base_context, composition, outputs)`，每项输出含 control_id、entrypoint、suffix、selected。
5. 教师先验证记录可逐字复现学生 prompt，并核对 H− 的控制 ID、顺序和入口。随后只在该基础上下文执行目标控制，按 H+ 的注册位置插入目标结果。其他控制的实际输出直接复用，包括随机生成的内容。

因此撤除排在首位的控制时，评分不会把它追加到末尾。非目标输出不重新生成；无响应记录、组合元数据缺失/不一致时明确拒绝，不回退为简单追加字符串。基础上下文必须是实际学生 prompt 的组成部分，不得补回截断历史。原 response IDs、old_log_prob、行为策略快照和 loss mask 检查保留。

独立控制仅允许内部模型计算，不能调用环境；真实工具仍由正常 prepare/execute 路径经 broker 执行。schema 2 的候选入口和控制不能读取 `config/harness.json`，由 Landlock 文件读权限隔离，避免根据开关配置重建不同基础上下文或非目标指导。工具注册配置、源码和其他共享文件继续可读；未放宽原进程、网络或文件写入限制。

依赖前面控制输出的 `sequential_suffix` 仍可作为完整 Harness 运行，后续控制获得累积上下文。它不能进入当前训练桥：候选构造时记录 internalization_declaration_error，保留已验证的 H+ 和旧模型，不训练。需要任意环境交互的控制可以使用该运行路径；不将其自动转成独立指导。

这些是受保护接口、组合协议和实际运行检查，不是任意 Python 程序等价性证明。不要通过路径/时钟等环境细节推断启用状态，或在控制内部隐藏跨控制依赖后声称支持独立内化。检查覆盖真实访问的状态，不能保证所有未来状态；运行异常仍按既有流程保留 H+、停止内化。没有自动依赖分析器或可内化性分类器。

实际环境事件和学生决策另行记录；在 prepare 或合法顺序控制中终止任务也不会丢掉环境奖励。事件记录不是新的学生 response，也不进入增强评分上下文。见 [环境事件与回报](ENVIRONMENT_EVENTS.md)。

训练公式保持 `A_outcome + lambda * g_target * (logp_old(enhanced,response_ids) - old_log_prob)`；g 只由目标控制决定，非目标控制触发不会代替目标 mask。两侧同一个 batch 的 old policy，评分完成后才更新 actor。

## 兼容与接口迁移

| 位置 | 变化 |
| --- | --- |
| `harness/revision.py` | 增加 schema 2、具名目标与确定性单 ID 关闭；保留 schema 1/旧目标序列化 |
| `evolution/code_proposer.py` | 一次提案内置可选具名目标；hash 与 H− 仍由宿主计算 |
| `harness/control_runtime.py` | 独立/顺序组合、目标执行、非目标输出复用和位置校验 |
| `harness/code_runtime.py` | 根据 schema 选择旧单 hook 或具名控制路径 |
| `harness/sandbox.py`、`sandbox_worker.py` | schema 2 文件读权限收紧，屏蔽开关配置 |
| `core/trajectory.py`、`serialization.py` | 新 `RevisionTransition/ControlContext/ControlOutput`；旧 Transition JSON 不添加字段 |
| `training/revision_rollout.py`、`revision_scoring.py` | 保存实际组合，按部署顺序增强评分；旧桥保留 |

原 schema 1 的可执行 `supervision` string/null 在状态加载、运行和合成演示中继续支持；旧三类模板生产入口已删除，旧 revision hash 和归档不重写。没有自动把旧总 hook 拆成多个 ID；可以由后续候选显式升级配置，通过原 search/dev 收益规则并复用 dev 正式接受后生效。状态加载、实验隔离、归因门槛和 accept/retain/rollback 规则均沿用现有实现。

## 运行与证据

```bash
cd /data/miyapeng/harness-internalization
PYTHONPATH=src python3.12 scripts/demo_named_controls.py \
  --output runs/named-controls-example
PYTHONPATH=src python3.12 -m unittest discover -s tests -p 'test_named_control*.py' -v
```

输出目录必须尚不存在。工作区是 `examples/named_controls/`；默认目标为 review_v1，也可传 `--target-control-id recovery_v1` 验证撤除第一个控制。示例保存实际 full/reduced 快照、目标、学生轨迹、组合记录、教师评分及 full 运行记录。脚本模型/环境是合成的，实际 optimizer updates=0；成功率不是学习或官方任务成绩。

已运行示例产物：`runs/named-controls-proof/`。学生实际调用日志查询工具，保留 recovery，只对 review 构造增强评分。两周期测试使用 mock 分数与 checkpoint 转换、真实原统计门槛；CPU 训练测试另行执行两批真实 SGD，验证同批同步、no-op、inactive mask、梯度隔离和共享上下文。真实 API proposer、HF/veRL/GPU 与官方 benchmark 未运行。完整回归记录见 `validation/named-controls-report.json`。
