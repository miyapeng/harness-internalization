# 来源与核验

核验日期：2026-09-13。以下为源码检查，没有 GPU 复现。

| 来源 | 用途 | 核验结果 |
|---|---|---|
| [OPID](https://github.com/jinyangwu/OPID) | 训练主仓 | 已 clone，固定 `37a15a5f3c0f1ecc651e4be4a0c257b313fa0756` |
| [Meta-Harness](https://github.com/stanford-iris-lab/meta-harness) | 独立外层提案方式 | 已 clone，固定 `0cbc31e97c9e6d24232d1dc754827c02e1ec415c` |
| [OPHSD](https://github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation) | 蒸馏直接近邻与基线 | 本次未 clone/复现，不把用户提供判断当作本次实测 |
| [SLIM](https://github.com/ejhshen/SLIM) | 生命周期直接基线 | 本次未 clone/复现 |

OPID 实际接入点：

- `verl/trainer/ppo/ray_trainer.py::_prepare_opid_teacher_signals` 原来分析已完成轨迹，并在 `_compute_skill_log_probs` 用 `actor_rollout_wg.compute_log_prob` 评分。本项目在方法入口新增独立冻结教师 provider，旁路 hindsight analyzer。
- `gigpo/core_gigpo.py::compute_opid_outcome_advantage` 已接收 step teacher log probs/mask。本项目保留该优化路径，没有替换 RL 算法。
- `agent_system/multi_turn_rollout/rollout_loop.py` 已记录 obs_text 和 step_num，补充真实 task ID 与 rollout 成本；异步 snapshot 补齐 prompts、task_id、step_num。
- 保留模块通过原 actor worker 生成辅助指导。`vllm_rollout_spmd.py` 新增 `hi_max_tokens` 元数据读取，在 greedy 分支之后限制辅助生成 token 数，保证不会被默认 sampling 参数覆盖。
- 默认 ALFWorld 脚本使用左截断，本项目改为 error 并逐 token 比对学生实际 prompt。教师不能读左截断之前的隐藏信息。
- WebShop 源码原本 `is_train=True` 使用 `range(500, len(goals))`；与用户指定的原始 train/dev/test 协议不一致。本项目改为 `range(1500, len(goals))`。完整 dev 子集采样仍待实现。
- 当前 OPID 已含 AppWorld manager，后续应先核验再决定扩展方式。

Meta-Harness README 说明更换 proposer 需适配 wrapper，并保存提案交互记录。本项目实现独立文件请求/响应与 API proposer，使用有界候选解释器；没有直接复制其 Claude wrapper 的权限跳过参数，也没有运行其原有文本分类/Terminal-Bench 实验。

上游许可保留在各自 checkout。根目录记录 lock 和补丁，未整体复制 OPHSD/SLIM/WHALE 的 verl。
