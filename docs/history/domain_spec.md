# Domain Spec：可执行控制模块内化

## Domain Summary

任务单位为完整交互 episode。固定环境观察、工具权限和统一动作接口。搜索规划、审查、恢复控制，蒸馏后以真实成功率与成本决定退役。初始模型沿用 Qwen2.5-3B-Instruct；具体本地 checkpoint 未指定。默认三周期、每轮两候选、300 optimizer steps 均分；模型/搜索 token、GPU 小时预算 unknown，等待首轮测量后固定。

## Harness and Search Plan

候选接口为 literal metadata + `trigger(history, step)`，由有界解释器执行。runtime 统一调用冻结模型，候选无环境、I/O 和任意 Python 执行能力。先在 search 轨迹上提案，再 train/search 复测和 dev 筛选，最终用单独 retirement 队列判定。复用模块加载、历史隔离、成本账本、任务 manifest、配对统计。

基线包括裸模型、固定三类模块、原 OPID、固定 Harness 蒸馏、文本技能、SLIM。最强 Harness unknown，不能预设演化模块优于文本技能。

## Evaluation Plan

ALFWorld 为首项，每个 task×seed 配对。最终 seen/unseen 不做选择。WebShop 和 Search-QA 待后续环境适配验收；复杂任务再扩展 AppWorld。记录成功率、总输入/输出 token、辅助/总模型调用、工具调用和延迟。噪声、单候选耗时和最低显存 unknown。默认配对任务 bootstrap；预留不同周期的独立撤除集。

## Experience and Logging

当前没有真实离线轨迹。首次 baseline_search 采集学生失败，proposer 只获得公开 search 轨迹。记录实际任务 ID、seed、当步公开输入、动作、成功/奖励、教师专用指导、源码 hash、冻结 checkpoint 指纹、训练步数、成本、每轮判定。候选和阶段写入独立目录，不覆盖旧实验。

## Open Questions and Unknowns

真实 GPU 资源、首个 checkpoint、ALFWorld 数据路径、proposer endpoint、最终训练预算 unknown。它们不阻碍 CPU 工程搭建；真实调用前由运行配置提供。用户已有详细规格，因此直接将 unknown 写入记录并继续实现，没有再次进行上游 onboarding 问答。
