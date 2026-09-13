# 迁移前依赖审计（2026-09-13）

本表在修改实现之前生成。扫描 src、scripts、configs、tests、README、lock 与 patch；扫描包含 import、subprocess、sys.path、字符串路径和补丁触点，排除嵌套仓库本体、Git 元数据、pycache 与历史运行输出。

| 当前功能 / 位置 | 实际依赖 | 迁移目标 | 语义与许可证处理 |
|---|---|---|---|
| scripts/meta_proposer.py | 标准库 HTTP；无 Meta-Harness import | evolution/proposer.py | 本项目自写代码；保留模型参数和有界代码契约；接口增加历史候选、score、typed trajectories |
| outer_loop.py 候选生成/评分/选择 | 本项目混合 ExperimentBackend；无上游类型 | evolution/search.py、candidate.py、archive.py | 两候选、search/dev 正增益门槛、原排序与 tie-break 不变；归档失败/成功及 parent/hash |
| records.py、manifest.py | 标准库 | core/types.py、trajectory.py、manifest.py、interfaces.py | 保留 State/Cost/结果序列化和 task×seed 协议 |
| harness_modules.py、teacher_harness.py | 本项目代码 | harness/module.py、runtime.py、planner.py、reviewer.py、recovery.py | 不改触发解释器、三类调用顺序、持续窗口与 hash |
| opid_adapter.py | 被 OPID trainer patch 调用；读取 OPID config 和 batch 字段 | training/teacher_scoring.py | 可独立注入教师；保留原20测试所需的薄兼容 alias；移除 analyzer/episode-skill 生产路径 |
| student_harness.py | verl.protocol；collector.build_text_prompt_batch；私有 hi_max_tokens patch | training/rollout.py 与独立模型接口 | H− 使用当前学生；主动作才进入环境；不再依赖 patched vLLM 元数据 |
| OPID gigpo/core_gigpo.py 实际启用部分 | step teacher advantage + episode_norm_reward；step outcome 权重0，episode skill 权重0 | training/module_advantage.py | 自写机制实现；固定 teacherLP-oldLP、mask、std correction=1、按step加权；保存上游数值 fixture 作回归 |
| scripts/launch_opid_alfworld.sh | cd upstream/OPID；python -m verl.trainer.main_ppo；OPID Hydra 字段 | training/trainer.py + 本项目启动脚本 | 普通外部 veRL 负责优化/分布式，项目负责 rollout/scoring/advantage；无新分布式框架 |
| scripts/train_opid.py | 调用旧 launcher 与 OPID scripts/model_merger.py | training/trainer.py、checkpoint.py | 阶段冻结、预算、保存新 checkpoint；使用正常 veRL worker 保存与包内 merger，不复制 merger |
| alfworld_eval.py、evaluate_alfworld.py | agent_system manager/projection/嵌套 AlfredTWEnv；sys.path 插入仓库 | benchmarks/alfworld.py、training/rollout.py | ALFWorld 为正常外部包；仅保留提示模板/动作解析/历史格式适配，保留相应 Apache attribution；不复制环境 |
| ALFWorld patch | real task ID、训练 game_files allowlist | benchmarks/alfworld.py | 显式 real task 选择，不用 parquet 行号 |
| WebShop/Search patch | session ID/范围修正、task_id 透传 | benchmarks/webshop.py、search_qa.py | 保留任务约束验证；真实 adapter 原来未完成，继续明确未运行 |
| retirement_eval.py | 本项目配对统计 | evaluation/retirement.py、cost.py | 四格、同tasks/seeds、bootstrap、性能/成本门槛完全不变 |
| configs/alfworld_backend.json | 旧脚本 argv | 新独立脚本 argv | 文件协议保留，本项目接口隔离 subprocess |
| upstream.lock.json、patches/opid-integration.patch、install_opid_patch.py | 固定嵌套目录、6个patch文件 | THIRD_PARTY_NOTICES.md、第三方来源清单 | 先保存来源/许可和测试基准，全部迁移验收后删除指定四项 |
| README/RUNBOOK/STATUS 与历史记录 | 旧启动/路径说明 | 当前文档改写；历史放 docs/history | 历史扫描命中仅允许在历史文档中；不把旧记录伪装成新状态 |

## 迁移前确认的算法细节

- 三轮，每轮两个候选；search/dev 严格正增益；候选按 dev 均值增益、负总 token 排序。
- A/B/C/D、每轮独立撤除队列、task-level bootstrap 与固定门槛不变。
- student 按 H− 的输入生成响应；teacher 从相同公开原始历史重算 H+。
- 当前 reward normalization 的默认分组统计是跨交互 step，不是每条 trajectory 只取一次；样本 std 使用 correction=1。
- 模块 token signal = detach(teacher_log_prob − old_log_prob) × response_mask × module_mask；默认权重0.001，不归一化、不裁剪。
- 原 launch 启用无效动作惩罚0.1、PPO clipping、KL loss0.01。迁移需保留而非悄悄换成 SFT。
- Meta-Harness 仓库只是参考，没有直接 runtime import；不迁移其 benchmark/reference_examples。
- 原20测试与CPU demo 在迁移前重新运行并记录；GPU/Ray/vLLM 从未在本项目验证。

## 删除条件

原20项测试、独立 proposer/trainer mock、数值优势 parity、demo 逐文件语义对比和不依赖两个上游的运行检查通过后，执行用户明确指定的四项删除。保留其原始许可证和迁移映射。旧测试文件保持不变，必要的历史 import 使用本项目内 alias。
