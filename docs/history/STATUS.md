# 实现状态与限制

## 已实现且 CPU 验证

| 部分 | 已完成内容 | 验证 |
|---|---|---|
| 模块 | 规划、draft-review、恢复流程；有界 Python 触发解释器；源码 hash | 加载、非法代码拒绝 |
| 教师 | 学生当前公开输入、指导隔离、冻结身份、持续窗口、无环境句柄 | 防未来信息/指导泄漏与刷新测试 |
| OPID 接口 | 原生 step teacher 分数、选择掩码、学生 prompt token 一致性、错误传播 | 使用真实 CPU torch tensor 的桥接测试 |
| 保留模块 | H− 模块调用当前 actor；H+ 教师从相同原始历史重算；目标模块不进入学生 | 学生上下文隔离、继续后续周期测试 |
| 撤除 | 四格、task×seed 配对、任务 cluster bootstrap、成本门槛 | 联合退化、冗余、成本反升、伪重复等反例 |
| 外层 | 三周期、两候选、search/dev 选择、阶段固定、退役/保留、逐周期状态与归档 | 合成完整闭环 |
| 工程 | 独立进程协议、成本账本、真实任务 manifest、不可覆盖输出 | CPU 命令与补丁匹配检查 |

## 已写代码但尚未真实运行

- `FrozenHFBackend` 加载真实 HF 模型、生成与打分；当前测试未验证真实模型数值，也未跑 vLLM/Ray。
- ALFWorld evaluator、OPID 阶段启动、FSDP checkpoint 合并和完整 CommandBackend 调度。
- API proposer；未配置或调用外部模型。Meta-Harness 上游固定保存为参考，当前为其提案方式的本项目适配器，不是直接运行原仓搜索脚本。
- ALFWorld task manifest 生成与训练 game_files 过滤；尚未下载环境数据验证原生路径。
- WebShop 默认训练区间修正及真实 session ID 记录；Search-QA 原始 task_id 透传。

## 第一版明确未完成

1. **OPID 非空 H− 路径已补齐，但未做真实 Ray/vLLM 验证。** 保留模块在 rollout 内调用当前 actor，目标模块不执行；阶段冻结教师从相同原始历史重算完整指导。该入口目前限定同步 vLLM（与固定 OPID 的 0.11.0 / WebShop 0.8.2 入口对应），其他 rollout 后端尚未适配。
2. WebShop 的完整 dev/retirement 子集采样与独立环境评价、Search-QA 的 dataset ID/manifest 生产和检索服务固定未完成；不宣称三个真实主实验已接通。
3. 检查到当前 OPID 已有 `AppWorldEnvironmentManager`，与用户提供材料中的“需新建 Adapter”有差异；这里只记录已有入口，尚未确认它符合 AppWorld 完整协议，不能视作已验证适配。
4. 基线列入实验协议但未复现；all-step 监督已可配置，文本化技能、SLIM/OPHSD 及完整消融 runner 待接。
5. 搜索只演化触发条件、提示与持续窗口；流程本身是固定的三类模板。若要主张复杂程序决策迁移，需要扩大代码搜索空间及相应机制对照。
6. 自动恢复中断任务未实现。输出拒绝覆盖，每轮 `state.json`、phase、checkpoint、原始日志足够人工核查；不要直接复用旧输出目录重新运行。
7. teacher checkpoint 指纹使用路径内文件名/大小/mtime 元数据，不是全量权重 SHA256；正式发布需额外记录权重内容校验和。阶段内内存模型冻结不依赖该文件指纹。
8. 当前 HF scorer 串行、未做吞吐优化。真实 GPU 资源、采样温度协议、统计功效、多重比较与跨训练 seed 复验预算待首轮测量后固定。

## 当前环境

2026-09-13：默认 Python 3.14.6；使用 `/usr/bin/python3.12` 验证。Python 3.12 可用 CPU torch/numpy/pytest，缺少 transformers/ray/alfworld。`nvidia-smi` 无法与 NVIDIA 驱动通信。没有安装大型训练依赖、下载模型/检索索引或提交 GPU 任务。

这些限制阻碍真实复现，不影响已经记录的 CPU 工程验证。不能把 synthetic demo 的退役结果用作算法有效性证明。
