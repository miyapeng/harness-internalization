# 工作记录

## 2026-09-13：初始工程搭建

用户要求：独立文件夹，按 OPID＋Meta-Harness＋模块蒸馏与撤除思路先搭代码并记录。本次先完成 CPU 可验证的工程；受当前环境限制没有启动 GPU 实验，工程 smoke test 不代表方法有效性。

完成事项：

1. 检查工作区，无适用于新项目的上级 AGENTS.md；创建 `/data/miyapeng/harness-internalization` 和独立 Git 仓库，没有修改原有研究项目。
2. Git 默认代理 DNS 失败、提权代理连接返回 403；去除该命令的代理环境后直连成功。已下载 OPID 和 Meta-Harness，固定两个 commit。
3. 核验真实 trainer、rollout、prompting、环境 ID、官方脚本。确认原 OPID 教师评分会使用当前 actor；新增独立冻结教师。记录 WebShop 训练区间与材料不一致、AppWorld manager 已存在两处差异。
4. 实现有界可执行触发代码、三类辅助流程、学生公开输入隔离、冻结教师、OPID 原生打分桥接、任务 manifest、配对四格撤除与成本条件。
5. 实现默认三轮/两候选外层、独立进程协议、阶段/checkpoint/hash、候选轨迹和成本归档；补充 API proposer、ALFWorld evaluator、训练与合并入口。真实运行未启动。
6. 写出可幂等验证的 OPID 补丁安装器，保护未知本地修改；已应用到新 clone，完整 diff 保存在 patches 中。
7. 完成 CPU 合成三周期闭环。该演示使用确定性候选、学生访问状态上的表格更新、合成 token/时延。三轮退役仅说明流程可运行，不能作为实验主表。
8. 初始 16 项测试通过后，补充 retained-module rollout：保留模块使用当前 actor，冻结教师从共享原始历史重算完整指导；新增学生隔离、公共历史不可隐藏和保留后继续周期测试。最终完整输出见 validation/tests-final.txt。没有 Ray/vLLM/GPU 数值训练测试。
9. 最终 20 项测试全部通过；23 个项目 Python 文件语法检查、配置 JSON 解析、shell 语法、OPID 补丁幂等检查和 reverse apply 检查均通过。补充方法规格、domain_spec、运行说明、基线清单、来源、已知限制和检查记录。

## 验证产物

- `runs/cpu-smoke-001/`：初次完整 CPU 运行轨迹、各轮四格、退役结果和部署归档。
- `runs/cpu-smoke-002/`：增加每轮 state.json 后的完整 CPU 运行；后续保留分支由独立回归测试覆盖。
- `docs/validation/`：最终测试输出、补丁与脚本检查、合成运行摘要及环境信息。
- `upstream.lock.json`：上游 URL、commit 与补丁位置。
- `patches/opid-integration.patch`：本次 OPID 完整修改。

## 下一工作项

1. 在可用 GPU 环境准备真实 Qwen2.5-3B checkpoint 与 ALFWorld 数据，先用 1 个模块、少量任务做真实 forward/rollout/更新检查；核验 token 打分与 teacher 冻结。
2. 真实验证已实现的 OPID 非空 H− 路径，核验同步 vLLM 辅助调用 token 上限、配对行为与成本。
3. 完善真实 CPU/GPU 成本记账和任务/seed 配对，固定正式预算与撤除统计协议。
4. 接 WebShop 完整协议和 Search-QA 真实 ID＋检索固定，再补直接基线与文本技能对照。

未完成事项的详细边界见 STATUS.md。没有提交远端 fork/PR，没有安装其他项目的训练依赖，没有使用已有 GPU 任务资源。
