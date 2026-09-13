# Harness Internalization

把可执行的规划、审查、恢复模块提供的决策能力定向蒸馏到主模型，通过四格撤除检验决定模块退役。

这是按研究方案搭建的第一版工程。**20 项测试通过，CPU 三周期演示已跑通；真实 OPID/GPU 训练与 benchmark 结果尚未复现。** 演示使用表格学习与合成成本，不能作为论文能力证据。

## 当前目录

```text
src/internalization/
  harness_modules.py    可执行 Python 触发条件的受限解释器、模块版本
  teacher_harness.py    同状态规划 / draft-review / 恢复；教师指导隔离
  student_harness.py    保留模块使用当前 OPID actor，目标模块关闭
  opid_adapter.py       接入原生 OPID step teacher advantage
  backends.py           独立、冻结的 Hugging Face 教师与 token 打分
  retirement_eval.py    A/B/C/D 配对任务 bootstrap 与成本门槛
  outer_loop.py         候选→验证→定向训练→撤除→归档，默认三轮
  command_backend.py    proposer / evaluator / trainer 独立进程文件协议
  alfworld_eval.py      复用 OPID 提示词、历史、解析器的 ALFWorld 评价入口
  manifest.py           真实任务 ID、互斥划分、测试集禁入外层选择
  demo.py               CPU 合成集成演示
harness_modules/       planner.py / review.py / recovery.py 初始模块
upstream/OPID/          本地训练主仓，已应用有记录的补丁
upstream/meta-harness/  固定版本参考；不合并其训练环境
scripts/               补丁、任务划分、提案、训练、评价脚本
configs/               初始实验协议和 ALFWorld 进程配置
docs/                  方法、实现状态、来源、运行说明和工作记录
runs/cpu-smoke-002/    已运行演示的轨迹、四格结果和逐周期退役归档
```

## 立即验证

无需 GPU，也不安装 OPID 的重型依赖：

```bash
cd /data/miyapeng/harness-internalization
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3.12 -m unittest discover -s tests -v
python3.12 -m internalization.cli validate-module harness_modules/recovery.py
python3.12 -m internalization.cli demo --output runs/my-cpu-smoke
python3.12 scripts/install_opid_patch.py --check
```

输出目录必须是新目录，避免覆盖旧实验。可选 `python -m pip install -e .` 后使用 `hi` 命令。核心无第三方依赖；tensor bridge 测试有 CPU torch 时运行，否则明确 skip。

## 真实实验入口

先读 [运行说明](docs/RUNBOOK.md) 与 [当前实现状态](docs/STATUS.md)。ALFWorld 提供了提案、评价、阶段训练和 checkpoint 合并入口，但依赖 GPU、模型与数据；当前还未端到端验证。

OPID 上游代码已固定 commit，修改保存在 [OPID 补丁](patches/opid-integration.patch)，可用 `scripts/install_opid_patch.py` 重放。不会把其他项目的 `verl/` 合入。项目是本地 Git 工程和 OPID checkout，尚未创建远端 GitHub fork。

第一版搜索空间为**代码触发条件＋三类固定控制流程＋持续步数＋指导提示**，并非任意 Python 控制程序搜索。Meta-Harness 的外层提案方式已通过独立进程适配；`scripts/meta_proposer.py` 使用可配置 API，当前没有实际调用 proposer，也没有宣称重现其原论文搜索。

完整研究规格见 [METHOD.md](docs/METHOD.md)，进度与已知限制见 [WORKLOG.md](docs/WORKLOG.md)。
