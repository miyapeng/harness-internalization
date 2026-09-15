# 当前运行入口

从仓库根目录执行。正式参数、数据准备、四个阶段入口、两批真实 smoke 和最终评价命令统一以 [BUDGET_V1.md](BUDGET_V1.md) 为准；其他 benchmark 的专属权限与评价说明保留在 [AppWorld](benchmarks/APPWORLD.md) 和 [适配说明](benchmarks/ADAPTERS.md)。本轮未执行真实 API、官方环境或 GPU 训练。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3.12 -m unittest discover -s tests -v
python3.12 -m internalization.cli --help
python3.12 scripts/demo_versioned_harness.py --config configs/versioned_demo.json --output runs/new-versioned-demo
python3.12 scripts/demo_named_controls.py --output runs/new-named-demo
```

每次使用不存在的 output 目录。CPU 演示实际运行候选工具和隔离进程，模型与训练为 mock，不能代表真实学习效果。需要 Linux Landlock/seccomp；缺少隔离直接报错。

正式 `run` 必须显式给出代码 workspace、revision 或版本化 accepted state，不能隐式启动旧模块循环。状态加载与最终评价见 [ACCEPTED_AGENT_PIPELINE.md](ACCEPTED_AGENT_PIPELINE.md)。阶段内 request/response 评价和独立 final 评价保留各自数据权限，不互相替代。

保留的阶段入口是 `scripts/propose.py`、`scripts/train.py`、`scripts/evaluate_alfworld.py` 与 `python3.12 -m internalization.training.entrypoint`。retirement 可通过 `python3.12 -m internalization.cli retirement --help` 查看离线四格输入协议。它不触发演化或训练。

旧固定三类模板 CLI、phase/tensor 桥及历史 hash/demo 验证器已退役。历史报告按原内容保留，不能将其中命令当作当前运行说明。清理依赖表、测试迁移和结果见 [CLEANUP_REPORT.md](CLEANUP_REPORT.md)。
