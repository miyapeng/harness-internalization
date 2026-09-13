# Benchmark接入顺序

AppWorld之后，按用户追加要求接入TB2、SWE-bench Pro、HotpotQA和LawBench。当前是代码与CPU/mock验收，真实环境仍需依赖和数据。

| 顺序 | Benchmark | 当前状态 | 下一步 |
| --- | --- | --- | --- |
| 1 | AppWorld | adapter/manifest/评价/训练入口已实现，13项CPU/mock测试通过；真实环境NOT YET VERIFIED | 安装固定worker依赖和官方数据，运行无模型smoke，再真实模型评价与训练 |
| 2 | Terminal-Bench 2 | Harbor Trial/自有BaseAgent、manifest、训练/评价入口已实现；真实容器未运行 | 安装Harbor 0.23.0和Docker，核验真实task/verifier |
| 3 | SWE-bench Pro | 容器工作区、patch和官方Pro grader适配已实现；mock通过 | 固定官方checkout、镜像digest和测试资源，跑真实smoke |
| 4 | HotpotQA | distractor固定context检索与联合指标已实现；真实自有worker＋CPU更新通过 | 官方数据与官方评分脚本逐样本parity；fullwiki未接入 |
| 5 | LawBench | 20题型官方dispatcher、native最终评价与可评分单例训练入口已实现；mock通过 | 安装官方评分/ChERRANT依赖；运行全量原生最终评价 |

保留ALFWorld已有adapter。WebShop/Search-QA仍只有任务身份/划分约束，本轮未改。历史experiment_protocol.json主实验组合不自动覆盖；新增AppWorld入口是独立可选配置。本表是用户指定的工程排期，不声称这些benchmark全部出现在Meta-Harness原论文中。

AppWorld实现与运行说明见 [benchmarks/APPWORLD.md](benchmarks/APPWORLD.md)。新增四项的配置、划分与限制见 [benchmarks/ADAPTERS.md](benchmarks/ADAPTERS.md)。TB2/Pro/LawBench默认只导入test，不自动挪作训练数据。当前100项测试通过，其中本轮新增29项；GPU、真实官方环境均NOT YET VERIFIED。
