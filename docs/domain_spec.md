# 项目边界与约束

实现以 [core/interfaces.py](../src/internalization/core/interfaces.py) 的ProposerBackend、TaskRunner、TrainingBackend、RetirementEvaluator为工程边界，数据由core/types与trajectory定义。

- HarnessModule只分析同一公开历史；触发器为受限AST表达式，禁import/I/O/环境动作。Planner/Review/Recovery统一通过runtime生成内部指导，最终动作由runner唯一执行。
- Hcore保留原始公开历史、环境接口、解析与执行；不可蒸馏删除信息来源或权限检查。
- H+与target在外层阶段固定；每批H− rollout与H+评分共享同一个behavior policy，批内保持eval/no-grad且禁止权重变化，评分完毕后才update；校验snapshot、task allowlist、tokenization与响应mask。阶段冻结KL reference单独保留。
- source/hash/parent/version与所有候选成功失败归档；开发/撤除标签不提供给proposer。
- 训练前A/B须通过预先固定的task-cluster归因门槛；失败discard目标候选，不训练、不花费或重分配本轮训练预算，保留原checkpoint与residual Harness。判定独立归档，不进入proposer历史。
- 模型accept/rollback与模块retire/retain分别记录；C−A区间上界低于−ε时回滚旧模型并保留H+，独立任务不足拒绝新模型。下一轮使用正式采用的checkpoint，拒绝checkpoint与评价留档；回滚不抵消已花费训练预算。
- paired task×seed完整，退役使用独立cohort、四格能力/成本门槛；实验输出新建不覆盖。
- 不加载OPID/Meta-Harness私有类型或相对目录。外部veRL负责优化器与分布式能力；本项目不实现通用分布式训练框架。
- 外部ALFWorld提供环境。AppWorld由独立worker调用官方包，接口已有CPU/mock验证、真实环境尚未运行；WebShop/Search-QA完整adapter及TB2等扩展仍是未实现工作，不改变预注册主实验。

详细协议见 [METHOD.md](METHOD.md)，迁移前领域文档保存在 [history/domain_spec.md](history/domain_spec.md)。
