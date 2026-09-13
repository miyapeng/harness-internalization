# 来源与固定版本

代码来源、原版权与逐文件迁移记录统一放在 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。来源commit记录在 `upstream.lock.json`，不再作为运行目录配置。

- OPID：`jinyangwu/OPID`，commit `37a15a5f3c0f1ecc651e4be4a0c257b313fa0756`，训练机制参考。
- Meta-Harness：`stanford-iris-lab/meta-harness`，commit `0cbc31e97c9e6d24232d1dc754827c02e1ec415c`，外层代码提案/search参考，无源代码复制。
- 外部veRL：标准PyPI包 `verl==0.5.0`，API参考 [v0.5.0 PPO actor](https://github.com/volcengine/verl/blob/v0.5.0/verl/workers/actor/dp_actor.py)。未执行实际训练。
- ALFWorld：[官方项目](https://github.com/alfworld/alfworld)，正常外部环境包，未复制环境实现。本次没有安装或运行实际数据。

OPHSD、SLIM、WHALE等仍是待研究/复现的比较对象，未作为当前运行依赖，也未新增实验结果。迁移前完整研究来源说明保存在 [history/SOURCES.md](history/SOURCES.md)，其中旧工程路径和实现判断属于历史状态。
