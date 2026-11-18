# A 组 C 阶段配置与偏离

本次用作者训练集做一次完整的 4B Tutor 更新，并走一次末尾验证。此目录中的检查点用于核对训练、验证和导出。B 阶段的临时学习率、缩小的批量、四次后测和跳过验证仅用于 B。正式训练等待单独确认。

## 逐项核对

| 项目 | 作者论文或公开源码 | C 生效配置及影响 |
|---|---|---|
| 数据 | 作者公开数据集、作者代码的题对筛选与格式 | 沿用 A 组已冻结的作者题池。原始 train 共 10,000 行，其中作者筛选后有 4,830 行。隔离与测试/验证重合的原题后，训练为 4,712 行、4,615 个不同原题，验证为 100 个不同原题。预检复核正式 parquet 哈希、行数、选题清单哈希和数据版本。 |
| 模型和训练算法 | 公开主配置默认载入 27B，论文报告 4B。DPPO，组内优势，token-mean | 路径绑定本地 Qwen3.5-4B。冻结本地 Llama-3.1-8B-Instruct 和本地 Qwen3.6-27B。保留原有奖励、思考剥离、失败投票降级和学生答案规则判分。 |
| 每步采样数 | 论文附录写 16 个 prompt × 16 条轨迹。源码默认 rollout.n=8，train_batch_size=16 | 显式设为 16 × 16，共 256 条。16 指输入题对数，训练池允许同一原题有多个作者题对。 |
| 参数更新 | ppo_mini_batch_size=8，lr=1e-6 | 恢复两项作者值。 |
| 教师生成 | 训练温度 1.15。可见与思考预算分别为 150、512 | 训练入口在 veRL 迁移后向作者旧配置名恢复两个预算字段，使教师实际读取 662 单轮上限。上下文剩余长度也可能限制生成。提示长上限 2,048、会话响应预算 8,192、模型总预算 10,240。 |
| 学生训练后测 | 温度 0.6，每道通过门槛的轨迹独立采样 16 次 | 后测数与 rollout.n 分别核对。被硬门槛拦下的轨迹按作者实现跳过最终学生作答。 |
| 本地门槛 | 两个标准各三票，裁判温度 0.6 | 同一规则用于训练和末尾验证。256 条完整训练轨迹对应 1,536 张名义票，100 条验证轨迹对应 600 张名义票。空轨迹、重试等会使 HTTP 次数变化。按实测分别报告。 |
| 末尾验证 | 作者 test_freq=25、val_before_train=false。验证每题 1 段对话，温度 0.6、top_p=0.95 | 保留正频率。veRL 的末步分支即使没到第 25 步也会验证。C 使用作者留出的 100 道验证题，这一环节与发布测试集成绩独立。检查实际记录是否为 100 条。 |
| 外部服务 | 公开配置启用 W&B，并在验证中另外调用 Gemini-3.1-Pro 评教学质量，不参加奖励计算 | trainer.logger 只保留 console，W&B 禁用。eval_judge_model 传空串，作者函数在构造 Gemini 客户端前返回空指标。训练和验证所用的学生、门槛端点都复核为容器内 127.0.0.1。Gemini 分数应视为缺测。 |
| 总步数和保存 | 源码 total_epochs=5，save_freq=25，max_actor_ckpt_to_keep=5 | 显式传 trainer.total_training_steps=1。源码中显式步数优先。按当前物化训练池、每步 16 行且无额外长提示过滤估算，默认 5 轮约 1,470 步。C 沿用 save_freq=25，末步自动保存 global_step_1。未来正式训练的覆盖参数是 trainer.total_training_steps=100。 |
| 设备和环境 | 论文报告 GH200，作者 Dockerfile 固定 veRL v0.7.1 | 单机 A100，2 张 Tutor（rollout TP2、FSDP2），1 张 Qwen 裁判（TP1），1 张学生（TP1）。三种服务沿用既有 BF16 路径，vLLM 使用本机候选 0.19.1。GPU 并行布局、软件版本会影响吞吐及数值。 |
| 显存和并发 | Tutor vLLM 整体显存预算 0.58，free_cache_engine=true。作者 judge/student 每进程信号量默认 128/256 | Tutor 沿用作者预算，KV 动态分配。保留 B 的 2 个 agent worker、2 个 reward worker。裁判每个 reward 进程上限 16（名义合计 32）。学生每个 agent/reward 进程上限 16（名义合计 64）。等待发生在 HTTP 请求之前。 |
| 技术失败与报告 | 作者重试耗尽的门槛票记 REJECT | 沿用既有外部监控。观察到 fallback 就停止并标为无效。10 秒检查周期可能跨过一次更新。C 的更新仅用于功能和耗时检查。训练段与末尾验证分开计时，保存时间属于训练 step。 |
| 教师统计 | 作者保留思考/可见单轮共用上限和尾轮存储行为 | 用已落盘教师消息重分词近似统计达到 662 的回合，按作者去思考函数统计可见文本为空的对话。无法还原未落盘回合或原始 finish_reason，不作精确截断率解读。 |
| 结束与导出 | 作者脚本以 FSDP 分片保存，提供 CPU 合并工具 | 使用已修复退出分类的控制器，记录各角色退出顺序、退出码和合并工具实测结果。C 自动合并本步模型权重。正式多检查点保留策略和恢复测试待 C 后确定。 |

## 生成上限的入口适配

veRL v0.7.1 的 main_ppo.main 先运行 migrate_legacy_reward_impl。该函数把奖励配置迁到 reward.custom_reward_function，并删除顶层 custom_reward_function。作者 EduardoAgentLoop 仍从顶层读取教师预算，找不到时回退到 400 与 400。奖励端正常读取 150 与 512。CPU 测试用上述两个版本的实际函数与教师生成方法，复现原生代码传给下游的 max_tokens=800。仅看 Hydra 配置或用重分词统计大于等于 662 的回合，均无法证实生成的实际硬上限。B 的实机回执没有精确采样上限，不能据此改写 B 的实测数字。

C 的独立入口沿用 veRL 的设备选择、配置迁移及 run_ppo。在迁移完成后，仅将 max_tokens_per_turn 和 max_thinking_tokens_per_turn 两个数值提供给作者读取的旧命名空间。两处有冲突时中止。公开版本中，奖励函数加载器从新命名空间读取完整奖励配置。CPU 测试核对适配前后奖励配置相同，教师实际生成方法传出的上限从 800 变为 662。容器内会在占卡前用已安装的框架执行同一配置迁移预检。本改动以论文规定的预算为基准。它与作者公开版本直接组合 veRL v0.7.1 的行为存在差异。来源见 [veRL 迁移函数](https://github.com/verl-project/verl/blob/v0.7.1/verl/experimental/reward_loop/reward_loop.py#L57-L89) 和 [作者教师循环](https://github.com/eth-lre/EduardoRL/blob/e8c89b7293e65e632149571b16a1c2d4e33c77fe/eduardo/eduardo_agent_loop.py#L37-L66)。

训练策略以 [作者 eduardo_base.yaml](https://github.com/eth-lre/EduardoRL/blob/e8c89b7293e65e632149571b16a1c2d4e33c77fe/eduardo/configs/eduardo_base.yaml) 与 [user.yaml](https://github.com/eth-lre/EduardoRL/blob/e8c89b7293e65e632149571b16a1c2d4e33c77fe/eduardo/configs/user.yaml) 合成。每轮生成上限见 [EduardoAgentLoop](https://github.com/eth-lre/EduardoRL/blob/e8c89b7293e65e632149571b16a1c2d4e33c77fe/eduardo/eduardo_agent_loop.py#L37-L66)，额外评审开关与门槛见 [reward_function.py](https://github.com/eth-lre/EduardoRL/blob/e8c89b7293e65e632149571b16a1c2d4e33c77fe/eduardo/reward_function.py#L470-L500)。训练超参数和轨迹数见 [论文附录](https://arxiv.org/html/2610.06446#A1)。

末步保存和验证的代码路径见 [veRL v0.7.1 ray_trainer.py](https://github.com/verl-project/verl/blob/v0.7.1/verl/trainer/ppo/ray_trainer.py#L1517-L1552)。题池来源版本为 [作者数据集](https://huggingface.co/datasets/dmacjam/eduardoRL-transfer-dataset/tree/5d7529f06359f31b692c92634f5cc11c8b7953af)。
