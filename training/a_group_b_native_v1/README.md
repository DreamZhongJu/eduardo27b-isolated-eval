# A 组 B 阶段：原生配置功能测试

仅供经用户审阅后在指定单机 A100 容器运行。该版本不注入或修改作者代码和 veRL；工作目录必须全新。固定作者 e8c89b7293e65e632149571b16a1c2d4e33c77fe、veRL 0.7.1 与候选 vLLM 0.19.1。目标容器运行库、GPU、Ray 与模型训练尚未验证。

B 共两步，每步两道题、每题四条轨迹，学生后测四次。仅 B 把学习率设为 1e-5，每步保存，跳过末次验证。作者的教师温度 1.15、单轮 512+150=662 总上限、奖励与耗尽回退行为保持原样。C 不使用此启动器，需恢复论文口径和验证设置。偏离在每次运行目录 `deviations.md` 创建。

固定 2 个 agent-loop worker 与 2 个 reward worker，裁判每 reward 进程 16（理论客户端合计 32），学生对话与后测每进程 16（理论合计 64），服务 max-num-seqs 64。学生 interaction 的现有配置复制到本次目录后修改，作者源码不变。日志只反映能观察到的 semaphore 和进程；取消后服务端残留计算以及 worker 重启可能使瞬时服务负载不同，不能称作全局硬上限。

裁判和学生共享调用使用作者原有进程内 semaphore 和重试代码。每 10 秒扫描本次 Ray 会话和训练日志中三类已知最终失败：`gave up ... using fallback`、`non-retriable ... using fallback`、学生交互 hard failure；发现即停并标无效。轮询不能保证优化器尚未更新，且无法证明捕获无日志的全部故障；B 的 checkpoint 不作为论文结果。

GPU 按原 A 自检脚本 SHA 验证并复用锁和 parent-death；新资源层协调 2 Tutor / 1 judge / 1 student，凑卡超时释放、退避、按角色交接，超限释放本任务资源。GPU 快照非硬独占，运行中用户进程可竞态；不能证明 PID 归属时显存峰值标未知。`CUDA_VISIBLE_DEVICES` 若已设置将拒绝运行，不扩大继承的可见设备范围。

运行后用本目录 `status.py --work <run_dir>` 看脱敏状态表；用 `audit_posthoc.py --work <run_dir>` 汇总。两步的 rollout 与 actor 概率差及相关系数可提示异常，不构成全权重同步证明。作者记录没有原始 finish_reason、未追加尾轮和可靠逐步轨迹身份，因此这些只能报告未覆盖。原样 step 与 testing 计时平级；checkpoint 保存计入 step。

CPU 本地测试：`test_b_stage.py` 15、`test_status.py` 3、`test_resources.py` 4、`test_gpu_process_monitor.py` 4、`test_parent_guard.py` 1，均通过；原等待器测试 4 项、事后汇总测试 9 项通过。用户执行的 A 阶段已验证单卡强杀清理与锁，不能替代此 B 控制器在真实容器中的验证。

`b25fe23` 版把有效符号链接误判为缺失。当前版本允许指向实际文件的符号链接，并检查 Python 可执行权限；执行时始终使用虚拟环境原路径。失效链接、缺失文件、不可执行解释器依然拒绝；后续代码版本、数据散列和模型结构校验保持不变。

`d2daa4d` 版按字符串比较 GPU UUID，未兼容 NVIDIA 带 `GPU-` 前缀与 PyTorch 不带前缀的两种形式。当前版本对两者严格解析完整 128 位 UUID 后按顺序比较；无效 UUID、MIG、缺失/多余和真实不同 UUID 仍拒绝。容器里的实际 UUID 映射需运行时验证。

`9f4b434` 的 Hydra 预检只解析训练主配置。Ray worker 加载学生交互 YAML 时还会解析模型名中的 `${oc.env:USER}`。B 启动器现在用同一模型名和本机服务地址生成运行目录中的交互配置副本，在凑卡前用 OmegaConf 单独解析该文件。预检子进程移除 `USER`、`STUDENT_API_BASE` 和 `SERVING_API_KEY`。配置仅保存凭证变量名，真实服务鉴权由运行时完成。

`b09d2fb` 的训练监视循环两次轮询训练器，正常退出若发生在两次检查之间，可被记为服务异常。修复版每轮分别读取训练器、学生和裁判的退出码并区分终止原因。读取历史记录可用 `status.py --work <运行目录> --finish-only`。表中列出控制器最后写盘时各进程的状态、veRL 的正式步号和该步用时、末步结束标记、最新存档步号和 CPU 合并痕迹。旧版只保存进程是否存活。存档目录的大小与结束标记都不能单独证明检查点可读。

### 补做既有 B 运行的 CPU 验收

用同一解释器和 `CUDA_VISIBLE_DEVICES=''` 运行 `audit_posthoc.py --work <原运行目录> --check-terminal-failures`。返回 0 表示已覆盖的日志中没有技术终止记录，2 表示出现终止记录，3 表示可用日志覆盖不足。可恢复的重试以及正则成功提取决定会单列。

随后运行 `verify_checkpoints.py --work <原运行目录> --output <运行目录之外的新目录>`，按步依次调用 veRL 的 FSDP 合并器，生成两份 HF 权重，检查模型配置、张量键与索引，并逐片抽读一个 CPU 数值。输出目录保存验收表和合并日志。优化器等训练状态文件只检查存在且非空。两项操作均在容器内进行。
