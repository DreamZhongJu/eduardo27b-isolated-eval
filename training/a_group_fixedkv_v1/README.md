# A组共享显卡单步：固定 KV 缓存

仅用于隔离单步试验：Tutor 2 卡、学生 1 卡、裁判 1 卡。使用作者数据及 BF16；每题 16 次采样和上下文长度不变。

相较原共享版，新增覆盖 `actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=4294967296`，每 Tutor GPU 固定 KV 4 GiB，以绕过共享显卡环境下 vLLM profiling 的空闲显存波动断言；无需修改底层框架。框架源码说明固定 KV 后不再用 `gpu_memory_utilization` 限制缓存。FSDP 训练和激活内存并无本参数的硬限制；现有剩余显存启动阈值及每2秒低余量保护继续有效，但不能保证系统隔离。4 GiB 是首次试验，不保证装载成功或单条上下文容量，若不足须报告。Tutor rollout 最大并发序列设为 64，减小并发和缓存可能降低吞吐，且不能保证两种调度逐 token 位相同。`deviations.md` 将记录此适配。

不改变历史数据，单步试验应使用独立新名称；未在服务器/GPU进行实测。
