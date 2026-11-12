# 正式训练结束后自动释放学生卡

`stream_train/auto_stop_student.py` 是独立看护程序，不修改训练代码和队列。正式数据初始化成功、学生消费者和专用 vLLM 服务均已启动后运行；请勿绑定虚构题试点。初始化后不再从外部追加来源，流水线内部创建后续子任务不受影响。

看护只读正式 SQLite 队列，并检查学生服务的本机指标。所有任务不再处于 `pending` 或 `started`，且队列及服务空闲状态连续稳定至少 120 秒后，先停止该队列的学生消费者，再次核验并停止已匹配的专用 vLLM 服务。模型服务必须只供本次实验使用。还有 `unknown` 时也会释放显卡，但明确报告仍有未决，不代表所有结果完成；旧记录和未决状态均保留。

如果队列未结束、进程身份无法确认、系统不支持 pidfd、指标不可用，或验证中发生异常，程序不会主动关闭模型。训练遇错停滞仍有待处理任务时也不会自动停卡，需检查日志；不要把队列暂时空闲当作任务结束。程序只关闭精确识别的进程，不使用进程组信号。

需 Linux `/proc`、SQLite WAL 可读、vLLM 本机 `/metrics` 指标可读，以及 Python 内置 `os.pidfd_open` 和 `signal.pidfd_send_signal`。不同容器和安全配置需现场验证。停止后用 `nvidia-smi` 检查显存是否释放。

例如正式数据库和 namespace 分别使用 `/workspace/results/train-stream-v2-main/queue.sqlite`、`train-stream-v2-main`，模型为 `/workspace/models/Meta-Llama-3.1-8B-Instruct`，服务名为 `student-8b`，端口为 `18761`。这些必须与正式训练的真实配置一致；以下命令只能在正式初始化、学生消费者、专用模型服务启动之后执行：

```bash
nohup /workspace/eduardo-vllm018/bin/python /workspace/stream-train-autostop/auto_stop_student.py \
  --db /workspace/results/train-stream-v2-main/queue.sqlite \
  --namespace train-stream-v2-main \
  --model-path /workspace/models/Meta-Llama-3.1-8B-Instruct \
  --student-model student-8b --port 18761 \
  --confirm-initialized --confirm-dedicated \
  >> /workspace/results/train-stream-v2-main/autostop.log 2>&1 < /dev/null &
```

查看：`tail -n 20 /workspace/results/train-stream-v2-main/autostop.log`。如果服务或消费者没有被唯一识别，可在核对后指定 `--service-pid` 和 `--consumer-pid`；看护仍会进行完整身份检查。不要与其他任务共用这个服务。

现有 7,246 来源包包含历史已启动和状态不明的来源，需要先处理历史防重清单，再初始化正式队列；自动停卡脚本不负责生成防重清单。
