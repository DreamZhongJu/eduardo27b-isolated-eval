# Author-494 Eduardo-27B Evaluation on A100

本目录用于在 A100 上以最多 4 张卡（Tutor 1 卡 TP=1 + 学生 3 卡 DP=3；若单卡显存/KV 不足则 Tutor 2 卡 TP=2 + 学生 2 卡 DP=2）复跑作者上游 `e8c89b7` 的 `eval/run_env_test_eval.py`，评测 `Eduardo-27b-3bbf17e`。

## 文件说明

- `preflight_27b_author494.sh`：只读预检（不占卡、不调 Agnes），检查卡与进程归属、端口、磁盘、作者上游 `e8c89b7` 22 文件 SHA-256、严格版本钉死（`vllm==0.19.1`、`math-verify==0.8.0`、`latex2sympy2_extended==1.10.2`）、`Eduardo-27b-3bbf17e`（6 文件 + 与 `own-test-499-v1/PILOT/RUN_CONFIGURATION.json` 的 `teacher_config_sha256` 比对）、`Llama-3.1-8B-Instruct`（7 文件）及官方数据集 SHA-256。
- `agnes_window.py`：面向 Agnes 接口的进程内 AIMD 拥塞窗口（初值 `1`、成功 `+1`、上限 `3`、遇到 408/429/5xx/超时/网络/结果不明时减半至不低于 `1` 并做指数退避加抖动上限 `60s`）与内层最多 `2` 次传输重传（叠加作者外层 `num_retries=3`，每票最多 `12` 次 HTTP 请求）。HTTP 200 且具备合法 completion 结构时原样交回作者解析，不额外重传；结果不明记成技术失败。
- `run_eval_27b_author494.py`：不修改作者任何文件，在内存中包装 `eval.run_env_test_eval.make_judge_client` 与共享的 Agnes `get_async_openai_client`，确认 Tutor `config.json` 为 `bfloat16`，并写出脱敏汇总 `agnes_window_stats.json`。
- `probe_27b_load.py`：只读等 1 张空闲整卡，验证 `Eduardo-27b-3bbf17e` 在 `env-candidate-vllm0191` 下能否以 `tensor_parallel_size=1, gpu_memory_utilization=0.85, max_model_len=20000` 加载（可选 `--allow-tp2-fallback` 在 OOM/KV 不足时探测 TP=2）。
- `run_27b_author494.sh`：阶段 1–2 启动器（支持 `--tutor-tp 1|2|auto`、`--auto-wait`、Tutor 占位防抢卡、`--judge-concurrency 2 --eval-judge-concurrency 1`、`--status` 及自动调用 `summarize_author494.py --paper-baseline eduardo-27b`）。
- `agnes_ping_author494.py`：用 1 条合成假对话走作者 `make_judge_client` + `AgnesWindowClient` 调 1 次 Agnes，只打印 HTTP 状态、返回模型名、JSON 解析结果与窗口计数。
- `wait_gpus_author494.py`：只读等卡器与 Tutor 卡占位器（两次快照一致且 `>=78000MiB`、`util=0`、`0` 进程才选卡）。
- `summarize_author494.py`：脱敏汇总脚本，支持 `--paper-baseline eduardo-27b`（对照论文 Eduardo-27B：$\Delta_{\text{transfer}}=+0.29\ [+0.25,+0.32]$、$\Delta_{\text{same}}=+0.46\ [+0.42,+0.51]$）与 `--agnes-stats`。
- `test_agnes_window_27b.py` / `test_summarize_author494.py`：标准库 `unittest` 测试（纯假数据 / 假客户端）。

## cu04 单元测试

```bash
cd ~/项目/数学迁移/代码与工作树/eduardo-author494-eval-27b-20261123
~/.local/envs/mhpo/bin/python -B -m unittest discover -s evaluation/author494 -v
```
