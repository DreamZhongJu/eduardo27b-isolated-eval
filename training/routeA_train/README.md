# Route A Training Stage 1 (`training/routeA_train`)

T-026 路线 A 正式训练（阶段 1：代码、单测与 Pro 5000 无卡预检）模块。

## 组件

- `prepare_parquet.py`：在 Pro 5000 本机直接读取 T-023 重测包 `routeA_stage2_measured.jsonl`（核对 SHA-256 `9ff8449b…`），按作者 `process_dataset.py` 格式生成 `train.parquet`（剔除所有变体 `n_correct_hybrid >= 15` 的天花板组）和 `val.parquet`（`split = dev` 的 100 道原题），并把梯子信息写入 `extra_info.variants`。仅输出纯数字计数与统计，不打印任何题面、答案或作答原文。
- `extract_t017_control.py`：只读读取 T-017 正式训练运行目录（`stage3_formal_20261121_144311`）的 `step-statistics/` 与 `trainer.log` 数字摘要，提取逐步奖励与门禁对照表，供阶段 3 防奖励作弊监控使用。
- `preflight_stage1.py` & `run_stage1_preflight.sh`：Pro 5000 无卡预检入口（`CUDA_VISIBLE_DEVICES=""`），核对 `math-verify==0.8.0` / `latex2sympy2_extended==1.10.2`、文件与模型分片 SHA-256、`/dev/shm` 权重、`/data >= 70 GB`、8 张卡空闲、生成 parquet、运行 Hydra `--check-only`、运行全部单元测试并提取 T-017 对照表。
- `test_routeA_train.py`：仅使用假数据的 Stage 1 全覆盖单元测试。

## 27B 并发分配说明（#6）

Pro 5000 上 27B 服务（`Qwen3.6-27B`, `TP=2`, GPU 5 & 7）配置 `--max-num-seqs 256`，`reward.num_workers = 2`：
- 门禁裁判：`judge_concurrency = 16` / worker（2 workers 合计 `32`，按 T-026 卡第 2 版与 T-017 正式运行一致）；
- 混合判法兜底：`grader_max_concurrency = 32` / worker（2 workers 合计 `64`，配合 AIMD 拥塞窗口 `1..32` 与独立 `hybrid_grader` 信号量）；
- 合计并发上限：`32 + 64 = 96 <= 256`（避免 T-017 在 `judge_concurrency=64` 时触发的 `300s` 长尾超时，同时给阶段 2 试跑留足排队余量）。
