# T-031 Route C Data Kit (`routeC_data`)

- 任务卡：`T-031`（路线 C 训练数据：按原题难度分流清点，补测原题前测）
- 分支：`feature/routeC-data-20261127`
- 隐私与白名单：本包仅含纯代码、假数据单测与校验文件，不含任何题面、答案、作答或密钥。

## 模块说明

- `routeC_data/audit_cu04_source_pretest.py`：cu04 只读盘点 `data/训练集正式出口/` 与 `data/训练来源/` 中的原题旧前测记录（纯计数/哈希）。
- `routeC_data/routeC_inventory.py`：按 `(τ, m)` 对 2,100 道原题（`train=2,000`、`dev=100`）与 3,736 道变体做分流清点、地板/天花板统计及方向标签交叉表；阶段 1 使用旧 `source_pre_solve_rate_old`，阶段 2 自动接入新测得的 `source_pretest_measured.jsonl`。
- `routeC_data/stage2_source_runner.py`：阶段 2 补测 2,100 道原题前测（学生 Llama-3.1-8B bf16，`k=16`，`temp=0.6`，`max_tokens=1024`；复用 T-023/T-024 混合判法 `routeA_data/hybrid_grader.py` sha256 `7d2a5f46...` + 本地 Qwen3.6-27B TP=2；不使用 Agnes）。
- `routeC_data/run_stage1_pro5000.sh`：阶段 1 Pro 5000 纯 CPU 入口脚本（`CUDA_VISIBLE_DEVICES=""` + `nice -n 19`，不碰 T-026 目录）。
- `routeC_data/preflight_stage2_pro5000.sh` / `routeC_data/run_stage2_pro5000.sh`：阶段 2 Pro 5000 预检与运行脚本（仅在阶段 1 审过、T-026 结束且 8 卡全空、用户同意后使用）。
- `tests/test_routeC_data.py`：纯假数据单测与防正文泄露检查。
