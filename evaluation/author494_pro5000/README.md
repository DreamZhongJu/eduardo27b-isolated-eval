# Author-494 Evaluation on RTX PRO 5000 (T-020: A-group step_100 & Qwen3.5-4B base)

用于在 RTX PRO 5000 服务器（`~/bzhang/a_group_pro5000`）上按 T-018 同一套作者上游 `e8c89b7` 评测代码和参数，对以下两个 Tutor 各跑一次作者 494 题全量评测并配对对比：

1. **A 组第 100 步导出**：`~/bzhang/a_group_pro5000/runs/stage3_formal_20261121_144311/exports/step_100`
2. **Base 基座模型**：`/data/models/Qwen3.5-4B`（HF `851bf6e8`）

## 与 T-018（A100）的核心差异（已固定）

- 硬件：`8 × RTX PRO 5000 72GB Blackwell`（`73,415 MiB`/卡，`--min-free-mib 70000`；处理 GPU 6 空闲传感器 `util=100` 特性）
- 运行时：`LD_LIBRARY_PATH=$ENV_DIR/lib`（conda `libstdc++ CXXABI_1.3.15`）、`CUDA_DEVICE_ORDER=PCI_BUS_ID`、缓存目录全部指向 `$BASE`
- Agnes 并发：Pro 5000 上限 **3 路**（`--judge-concurrency 2 --eval-judge-concurrency 1`）

## 文件清单

- `preflight_author494_pro5000.sh`：只读预检（不占卡、不调 Agnes），核对卡/进程/端口/磁盘、上游 `e8c89b7` 22 个文件、包版本、A 组第 100 步导出清单（T-017 收尾第 2 步生成的 `export_step100_verified.json` 与 `export_step100_sha256.txt`）、base Tutor 11 文件、`/dev/shm/zhujie_a_group/` 学生 8B（7 文件）及 `test.parquet`（494 行）。
- `agnes_ping_author494.py`：用 1 条合成对话走作者 `make_judge_client` 路径测试 Pro 5000 到 Agnes 的连通性与 JSON 解析。
- `wait_gpus_author494_pro5000.py`：只读 4 卡等卡器与学生启动期间 Tutor 卡占位器（`min_free_mib=70000`）。
- `run_author494_pro5000.sh`：Pro 5000 通用评测启动器（A 组与 base 共用，参数完全一致），支持 `--prepare-only`、`--auto-wait`、`--status`。
- `make_paired_csv.py`：读取 A 组和 base 两次全量结果 JSON，按 `row_index` 对齐生成 `paired_gain.py` 所需 CSV（只留 Pro 5000），校验前测一致且 CSV 均值与作者 JSON `DELTA_TRANSFER` / `DELTA_SAME` 差 `< 1e-9`。
- `summarize_author494_pro5000.py`：单次运行汇总（用于阶段 1 base 20 题冒烟）与全量三表汇总（判定 A 复现、判定 B `paired_gain.py` 配对差、判定 C Pro 5000 base vs A100 base T-018，及次要指标并排表）。
- `paired_gain/`：照原样收入 `组织/参考/a-group-paired-gain/` 的三个文件（`paired_gain.py`、`test_paired_gain.py`、`README.md`）。
- `test_author494_pro5000.py`：标准库 `unittest`（纯假数据）。

## cu04 本地单元测试

```bash
cd ~/项目/数学迁移/代码与工作树/eduardo-author494-eval-pro5000-20261123
~/.local/envs/mhpo/bin/python -B -m unittest discover -s evaluation/author494_pro5000 -v
SOURCE_GAIN_PY="$PWD/evaluation/agnes_rejudge/source_gain.py" \
  ~/.local/envs/mhpo/bin/python -B -m unittest discover -s evaluation/author494_pro5000/paired_gain -v
```
