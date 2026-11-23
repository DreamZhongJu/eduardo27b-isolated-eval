# A-Group Step 100 x Self-Built Three-Tier Evaluation (Pro 5000, T-021)

本目录用于在 RTX PRO 5000（72GB）上以 2 张空闲整卡（1 卡 A 组第 100 步 Tutor `bfloat16` + 1 卡现有 `pro5000-fast` 学生 `Meta-Llama-3.1-8B-Instruct`，严格 `--dtype float16`，若 `float16` 跑不起则立即停机不换 `bfloat16`）运行自建三档测试集评测，只生成对话与后测并调用 Agnes（<=3 路，严格等 T-020 全量完成后）。

## 文件说明
- `export_27b_selfbuilt.py`：在 A100 上只读运行，导出 1,489 道仅含输入题干/答案的 JSONL（无 27B 对话/作答）、27B Agnes + 离线 `math_verify` 作者规则逐来源计数 CSV 与核验清单 `tar.gz`；另在阶段 3 支持 `--mode t019-base` 导出 T-019 base 计数。
- `import_bundle.py`：Pro 5000 侧导入核对（SHA-256、1,489/8 数量、`(difficulty, source_id, source_key, row_index)` 严密对齐）。
- `preflight_agroup_pro5000.sh`：Pro 5000 只读预检（不占卡、不调 Agnes；打印既存 `pro5000-fast` 环境版本及与 27B 学生的偏离项，并核对两个模型权重 SHA-256 与导入包）。
- `setup_math_verify_090.sh`：在 `$BASE/envs/math_verify_090` 搭建纯 CPU、不改 `pro5000-fast` 的 `math-verify==0.9.0` 隔离小环境，供作者规则重判与 2,880 份换机器作答重判使用。
- `wait_gpus_agroup_pro5000.py`：Pro 5000 2 照等卡器与 Tutor 占位器（70,000 MiB 门槛、兼容 GPU 6 传感器特性）。
- `stack_check_selfbuilt.py`：阶段 1 换机器学生检查（每档按来源哈希种子 42 抽 30 题，共 2,880 次重跑前测并与 A100 27B 比对均值差 `<=0.03` 与相关性 `>=r_bench-0.05`）。
- `run_post_eval_pro5000.py` & `run_agroup_pro5000.sh`：Pro 5000 启动入口（`--stack-check` / `--mode pilot` / `--mode full` / `--grade-only` / `--status`），调用 Agnes 前强制机械检查 T-020 全量已结束且并发 `<=3`。
- `make_selfbuilt_paired_csv.py` & `summarize_agroup_pro5000.py`：生成 `paired_gain.py` 所需 CSV（`orig`、`easy`、`medium`、`hard`）并交叉核对（点估计一致、CI `<=0.01`）。
- `test_agroup_pro5000.py`：纯合成数据单元测试。
