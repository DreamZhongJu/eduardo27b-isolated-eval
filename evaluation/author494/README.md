# Author-494 Eduardo-27B Evaluation (Pro 5000 & A100)

本目录用于复跑作者上游 `e8c89b7` 的 `eval/run_env_test_eval.py`，评测 `Eduardo-27b-3bbf17e`（`bfloat16`）与学生 `Meta-Llama-3.1-8B-Instruct`（`bfloat16`）。

## Pro 5000 版本（T-022 第 3 版，TP=2 + DP=3/2，5 张卡优先，不足 5 张降为 4 张）

- `download_27b_pro5000.py`：从公开渠道 `hf-mirror.com`（无 token、无代理）断点下载 `eth-nlped/Eduardo-27b`（`3bbf17e44287eadbfda1fa09bb73a1bade407ff2`）到 `/dev/shm/zhujie_a_group/Eduardo-27b-3bbf17e`，校验全部 9 个模型文件（含 `TUTOR_LOCK` 6 文件及 `model.safetensors.index.json` 等）精确字节数与 SHA-256；遇 401/403 立即报错停下。
- `preflight_27b_pro5000.sh`：Pro 5000 只读预检（`envs/pro5000-fast`、`$BASE` 缓存），核对 `vllm==0.19.1`、`math-verify==0.8.0`、`latex2sympy2_extended==1.10.2`、上游 `e8c89b7` 22 文件、Tutor 9 文件及 `bfloat16`、学生 7 文件、T-020 `test.parquet`（`338dc73b...`）与 70,000 MiB / GPU 6 空卡状态。
- `wait_gpus_27b_pro5000.py`：Pro 5000 只读等卡与占位器（70,000 MiB 门槛、GPU 6 `used_mib < 500` 规则、优先 `util < 10` 卡），支持 `--num-gpus 5 --fallback-min-gpus 4`（优先 5 卡 TP=2+DP=3，不足 5 卡自动降 4 卡 TP=2+DP=2）。
- `run_27b_pro5000.sh`：Pro 5000 启动器（强制 `--tutor-tp 2`、`--student-dp auto|2|3`、`--auto-wait`、`--resume`、`--status`、`bfloat16` 学生、Agnes 并发 `12+4=16`）。
- `run_27b_pipeline_pro5000.sh`：Pro 5000 一体化入口（下载核对权重 → 预检 + Agnes ping → 20 题冒烟 → 冒烟闸门校验 → 自动全量 494 题与汇总；支持 `--resume` 与 `--status`）。

## 共享加固层与 A100 原版文件

- `eval_hardening.py`：幂等落盘与 `--resume` 回放层（覆盖 Tutor `LocalVLLMTeacher`、学生与 Agnes；逐条 SHA-256 + `replay_manifest.json` 严格绑定）。
- `agnes_window.py`：Agnes AIMD 拥塞窗口（`1..16`，内层重传 `2` 次叠外层 `3` 次最多 `12` 次/票，技术失败 `>1%` 不可用）。
- `run_eval_27b_author494.py`：内存包装入口（零修改作者源码，强制验证 Tutor `config.json` 为 `bfloat16`）。
- `agnes_ping_author494.py`：单次合成对话 Agnes 连通测试。
- `summarize_author494.py`：脱敏汇总脚本（`--paper-baseline eduardo-27b`、`--ours-label`、`--smoke-gate`）。
- `preflight_27b_author494.sh` / `run_27b_author494.sh` / `wait_gpus_author494.py` / `probe_27b_load.py`：A100 原版脚本（保留以兼容既有单测）。

## cu04 单元测试

```bash
cd ~/项目/数学迁移/代码与工作树/eduardo-author494-eval-27b-20261123
~/.local/envs/mhpo/bin/python -B -m unittest discover -s evaluation/author494 -v
```
