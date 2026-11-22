# Qwen3.5-4B Base × Self-Built Three-Tier Evaluation (A100)

本目录用于在 A100 容器（`edu-contain`）上以 2 张空闲整卡（1 卡 Tutor `Qwen3.5-4B` + 1 卡 Student `Meta-Llama-3.1-8B-Instruct`）运行自建三档测试集评测，只生成对话与后测并调用 Agnes 判分，前测与无教学对照严格按 `(difficulty, source_id)` 复用 27B 已核验记录。

## 文件说明

- `ref27b_manifest.py`：只读核对 `test.jsonl` 与 27B 目录 `/workspace/results/own-test-499-v1`（含叠层恢复与补判层），跳过 27B 未完成的 8 个档位-来源，并提取前测与对照计数；任何来源或哈希不匹配即报错停止。
- `run_post_eval.py`：复用未修改的 `evaluation/app/`、`evaluation/official/` 与 `evaluation/agnes_eval_v3/agnes_judge.py`（按 SHA-256 `2ffe7464...` 校验），仅生成对话与 32 份后测（`source` 与 `ours_<tier>` 各 16 次）并调用 Agnes 判分，另提供离线作者规则（`\boxed{}` + `math_verify`）敏感性对照。
- `wait_gpus_base4b.py`：只读 2 卡等卡器与 Tutor 启动占位器。
- `preflight_base4b.sh`：不占卡、不调 Agnes 的只读预检（核对 GPU、进程、端口、`CODE_BINDING.json`、Student vLLM 0.18.0 环境、Tutor vLLM 0.19.1 环境、模型/数据 SHA-256 及 27B 1,489/8 清单）。
- `run_base4b_selfbuilt.sh`：启动入口（Student 沿用 `eduardo-vllm018` 的 `vllm 0.18.0 --dtype float16`，Tutor 使用 `env-candidate-vllm0191` 的 `bfloat16`），支持 `--auto-wait`、`--mode pilot|full` 与 `--status <dir>`。
- `summarize_base4b.py`：输出脱敏汇总（不含来源编号、题面、答案、作答正文、WARNING/ERROR 原文或异常消息），报告 R0–R4、未决/高分剔除敏感性及 `base4b - 27B` 配对 bootstrap（10,000 次，种子 42）。
- `test_base4b_selfbuilt.py`：纯合成数据单元测试。

## cu04 单元测试

```bash
cd ~/项目/数学迁移/代码与工作树/eduardo-base4b-selfbuilt-20261121
~/.local/envs/mhpo/bin/python -B -m unittest discover -s evaluation/base4b_selfbuilt -v
```
