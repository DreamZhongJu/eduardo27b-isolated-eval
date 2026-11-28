# Deviations Log (T-033 Route A Step-100 Evaluation on Pro 5000)

1. **自建三档学生模型精度差（bf16 vs fp16）**：
   - 路线 A 第 100 步（T-033）在自建三档评测中，学生模型 `Meta-Llama-3.1-8B-Instruct` 统一按 `bfloat16` 运行（遵循工人须知第 9 条及 T-033 总览定法）。
   - 所复用的 A100 27B 前测与无辅导对照，以及待配对比较的 A 组（T-021）和 base（T-019）学生模型当时按 `float16` 运行。
   - T-033 阶段 1 第 4 条通过 90 个确定性抽样来源（每档 30 个，种子 42）的无辅导直测（`fp16` vs `bf16`，每题 32 次，同源配对 10,000 次 bootstrap）定量测量两精度在当前环境下的通过率差与相关系数，并在阶段 3 汇报中与所有自建三档配对差并排注明“路线 A 的学生是 bf16，对照是 fp16”。

2. **裁判模型与并发替换（Agnes 替代 Gemini）**：
   - 作者 494 题与自建三档评测中的对话门禁/打分/后测裁判均使用 `agnes-3.0-flash` 替代论文的 `gemini-3.1-pro-preview`，并发采用 `agnes_window.py` 的 AIMD 拥塞窗口（初始 1、成功 +1、限流/异常减半、在途合计上限 16 路）。
   - 因此，泄题率（`leak_rate`）、教学拒收率（`teaching_reject_rate`）、按泄题清零的 $\Delta$ 及 `Ped./Fact.` 仅在路线 A、A 组、base 三者之间内部对比，不与论文 Gemini 口径直接横向对比。
