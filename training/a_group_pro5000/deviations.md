# A 组 RTX PRO 5000 移植偏离清单（T-017）

1. **硬件平台替换**：由原论文 GH200（及此前重试包 v4 的共享 A100 80GB）改为 `8 × NVIDIA RTX PRO 5000 72GB Blackwell`（每卡 `73,415 MiB`，`sm_120`，卡间仅 PCIe 互联，GPU 0–3 在 NUMA 0、GPU 4–7 在 NUMA 1）。
2. **`sm_120` 源码编译**：`flash-attn 2.8.3` 与 `causal-conv1d 1.6.2.post1` 在独立 conda CUDA 12.8（`nvcc 12.8.93`、`g++ 12.4.0`）环境下按 `TORCH_CUDA_ARCH_LIST="12.0"` / `FLASH_ATTN_CUDA_ARCHS="120"` 从源码编译安装；其余 Python 包与 A100 `env-candidate-vllm0191` 对齐（`wheel 0.48.0`、`triton 3.6.0` 为本环境配套版本）。
3. **运行时环境与选卡参数**：子进程需将独立环境 `lib/` 置顶到 `LD_LIBRARY_PATH`（确保 `libicui18n.so.79` 加载 conda 的 `CXXABI_1.3.15` `libstdc++.so.6`），且 `vllm 0.19.1` 需在 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 下使用整数卡号（不能使用 `GPU-<uuid>`）。
4. **单卡对照样本与判据**：按 T-017 使用基座 `Qwen3.5-4B` 与作者公开训练集分两组随机种子构造 6 条长中短多轮对话代替 C 阶段检查点与站内轨迹；按阶段 1 审核采用无 autocast/关 TF32 的真 FP32 前向作为参考，核验快路径 BF16 相对 FP32 的 `p99`/`p99.9`（≤ 慢路径 2 倍）、最大值（≤ `max(1.0, 2×慢路径最大值)`）、有限值与均量相对差（a–d）。
5. **裁判 `Qwen3.6-27B` 改为 `TP=2`（与作者一致）**：实测 `73,415 MiB` 单卡在作者 `launch_judge_server.sh` 默认 `max_model_len=32768, max_num_seqs=256` 下无法完成初始化；`TP=2`（`0.90`）实测可提供 `33.72 GiB`/卡 KV 与 `29.35x` 并发，与作者本身采用 2 卡一组部署裁判的并行结构一致。
6. **检查点与磁盘策略（计划）**：为适配 `/data` 有限空间（当前余 `61G`），检查点仅保存 `[model, extra]`（不含优化器，回齐作者默认 `load_contents=[model,extra]`）并只保留最新 1 份，日志只留指标不落盘 rollout 明细；系统盘 `/` 仅余 `7.8G`，所有缓存与临时目录强制落到 `~/bzhang/a_group_pro5000/`。
