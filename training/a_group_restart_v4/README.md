# A 组快内核 v4

本包代码部署到独立目录。把已提供的 `a100-fast-kernels-offline-v1.zip` 上传到 `/workspace` 后，从容器执行。

```bash
v4=/workspace/a-restart-kit-v4/a_group_restart_v4
bash "$v4/prepare_wait.sh" /workspace/a100-fast-kernels-offline-v1.zip
```

该命令先在 CPU 上核对并准备独立环境。无 GPU 时检查 Qwen3.5 的真实类和模拟 GPU 门槛下的函数绑定，实卡实际 dispatch 由后续对照核对。脚本以后台方式等一张整卡，领取后两小时硬截止。通过的对照会创建快路径回执，拒绝对照保存聚合数字供检查。耐久阶段逐轮落盘耗时与负载。候选只读取用户已经验收的 C 阶段副本。

对照通过后的只读预检和状态查看。

```bash
v4=/workspace/a-restart-kit-v4/a_group_restart_v4
py=/workspace/A_group_repro_20261117/envs/env-candidate-vllm0191/bin/python
"$py" -B "$v4/restart_v4.py" --check
"$py" -B "$v4/restart_v4.py" --status
```

预检列出卡号、布局、磁盘余量和批准后启动命令。四、五张整卡选择四卡，六张及以上选择六卡。实际启动只允许使用明确批准的卡号。续训使用同布局最新完整检查点，并在预检和已批准的启动命令中指定 `--resume`。状态默认报告第 1、2 步和最近一次存盘，也可通过 `--step 25` 等指定本次运行的某步。
