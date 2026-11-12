# 训练集流式生成：待小样本验证的独立版本

此目录提供新服务器的实验性生产者—消费者实现，和本仓库 Eduardo 评测程序独立。旧结果、冻结代码和原判分不会被覆盖。本目录不包含题目、作答、模型、日志或密钥；数据、数据库及导出必须放在仓库外。该实现尚未在目标 A100 机器或真实 Agnes 接口验证，不应直接当作旧产线的完全等价替换。

## 流程和边界

一个 API 生产者以 5 个线程处理 Agnes 出题、盲审、答案等价复核和最终答案判分，因此这些环节**合计**最多 5 个并发请求；不得在其他进程或机器绕开该限制。程序在本机使用一个共享文件锁防止多数据库同时启动生产者。每个可复算来源最多 6 次提案；保留作者迁移题并单独标注来源。独立 vLLM 学生消费者领取已审核候选的 16 个独立采样任务，判分请求随后进入同一 API 队列。GPU 不必等整片出题完毕。

学生采用 Llama-3.1-8B-Instruct、FP16、温度 0.6、top_p 0.9、top_k 50、4096 输出 token、每题 16 个固定样本 ID；保持 `length` 截断判错，以及 boxed 附近各 1000 字符、无 boxed 时首尾各 1000 字符的答案截取规则。出题和盲审沿用原训练方向和 Skill 的核心要求。本实现重新组织了代码和提示词，不能直接断言判分、审核或学生推理与旧产线逐字逐项等价，须先用已保存样本校验。

本版本把每个请求开始状态、完整成功响应、候选和结果写入站内 SQLite。显式 HTTP 408/429/5xx 最多再请求三次；传输中断、响应无法解析以及进程在途退出均保留 unknown/started，**不会自动重复请求**。这会让单题停留未决，但其他任务可以继续。身份碰撞或协议改变会报错。队列在单机本地磁盘使用 SQLite WAL；不要放在 NFS，也不要跨机器共享同一个数据库。需要磁盘权限隔离，不能把数据库、导出或数据加入 Git。

## 迁移旧数据

旧两站原始数据须通过管理员允许的方式单向导入新服务器；不要从受限服务器向外传题目、作答、日志或文件。以下程序对已经有运行目录的旧片**整片保守冻结**，以避免新程序重复生成已发生过的请求；它不会把旧片结果转换为新队列，也不会完成旧片中断样本的续跑。不要声称 100% 自动续接历史数据；旧片还需要单独核对身份与收据，编写/验证导入后再解锁。

示例，`--old-root` 可重复并指向迁移进站内的两个旧训练根目录（及旧范围，均要有 `runs` 和 `data/shards`）：

```bash
python -m stream_train.history --old-root /input/old-a --old-root /input/old-b --out /work/hold-source-ids.txt
```

每行是一个需要冻结的来源 ID。初始化时对照文件中的 ID 跳过已处理来源；没有旧数据时，只有确保输入确为全新且互不重叠，才用 `--fresh-only`。务必自行核验清单覆盖全部旧实验目录，程序不能发现未提供的历史范围。旧正式结果不被并入新数据库或新导出；后续汇总需按来源和候选身份去重。

## 配置与启动

下面是站内示例。路径、模型版本和接口可达性需现场核对；不要在命令或配置里写入密钥。使用环境变量或站内密钥管理器注入 `AGNES_API_KEY`。生成任务会向 Agnes 发送原题与参考答案；盲审会发送原题与候选题；判分会发送候选题、参考答案及学生答案截取。用户已授权这些用途；不发送教师对话。最终配置、模型权重哈希、代码提交和来源文件哈希须留在站内。

```bash
mkdir -p /work/run
cat > /work/run/config.json <<'JSON'
{"namespace":"train-stream-v1", "api_base":"https://api.agnes-ai.cn/v1", "api_model":"agnes-3.0-flash", "student_base":"http://127.0.0.1:8000/v1", "student_model":"student-8b"}
JSON
python -m stream_train --db /work/run/queue.sqlite --config /work/run/config.json init --sources /input/train.jsonl --hold-source-ids /work/hold-source-ids.txt
```

输入是本项目冻结的官方 train JSONL，每行包含 `source_id`, `source_row_index`, `source_split`（`train`）, `problem`, `answer`，可含 `author_rates`, `direction_band` 和 `paper_transfers`（每项至少有 `candidate_id`, `question`, `answer`）。来源文件不随代码上传。无 `direction_band` 时以 `mixed` 处理，不能把它当作作者实测方向。

启动 vLLM 的示例以目标机实际安装版本的 CLI 为准：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /models/Meta-Llama-3.1-8B-Instruct --served-model-name student-8b --dtype float16 --tensor-parallel-size 1 --max-model-len 8192 --gpu-memory-utilization 0.80 --host 127.0.0.1 --port 8000
python -m stream_train --db /work/run/queue.sqlite --config /work/run/config.json api
python -m stream_train --db /work/run/queue.sqlite --config /work/run/config.json student --workers 16
python -m stream_train --db /work/run/queue.sqlite --config /work/run/config.json status
python -m stream_train --db /work/run/queue.sqlite --config /work/run/config.json export --out /work/run/export.jsonl
```

上面三个长期命令放在容器的独立进程或受控作业中；容器禁止代理和外部端口，按站点规则从阿里源安装，API 保持本机适用的网络策略。`export` 使用新格式，每个候选包含 16 个样本、可判定个数、正确数与通过率；仅 `complete=true` 是全部 16 次已明确判分。`source_has_complete_self_generated=true` 表示该来源在本次导出中至少有一道自生成题完整判分；训练入集需要同时检查这两个标记，旧数据仍需另行合并。

## GPU 选择与验证

建议从 **1 张 A100 80GB、TP=1、一个 8B 学生实例**开始，16 个请求由 vLLM 连续批处理。额外显卡用于另一个单卡实例和独立本地端口，而不是把一个 8B 模型拆成张量并行；实际加到 2、4 张还是维持 1 张应根据队列积压、输出 token/s、显存和 API 限速实测决定。本版本的 student endpoint 是一个配置绑定；多实例负载均衡、单机两张卡协同和多服务故障转移尚未实现，不能直接照此启动多个不同端口消费者接同一个数据库。

设计参考：[vLLM 并行策略](https://docs.vllm.ai/en/v0.33.0/serving/parallelism_scaling/)建议模型能装入单卡时先使用单卡；[vLLM 数据并行部署](https://docs.vllm.ai/en/v0.20.1/serving/data_parallel_deployment/)说明可复制模型实例处理独立请求。具体版本能力和吞吐仍以站内运行实测为准。

先在站内完成 1 个此前未使用的合成来源/小样本、真实 vLLM 与 Agnes 端到端测试，检查输出、截断、完整率、单卡显存、吞吐、API 并发，以及重启时已开始任务不重复提交；再针对新鲜来源扩大。测试不能证明旧批次能直接恢复，也不能证明不同模型或环境的结果可无条件混用。
