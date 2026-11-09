# Eduardo-27B 实验：基础镜像与独立代码

本仓库的 Dockerfile 使用阿里云官方镜像构建 **CUDA 12.8.1 开发工具 + Ubuntu 24.04 的基础环境**，包含 `nvcc`、Python 3.12、pip、venv、Python 头文件、C/C++ 构建工具、git、curl 与 uv 0.9.5。按用户最新要求，不安装 Miniconda。基础镜像来自阿里云容器仓库，APT、pip 与 uv 配置使用阿里云源；不会内置实验代码、vLLM、PyTorch、判分程序、模型或数据，这些依赖和文件需按服务器规则另行准备。

本地没有 Docker daemon 权限，也没有连接目标服务器；镜像构建、驱动兼容、8×A100、依赖安装及 GPU 运行均未实测。已核验阿里官方 CUDA 开发镜像的索引摘要并固定在 Dockerfile，尚未拉取或构建该镜像。如站点已有经批准镜像，可由管理员审核调整。Ubuntu APT 包并非逐项固定版本，严格复现需记录实际镜像摘要与包版本。CUDA 12.8 的配套驱动和 NVIDIA Container Toolkit 须在站内确认；不要通过禁用版本检查绕过驱动不足。

## 数据边界

详见 [AGENT_GUIDE.md](AGENT_GUIDE.md)。服务器上的源码、题目、答案、模型、日志和结果文件不能被上传、下载出站或发给云端 AI。用户允许对外反馈汇总数字和固定错误类型；唯一获准的数据发送例外是：将**题目、参考答案、学生作答**发送给 Agnes 用于判分，不得附带师生对话、原始日志、文件、其他来源字段，也不得发送给其他外部服务。Agnes 账号总并发最多 **5**；需要由站内批准的唯一调度点或跨进程共享机制约束，单进程限制不能保证全局限制。用户另行允许代码和依赖从批准来源向服务器**单向下载**；不要使用代理、隧道、反向转发或上传途径规避站点规则。

## 先准备基础镜像

按站点制度，由管理员或有权限的人使用本仓库中的 Dockerfile 构建基础镜像，或在批准的机器上构建并以批准方式单向导入。构建需要访问阿里云容器仓库、Ubuntu 软件源和 PyPI 软件源，目标服务器是否允许访问这些地址须先按其制度核实。示例为 `docker build -t eduardo-base:local .`；构建日志留在服务器内。基础镜像没有 `COPY` 实验代码，也没有安装 vLLM 或训练依赖。Ubuntu 24.04 的系统 Python 不应直接用 pip 改写；在可写目录使用 `python3 -m venv /path/to/eduardo-env` 创建虚拟环境，再以该环境的 `python -m pip` 安装依赖。

在允许的 GPU 节点构建后，应分别运行 `docker run --rm IMAGE nvcc --version`、`docker run --rm IMAGE python3 --version` 和 `docker run --rm --gpus all IMAGE nvidia-smi`。镜像提供 CUDA 用户态工具，GPU 驱动由宿主机通过 NVIDIA Container Toolkit 注入；最后一项是否通过只能以目标机器实测为准。

之后按允许方式把本公开仓库的代码单向取得到服务器本地，或由管理员提供只读目录。不要在目标服务器对源码执行 `git push`，不要把凭据写入镜像、代码、命令行参数或日志。额外环境应单独安装在站内持久化目录：例如 Python 虚拟环境、`vllm==0.18.0`、兼容的 PyTorch/Transformers、OpenAI 兼容本机客户端、math-verify 等；实际版本及 CUDA/驱动兼容性需由管理员和单题测试确认，阿里镜像不保证包含每个 wheel。下载依赖属于批准的入站操作，不能解释成可外传实验数据。

版本选择依据：作者的 Dockerfile 实际基于 `nvcr.io/nvidia/vllm:26.03.post1-py3`，文件注明面向 GH200/aarch64 且包含 veRL 训练环境；这里准备的是 A100 推理基础环境，不是作者训练镜像的逐项复刻。我们之前在 2×A100 上用 vLLM 0.18 和 BF16 加载 Eduardo-27B 并完成短生成；另一次 CPU 依赖核对使用 torch 2.10.0+cu128。那次完整评测因缺少 pandas 中断，因此新站需要安装并验证 pandas、parquet 相关依赖，不能仅凭模型加载通过便判定全链路已通过。vLLM 及其 PyTorch、transformers 依赖应在独立环境解析并冻结，避免预先强装未经验证的组合。

## 代码原型与评测

`app/` 是在现有研究思路上改写的独立**离线评测原型**，不是正在其他服务器运行的冻结代码，也不是作者原版实验或 27B 的 DPPO 训练程序。程序期望本地教师和学生模型、题目及结果目录；可先用2张A100运行教师 TP=2、另1张A100运行学生，8卡足够不代表必须占满。教师 BF16、学生 FP16；本草稿建议同一依赖环境均用 vLLM 0.18，旧站学生用的是 vLLM 0.17，因此不能把两者直接当成完全相同的协议。`app/launch.py` 路径以其所在仓库自动解析，不需要将代码放进基础镜像 `/app`。

初始数据由站内合规准备的 `sources.jsonl` 给出，每行 `source_id`、`problem`、`answer`、`transfer_problem`、`transfer_answer` 为非空字符串。不要把题目和参考答案提交到 GitHub。本原型为**独立新测量**，不含或复用其他服务器 test:296 的64份历史作答；如需对照已选的10个来源，见不含题目答案的 [来源身份及哈希](SOURCE_SELECTION.md)，在站内按获准数据核实内容。先运行1条并核对结果，再决定批量。每来源设前测、独立冷启动重复和后测，学生各16次、上限1024；后测按作者函数遮蔽教师发言。原始输出、日志、来源和未决留在本机，绝不外传。

### 选择迁移题难度

公开 GitHub 仓库不包含题目数据。已导出的 499 组自建测试集保存在独立的私有数据仓库；应按站点允许的方式单向准备文件，不能把服务器中的数据和结果推回 GitHub。本程序可直接读取该数据集的 `data/test.jsonl`，也可读取本项目准备的 `paired_493.jsonl`（后者额外包含论文作者迁移题）。

使用 `--difficulty` 选择一档后，逐来源运行原题和该档迁移题；输出和独立来源回执记录来源 ID、候选题 ID、对应位置及历史计分，结果只保存在本机。参数含义是数据集**每组内相对位置**：

| 参数 | 输入数据 | 选择方式 |
| --- | --- | --- |
| `--difficulty hard` | 499 或 493 组自建题 | 历史学生正确数排序第 1 位 |
| `--difficulty medium` | 499 或 493 组自建题 | 排序第 2 位 |
| `--difficulty easy` | 499 或 493 组自建题 | 排序第 3 位 |
| `--difficulty paper` | 原论文平铺格式或 `paired_493.jsonl` | 作者原配迁移题，默认值 |

排序基于每题已有的 16 次学生判分；同分时保留数据集中原有次序并记录整组并列标记，不代表符合固定的绝对难度区间。只读取已有题目，不重新出题、不复用迁移题的旧作答，也不把自建题标为作者题。

代码和独立虚拟环境准备好后，在容器内分别运行（路径按站点挂载调整）：

```bash
SOURCE_FILE=/input/test.jsonl RESULTS_DIR=/results/easy /path/to/venv/bin/python app/launch.py --difficulty easy
SOURCE_FILE=/input/test.jsonl RESULTS_DIR=/results/medium /path/to/venv/bin/python app/launch.py --difficulty medium
SOURCE_FILE=/input/test.jsonl RESULTS_DIR=/results/hard /path/to/venv/bin/python app/launch.py --difficulty hard
# 若使用含论文题的 paired_493.jsonl，另可用 --difficulty paper。
```

这些命令每个都需要独立 `RESULTS_DIR`；程序将输入文件哈希、难度与环境写入运行配置，禁止不同难度复用同一目录，并使用目录锁阻止并发写入。每次运行会独立生成原题前测和教学对话，不能当作共享同一对话的严格配对试验。原型只导出待判分材料，不包含 Agnes 自动判分。若并行运行，必须给每个容器独立结果目录、独立网络命名空间和独立三张 GPU；当前程序每次教师占两卡、学生占一张，8 卡最多同时运行两组，第三组需等待。容器通过 `--gpus` 映射各自的物理卡，容器内部使用 0、1 给教师、2 给学生；不要只设置宿主的 `CUDA_VISIBLE_DEVICES`，因为程序会给子进程重新设置它。

环境准备完成后，若已把代码、虚拟环境、模型和输入放在站内受控目录，可用**单独**的评测容器采用 `--network none`，将代码/环境/模型/输入只读挂载、结果挂载到站内可写目录；按管理员分配传入3张GPU，并在容器内执行虚拟环境 Python 和仓库 `app/launch.py`。为只读文件系统提供可写 `/tmp`，设置 `HOME`、`HF_HOME`、`XDG_CACHE_HOME`、`VLLM_CACHE_ROOT`、`TRITON_CACHE_DIR`、`CUDA_CACHE_PATH` 到 `/tmp`，并设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_HUB_DISABLE_TELEMETRY=1`、`VLLM_NO_USAGE_STATS=1`。不挂 Docker socket、不发布服务端口、不将云API密钥放入评测容器。站内启动日志不得向外展示；只可用 `app/status_summary.py` 反馈聚合数字/固定状态码。

本原型默认结果为未判；`app/score_local.py` 可在站内对明确的单数字 boxed 结果做保守 math_verify 比对，多部分、描述型、模糊或解析失败均为 unknown；这不是旧 GPT v5，结果不可混为一个口径。用户允许的 Agnes 判分仅限上述三字段，由**另一个**受控出站组件处理，不能把完整对话/日志交给 Agnes；若没有经过审批及全局限流的组件，保留未决而非擅自联网。

## 许可与验证

`official/` 中必要提示词以及 `app/protocol.py` 的部分函数源自 [EduardoRL](https://github.com/eth-lre/EduardoRL) 本地固定提交 `e8c89b7293e65e632149571b16a1c2d4e33c77fe`，保留原许可 [LICENSE.author](LICENSE.author)。`app/dialogue.py` 基于本项目对评测流程的改写，其余为本地迁移草稿；分别保留出处及依赖各自许可。

CPU 合成数据与内存假服务测试已验证基础协议、幂等收据、回环限制、数字判分与脱敏错误输出；不代表镜像、真实模型或目标服务器已经跑通。不要将未验证的原型宣传为现有实验的直接迁移版。
