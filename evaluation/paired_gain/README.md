# `a-group-paired-gain`：配对 Bootstrap 归一化增益评测工具

用于比较两个模型（如 `base` 与 `a_group`）在同一批来源上的非对称归一化学习增益差值（`target - base`），并计算按来源有放回重采样 10,000 次的 95% 置信区间。

## 依赖

- Python >= 3.8（仅使用标准库，未使用 Python 3.11+ 专属语法）
- `numpy`

## 输入 CSV 格式

每行为一条对话的汇总记录，必须包含以下表头列：

| 列名 | 类型 | 说明 |
|---|---|---|
| `unit_id` | 字符串 | 来源或题目编号（输出中绝不打印） |
| `condition` | 字符串 | 模型/条件名，例如 `base`、`a_group` |
| `group` | 字符串 | 题集名称，例如 `author494`、`orig`、`easy`、`medium`、`hard` |
| `pre` | 浮点数 `[0, 1]` | 前测通过率 |
| `post_correct` | 整数 `>= 0` | 后测答对数 |
| `post_total` | 整数 `> 0` | 后测作答总数（包含 `post_unknown`） |
| `post_unknown` | 整数 `>= 0` | 后测未判出数（需满足 `post_correct + post_unknown <= post_total`） |

## 用法

```bash
python paired_gain.py \
  --input summary.csv \
  --base base \
  --target a_group \
  --draws 10000 \
  --seed 42 \
  --json-out report.json \
  --md-out report.md
```

主要参数：
- `--input`：输入 CSV 路径（必填）。
- `--base`：基线模型名称，默认 `base`。
- `--target`：待对比模型名称，默认 `a_group`。
- `--draws`：Bootstrap 重采样次数，默认 `10000`。
- `--seed`：基础随机种子，默认 `42`（内部按题集与口径用 SHA-256 派生独立可复现种子）。
- `--json-out`：可选，输出 JSON 报告路径。
- `--md-out`：可选，输出 Markdown 表格路径；未指定时直接打印到标准输出。
- `--no-high-pre-filter`：可选，跳过剔除 `pre >= 15/16` 的次要口径报告。

运行单元测试：

```bash
python -B -m unittest -v
```

当脚本不在本机仓库默认目录结构中（例如移入 cu04 工作树）时，可通过环境变量 `SOURCE_GAIN_PY` 指定参考实现 `source_gain.py` 的完整路径：

```bash
SOURCE_GAIN_PY=/path/to/source_gain.py python -B -m unittest -v
```

若未设置 `SOURCE_GAIN_PY`，测试会回退到本机默认路径 `../../代码/eduardo-a-group-prep/evaluation/agnes_rejudge/source_gain.py`；若两者均找不到或 `SOURCE_GAIN_PY` 指向不存在的文件，测试会直接失败并提示设置 `SOURCE_GAIN_PY`。

## 计算口径与输出说明

1. **前测一致性校验**：同一 `group`、同一 `unit_id` 下的所有行（无论属于哪个 `condition`），`pre` 必须一致（允许 `1e-9` 以内的浮点误差）。若发现不一致，脚本直接报错退出；报错信息仅包含题集名、出错来源个数以及前几处冲突的 CSV 行号，绝不输出 `unit_id`。
2. **后测通过率**：
   - 主处理（未决按错）：`post_correct / post_total`
   - 敏感性处理（排除未决）：`post_correct / (post_total - post_unknown)`；分母为 0 的行跳过并计数。
3. **公式 4 归一化增益**：`post > pre` 时为 `(post - pre) / (1 - pre)`，否则为 `post - pre`。
4. **来源等权重**：同一 `unit_id` 在同一 `(group, condition)` 下有多条对话时，先在该 `unit_id` 内取平均，再在配对来源上等权求均值。
5. **配对 Bootstrap**：仅保留 `base` 和 `target` 均存在有效值的 `unit_id`（同时报告各自未配对数量），按 `unit_id` 有放回重采样 10,000 次，取 2.5% 和 97.5% 分位数作为 95% CI，并报告 `P(diff > 0)`。
6. **主口径与次要口径**：主口径始终使用全部配对来源；次要口径额外报告剔除前测 `>= 15/16` 来源后的结果。
7. **零标识输出**：JSON 与 Markdown 输出仅包含汇总数字，不包含任何 `unit_id`。

## 与 `source_gain.py` 的关系

- **一致部分**：
  - `normalized_gain(pre, post)` 与 `代码/eduardo-a-group-prep/evaluation/agnes_rejudge/source_gain.py` 完全一致（已在单测中按网格逐点比对）。
  - 均采用“未决按错”为主处理、“排除未决”为敏感性对照、同来源内先均值再按来源等权有放回重采样 10,000 次、取 2.5%/97.5% 线性插值百分位数、并用 SHA-256 派生可复现子种子。
- **差异部分**（`source_gain.py` 面向单模型原始目录结构，`paired_gain.py` 面向双模型配对 CSV 汇总）：
  - `source_gain.py` 直接读取 `RESULTS.json`（固定 16 次采样）并支持 `pre`/`control` 两种基线；`paired_gain.py` 读取通用汇总 CSV 并计算 `target - base` 的配对差值 CI。
  - `source_gain.py` 在 `main()` 中硬性要求 `draws == 10000`；`paired_gain.py` 默认 `10000`，同时允许传入自定义 `--draws` 以便测试与模拟。
  - `source_gain.py` 使用标准库 `random.Random.choices`；`paired_gain.py` 使用 `numpy.random.default_rng` 做向量化重采样。
  - `source_gain.py` 仅在“简单迁移题”上按 `pre.correct >= 15` 剔除高分来源；`paired_gain.py` 在开启次要口径时按 `pre >= 15/16` 对各题集分别额外报告。
