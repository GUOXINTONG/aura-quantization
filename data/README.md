# 评测提示词

`evaluation_prompts.jsonl` 是 `scripts/generate_samples.py` 默认读取的提示词集，
BF16 / FP8 / INT8 三个变体使用完全相同的提示词、聊天模板和采样参数生成输出，
再由 `scripts/compare_results.py` 计算与 BF16 的回归指标。

每行一个 JSON 对象：

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 唯一 ID，用于在不同变体间对齐同一条提示词 |
| `prompt` | 是 | 用户消息原文，生成时会套用源模型的 chat template |
| `category` | 否 | 类别，例如 `factual`、`json`、`code`、`safety` |
| `expected_keys` | 否 | 仅 JSON 类任务使用：输出必须能解析为 JSON 且包含这些键 |

当前覆盖的类别：事实问答、摘要、信息抽取、JSON 输出、短推理、多步推理、
代码生成、长上下文召回、领域占位、安全拒答，中英文混合。

注意：

- 这只是开发期的小型回归集，不是完整的模型质量评测。
- `domain_placeholder` 类别需要替换为微调模型目标领域的真实请求。
- 评测提示词必须与 INT8 校准数据**分开**，不要把校准样本放进来。
