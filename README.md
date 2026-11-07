# Aura 量化实验仓

用 [LLM Compressor](https://github.com/vllm-project/llm-compressor) 对 Qwen 检查点做离线量化，并在标准 vLLM 上对比三个变体：

| 变体 | 方案 | 是否需要校准数据 | 定位 |
|---|---|---|---|
| BF16 | 原始检查点 | — | 基线 |
| FP8 | `FP8_DYNAMIC`（FP8 权重 + 动态 FP8 激活），`model_free_ptq` | 否 | Hopper / H800 上的**首选候选** |
| INT8 | W8A8（SmoothQuant + GPTQ，INT8 权重 + 动态逐 token INT8 激活），`oneshot` | 是 | 实测对照方案，不预设它更快或更好 |

开发阶段用公开的 `Qwen/Qwen3.5-9B` 代替我们的微调模型；之后只需改一个配置项或命令行参数 `--model` 就能切换到合并后的私有 BF16 检查点。

目标环境：标准 vLLM + NVIDIA Hopper/H800 + Slurm 集群。

## 整体流程

```text
inspect（检查源模型）
   ↓
fp8 / int8（量化，写出新检查点，绝不原地修改源模型）
   ↓
validate（文件、张量、忽略层精度、Transformers、vLLM 加载与生成）
   ↓
generate（三个变体用相同提示词和采样参数生成输出）
   ↓
benchmark（逐个变体测加载时间、显存、延迟、吞吐）
   ↓
compare（汇总成 comparison.json / comparison.csv / report.md）
```

每一步都是 `scripts/` 下的独立脚本，也可以用 `scripts/run_experiment.py` 按顺序编排。

## 目录结构

```text
aura-quantization/
├── configs/
│   ├── default.yaml              # 完整实验配置（512 条校准样本）
│   ├── smoke.yaml                # 冒烟测试：32 条校准样本，只验证流程能跑通
│   └── production-example.yaml   # 切换到私有微调模型的示例
├── data/
│   ├── evaluation_prompts.jsonl  # 默认评测提示词（20 条，10 个类别，中英文）
│   └── README.md                 # 提示词格式说明
├── scripts/
│   ├── inspect_model.py          # 检查源检查点
│   ├── quantize_fp8.py           # FP8 动态量化
│   ├── quantize_int8.py          # INT8 W8A8 校准量化
│   ├── validate_checkpoint.py    # 检查点验证
│   ├── generate_samples.py       # 生成对齐输出
│   ├── benchmark.py              # 性能测试
│   ├── compare_results.py        # 汇总报告
│   └── run_experiment.py         # 端到端编排
├── src/qwen_quant_lab/
│   ├── config.py                 # YAML 配置 + CLI 覆盖 + 合法性检查
│   ├── constants.py
│   ├── model_utils.py            # 环境元数据、路径安全、模型加载、忽略层匹配
│   ├── inspection.py             # 源模型检查（只读 config 和张量索引，不加载权重）
│   ├── calibration.py            # 校准数据准备
│   ├── checkpoint.py             # 检查点验证与 MTP 修复
│   ├── inference.py              # vLLM 离线生成
│   ├── metrics.py                # 文本回归指标
│   └── reporting.py              # 汇总与报告输出
├── tests/                        # 单元测试，不下载、不加载 9B 模型
├── requirements-quantize.txt     # 量化环境依赖
├── requirements-vllm.txt         # vLLM 环境依赖（vllm==0.19.0）
└── requirements.txt              # 量化依赖 + pynvml + pytest
```

阅读主流程建议按这个顺序：`quantize_fp8.py` → `quantize_int8.py` → `validate_checkpoint.py` → `benchmark.py`。

## 安全设计

- **绝不覆盖源模型。** 源目录与输出目录相同或互相嵌套时直接报错。
- **默认拒绝覆盖已有产物。** 输出目录非空时必须显式加 `--overwrite`。
- **量化完成 ≠ 检查点可用。** 只有通过文件、张量索引、Transformers 和 vLLM 生成检查后，才会把状态标为 `complete`。
- **LoRA 适配器不会被直接量化。** 如果路径里只有 adapter，会报错并提示先与 BF16 基座合并。
- **全程留档。** 每次量化都会记录包版本、CUDA、GPU、Git 提交、命令行、有效配置和 Slurm 信息。

## 环境安装

LLM Compressor 和 vLLM 的依赖可能冲突，建议分成两个虚拟环境。除 `vllm==0.19.0` 外没有锁定版本，请在集群上实测后记录可用版本（每次量化的 `experiment_metadata.json` 也会自动记录）。

```bash
python -m venv .venv-quantize
source .venv-quantize/bin/activate
pip install -r requirements-quantize.txt
```

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -r requirements-vllm.txt
```

项目要求 Python ≥ 3.10。

## 在 Slurm 上运行

下载模型、量化、验证、benchmark 都必须在 GPU 分配内执行，**不要在登录节点上跑**。先进入已分配的作业：

```bash
srun --jobid=<作业ID> --overlap --pty zsh
```

进入项目并确认环境：

```bash
cd /home/dyvm6xra/dyvm6xrauser60/aura-quantization
source .venv-quantize/bin/activate

python --version
nvidia-smi
echo "${SLURM_JOB_ID}"
echo "${CUDA_VISIBLE_DEVICES}"
```

所有脚本都应**在仓库根目录下运行**：配置里的 `outputs/...` 等路径是相对当前目录解析的。

## 快速开始（smoke 配置）

### 1. Dry run

不下载、不加载模型，不占用 GPU，也不创建输出目录；会打印有效配置和每一步要执行的命令，并做启动前检查（例如评测提示词文件是否存在）：

```bash
python scripts/run_experiment.py --config configs/smoke.yaml --dry-run
```

### 2. 检查源模型并做 FP8 量化（量化环境）

```bash
python scripts/inspect_model.py --config configs/smoke.yaml
```

```bash
python scripts/quantize_fp8.py --config configs/smoke.yaml --skip-runtime-validation
```

如果 vLLM 和量化工具装在同一个环境里，去掉 `--skip-runtime-validation`，量化后会直接做 vLLM 加载与生成验证。

### 3. INT8 量化（量化环境）

smoke 配置只用 32 条 UltraChat 对话做校准，只能验证流程，**不能用来产出最终的 INT8 检查点**（少于 128 条时脚本会打印警告）：

```bash
python scripts/quantize_int8.py --config configs/smoke.yaml --skip-runtime-validation
```

### 4. 验证、生成、benchmark、汇总（vLLM 环境）

```bash
source .venv-vllm/bin/activate
```

```bash
python scripts/validate_checkpoint.py --source Qwen/Qwen3.5-9B --checkpoint outputs-smoke/Qwen3.5-9B-FP8-Dynamic
```

```bash
python scripts/validate_checkpoint.py --source Qwen/Qwen3.5-9B --checkpoint outputs-smoke/Qwen3.5-9B-INT8-W8A8
```

```bash
python scripts/generate_samples.py --config configs/smoke.yaml
```

```bash
python scripts/benchmark.py --config configs/smoke.yaml --variants bf16,fp8,int8
```

```bash
python scripts/compare_results.py --config configs/smoke.yaml
```

在拆分环境的流程里，量化阶段跳过的 vLLM 验证**必须**在这一步补上并通过。

## 一键编排：`run_experiment.py`

```bash
python scripts/run_experiment.py \
  --config configs/smoke.yaml \
  --steps inspect,fp8,int8,validate,generate,benchmark,compare
```

| 参数 | 说明 |
|---|---|
| `--config` | YAML 配置文件，默认 `configs/default.yaml` |
| `--model` | 覆盖 `model.source`，可以是 HF 模型 ID 或本地路径 |
| `--steps` | 逗号分隔的步骤，可选 `inspect,fp8,int8,validate,generate,benchmark,compare` |
| `--dry-run` | 只打印计划，不执行 |
| `--overwrite` | 允许覆盖非空的量化输出目录 |
| `--skip-existing` | 复用已有量化产物（见下） |
| `--ignore-linear-attention` | 额外忽略所有 `linear_attn` 模块，不量化 |
| `--skip-runtime-validation` | 量化和 validate 步骤跳过 vLLM 加载与生成 |

`--skip-existing` 只会在以下条件**同时满足**时跳过量化：

1. 输出目录里有 `experiment_metadata.json`；
2. 其中 `status` 是 `complete` 或 `quantized_pending_runtime_validation`；
3. 记录的 `source_model` 与当前 `model.source` 一致；
4. 记录的忽略层列表与当前配置一致。

不满足时会说明原因并停止；如果同时传了 `--overwrite`，则直接重建。

## 配置说明

CLI 参数优先级高于 YAML。加载时会做基本的合法性检查，未知的顶层键会报错。

| 配置项 | 含义 |
|---|---|
| `model.source` | 源模型：HF ID 或本地合并后的 BF16 检查点 |
| `model.trust_remote_code` | 加载时是否信任远程代码，会传给所有阶段（包括独立 validate） |
| `model.dtype` | INT8 量化时加载模型的精度，`auto` 保持检查点原精度，避免 BF16 被展开为 FP32 |
| `output.fp8_dir` / `int8_dir` | 量化产物目录 |
| `output.results_dir` | 生成、benchmark、报告的输出目录 |
| `output.root` | `inspect` 报告的默认写入目录 |
| `quantization.ignored_layers` | 不量化的模块，`re:` 前缀表示正则 |
| `quantization.ignore_linear_attention` | 等同于 `--ignore-linear-attention` |
| `fp8.scheme` | 只支持 `FP8_DYNAMIC` |
| `fp8.device` / `max_workers` | `model_free_ptq` 的设备与并行数 |
| `int8.calibration_*` | 校准数据集、split、样本数 |
| `int8.max_sequence_length` | 校准样本最大长度 |
| `int8.seed` | 校准数据打乱与 torch 随机种子 |
| `int8.smoothquant_strength` | SmoothQuant 强度，范围 0–1 |
| `evaluation.temperature` | 默认 0，即贪心解码 |
| `evaluation.max_tokens` / `max_model_len` | 生成长度与 vLLM 上下文长度 |
| `evaluation.tensor_parallel_size` / `gpu_memory_utilization` | vLLM 参数 |
| `evaluation.concurrency` | benchmark 中每次 `generate()` 提交的提示词条数（离线批大小） |
| `evaluation.repeat_count` | 每个批大小重复次数，至少为 1 |
| `serving.standard_vllm_smoke_test` | 设为 `false` 时，量化和 validate 步骤跳过 vLLM 加载与生成验证，效果同 `--skip-runtime-validation` |

## 量化方案细节

### FP8

调用 `llmcompressor.model_free_ptq(scheme="FP8_DYNAMIC")`，不需要校准数据，不把整个模型加载进 Transformers。检查点由 LLM Compressor 直接写出，脚本随后补存 tokenizer、processor 和 `generation_config`，并确认 `config.json` 里有量化元数据。

### INT8 W8A8

1. 按检查点类型选择模型类：多模态用 `AutoModelForImageTextToText`，否则用 `AutoModelForCausalLM`。
2. 从 UltraChat 取样本，固定种子打乱，用模型自己的 chat template 渲染成文本。分词交给 LLM Compressor 做，避免重复添加 BOS/EOS。
3. 配方：`SmoothQuantModifier` + `GPTQModifier(scheme="W8A8", actorder=None)`。
4. `save_pretrained(save_compressed=True)` 保存压缩检查点和 processor。

**为什么关闭 GPTQ act-order：** LLM Compressor 当前默认的静态 act-order 与 W8A8 的逐通道权重策略不兼容，标准 vLLM 会拒绝加载；act-order 是为分组 GPTQ 设计的。

### 默认忽略层

| 模式 | 原因 |
|---|---|
| `lm_head` | 输出层保持原精度 |
| `re:.*embed_tokens.*` | 标准 vLLM 不加载 FP8 embedding 的 scale |
| `re:.*visual.*` | 视觉模块保持原精度 |
| `re:^mtp\..*` | MTP（多 token 预测）前缀保持原精度 |
| `re:.*linear_attn\.conv1d.*` | 该卷积核是三维张量，LLM Compressor 的 FP8 线性权重方案要求二维 |

线性注意力（`linear_attn`）的二维投影权重**默认仍会量化**；如需全部保留原精度，加 `--ignore-linear-attention`。

量化前会打印每个忽略模式匹配到的模块、线性模块总数、实际量化的模块数，以及 `lm_head`、视觉模块、线性注意力的处理情况。如果没有任何可量化模块，会在量化前失败。

## 检查点验证

`validate_checkpoint.py` 的检查项：

| 类别 | 检查内容 |
|---|---|
| 文件 | `config.json`、量化元数据、safetensors、tokenizer、chat template；源模型是多模态时还要求 processor 配置 |
| 张量索引 | 对比源与输出的张量名称；任何源张量缺失都判为失败，并单独列出 `visual`、`embed_tokens`、`lm_head`、`norm` 等重要张量 |
| 分片完整性 | 索引引用的每个分片都必须存在；本地检查点会读取每个分片的 safetensors 头，确认索引里的张量确实在对应分片里，能发现缺失或截断的分片 |
| 忽略层精度 | 被忽略的张量和所有视觉张量在输出中必须存在，且 dtype、shape 与源一致 |
| MTP | 发现输出缺少 `mtp.*` 张量时，用 `compressed_tensors` 官方的 `save_mtp_tensors_to_checkpoint` 补回，重新校验并写入报告（可用 `--no-repair-mtp` 关闭）。不会修补其他任意缺失张量 |
| Transformers | 加载 config、tokenizer、processor |
| vLLM | 标准 vLLM 初始化，并至少成功生成一次 |

结果写入 `<检查点>/validation_report.json`，每一项分别记录 `pass` / `fail` / `skip` / `error`。总体状态：

| `overall_status` | 含义 |
|---|---|
| `pass` | 全部通过 |
| `pass_with_warnings` | vLLM 加载和生成通过，但单独安装的 Transformers 无法识别该模型（例如不认识 Qwen3.5）；报告保留该错误，不算检查点失败 |
| `incomplete` | 要求做运行时验证，但 vLLM 未安装 |
| `fail` | 存在结构性失败，或 vLLM 初始化/生成失败 |

验证结果同步到 `experiment_metadata.json` 的 `status`：`complete`、`quantized_pending_runtime_validation`（结构检查通过但尚未做 vLLM 验证）或 `invalid`。

## 评测与报告

### 生成：`generate_samples.py`

三个变体使用相同的提示词、源模型的 chat template、种子、最大输出长度和采样参数，默认贪心解码。提示词默认读取 `data/evaluation_prompts.jsonl`，格式见 [data/README.md](data/README.md)，也可以用 `--prompts` 指定。结果写入 `<results_dir>/generations.jsonl`。

所有提示词在一次 `generate()` 中批量完成，因此单条记录**无法得到真实的请求延迟**。记录里给出的是 `batch_elapsed_seconds`（整批耗时）和 `amortized_latency_seconds`（整批耗时 ÷ 条数），后者只是摊销值。

### 性能：`benchmark.py`

每个变体单独启动 vLLM，测完后关闭引擎、释放显存，再测下一个。已量化的检查点不传 `--quantization` 之类的运行时参数，由检查点自己的配置决定格式。

- 正式计时前先做一次预热，避免把 CUDA graph 捕获、kernel 编译算进去。
- 显存用 NVML 后台线程在生成期间持续采样，`peak_gpu_memory_bytes` 是真实峰值（相对加载前）。注意 vLLM 会按 `gpu_memory_utilization` 预分配 KV cache，所以这个数字主要反映预分配量；只采样第一张可见 GPU。
- 按批大小分别汇总（`by_batch_size`）。`single_request_latency_seconds` 只取批大小为 1 的结果，这是唯一能等同于请求延迟的情况；吞吐报告最大批大小下的值。
- TTFT 为 `null`：离线批量接口不提供流式首 token 时间。
- 这是**离线批量推理**，`concurrency` 表示每批提交的条数，不等于线上服务的并发客户端数。线上选型还需要单独做服务端压测。

### 汇总：`compare_results.py`

生成 `comparison.json`、`comparison.csv`、`report.md`，内容包括检查点大小、量化耗时、加载时间、显存、延迟、吞吐、回归指标、验证状态。

回归指标（都与 BF16 输出对比）：

| 指标 | 说明 |
|---|---|
| `generation_success_rate` | 候选变体在基线提示词上的成功生成比例，缺失或失败都算失败 |
| `exact_match` | 完全一致 |
| `normalized_match` | 忽略大小写和标点后一致 |
| `token_similarity` | 词级编辑相似度 |
| `json_parse_success` | 仅统计带 `expected_keys` 的 JSON 任务：输出能否解析为 JSON |
| `expected_key_presence` | 同上：解析结果是否包含所有期望键，解析失败计为失败 |

规则：候选变体生成失败或缺失时，一律计为不匹配；BF16 自身失败的提示词没有有效参照，不参与文本相似度统计。

> 这些文本指标只是轻量的回归信号，**不是完整的模型质量评测**。

## 输出文件

| 路径 | 内容 |
|---|---|
| `<output.root>/inspection_report.json` | 源模型检查报告 |
| `<检查点>/experiment_metadata.json` | 量化元数据：耗时、环境、Slurm、有效配置、校准统计、状态 |
| `<检查点>/validation_report.json` | 验证报告 |
| `<results_dir>/generations.jsonl` | 三个变体的生成记录 |
| `<results_dir>/benchmark.json` | 性能测试原始结果 |
| `<results_dir>/comparison.{json,csv}`、`report.md` | 对比报告 |

## 切换到微调模型

假设私有模型是**已经合并好的 BF16 检查点**：

```bash
python scripts/run_experiment.py \
  --config configs/production-example.yaml \
  --model /models/our-finetuned-qwen-9b \
  --steps inspect,fp8,int8,validate,generate,benchmark,compare
```

注意：

- 切换 `--model` 时**输出目录不会自动变化**。请在配置里给新模型单独设置 `fp8_dir` / `int8_dir`，`production-example.yaml` 已经这样做了。
- 如果路径里只有 LoRA adapter，脚本会拒绝量化。需要先与基座合并；"量化基座 + LoRA" 的部署方式需要另做实验。
- INT8 校准请用目标领域的代表性对话替换或补充 UltraChat。
- 校准数据和评测提示词必须分开。把 `data/evaluation_prompts.jsonl` 里的 `domain_placeholder` 换成能覆盖微调新行为的保留样本。
- 上线前必须通过标准 vLLM 测试。

## 单元测试

测试不下载、不加载 9B 模型，只需要 PyYAML：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

或者安装 pytest 后：

```bash
pytest
```

## 常见问题

**`Cannot re-initialize CUDA in forked subprocess`**
vLLM 在 Linux 上默认用 fork 启动 worker，如果父进程已经初始化过 CUDA 就会报这个错。脚本已默认设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`；手动调用 vLLM 时也建议设置：

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

**为什么固定 `gdn_prefill_backend="triton"`？**
计算节点没有 `nvcc`，FlashInfer 首次运行需要 JIT 编译会失败，因此选用 vLLM 的 Triton GDN prefill 后端。

**validate 报 `pass_with_warnings`**
通常是 vLLM 环境里的 Transformers 版本不认识 Qwen3.5。只要 vLLM 加载和生成通过，检查点本身没问题。

**validate 报 `incomplete`**
要求做运行时验证，但当前环境没有安装 vLLM。到 vLLM 环境重新运行 validate。

## 已知限制

- benchmark 是离线批量测试，没有 TTFT，也不是线上服务压测。
- 图文多模态冒烟测试尚未实现。
- 评测集很小，文本相似度不能代替任务级的质量评测。
- 显存只采样第一张可见 GPU；张量并行时是单卡数值。
- 真实的压缩率、速度收益和精度损失需要在 H800 上实测，本仓库不预设结论。
