# Qwen 9B Quantization Experiment with LLM Compressor

## Objective

Create an experimental, easy-to-read Python project that uses **LLM Compressor** to quantize and compare:

1. BF16 baseline
2. FP8 dynamic quantization
3. INT8 W8A8 quantization

Initially use the public model:

```text
Qwen/Qwen3.5-9B
```

Treat the public checkpoint as our fine-tuned model during development. Later, we must be able to replace it with our merged fine-tuned checkpoint by changing only a configuration value or CLI argument.

The generated FP8 and INT8 checkpoints should be suitable for testing with:

- Standard vLLM
- NVIDIA Hopper/H800 GPUs
- A Slurm-managed compute environment

Keep the implementation simple and readable. Avoid unnecessary abstractions.

---

## Slurm Environment Commands

### Command 1: Enter the project and activate the installed environment

Replace `/PATH/` with the project directory:

```bash
cd /home/dyvm6xra/dyvm6xrauser60/BOOGU_hf_space
source ./venv/bin/activate
# Go back to project directory
cd /home/dyvm6xra/dyvm6xrauser60/aura-quantization
```

### Command 2: Enter a GPU job allocation

Replace `XXXX` with the active Slurm job ID:

```bash
srun --jobid=132211 --overlap --pty zsh
```

After entering the Slurm shell, activate the environment if it is not already active:

```bash
cd /home/dyvm6xra/dyvm6xrauser60/BOOGU_hf_space
source ./venv/bin/activate
# Go back to project directory
cd /home/dyvm6xra/dyvm6xrauser60/aura-quantization
```

Do not run the quantization scripts on a login node. Run GPU-dependent quantization, validation, and benchmarking inside the Slurm GPU allocation.

---

## Development Assumptions

- The public Qwen checkpoint represents our merged fine-tuned model during development.
- The real private checkpoint will later be a local merged BF16 checkpoint.
- Standard vLLM is the first compatibility and checkpoint-loading target.
- The target GPU is NVIDIA H800 or another Hopper GPU.
- FP8 is the expected primary candidate.
- INT8 is a calibrated comparison candidate.
- Quantization must not overwrite the source model.
- A checkpoint is not considered valid until it passes completeness and generation tests.

---

# Required Repository Structure

Generate the following project:

```text
qwen-quantization-lab/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-quantize.txt
├── requirements-vllm.txt
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── smoke.yaml
│   └── production-example.yaml
├── data/
│   ├── evaluation_prompts.jsonl
│   └── README.md
├── scripts/
│   ├── inspect_model.py
│   ├── quantize_fp8.py
│   ├── quantize_int8.py
│   ├── validate_checkpoint.py
│   ├── generate_samples.py
│   ├── benchmark.py
│   ├── compare_results.py
│   └── run_experiment.py
├── src/
│   └── qwen_quant_lab/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── model_utils.py
│       ├── calibration.py
│       ├── checkpoint.py
│       ├── inference.py
│       ├── metrics.py
│       └── reporting.py
├── tests/
│   ├── test_config.py
│   ├── test_calibration.py
│   ├── test_checkpoint_validation.py
│   └── test_reporting.py
└── outputs/
    └── .gitkeep
```

The main logic should be understandable by reading these files in order:

```text
scripts/quantize_fp8.py
scripts/quantize_int8.py
scripts/validate_checkpoint.py
scripts/benchmark.py
```

---

# Configuration

Use YAML configuration. CLI arguments must override YAML values.

Create `configs/default.yaml` similar to:

```yaml
model:
  source: Qwen/Qwen3.5-9B
  trust_remote_code: true
  dtype: auto

output:
  root: outputs
  fp8_dir: outputs/Qwen3.5-9B-FP8-Dynamic
  int8_dir: outputs/Qwen3.5-9B-INT8-W8A8
  results_dir: outputs/results

quantization:
  ignored_layers:
    - lm_head
    - "re:.*visual.*"
  ignore_linear_attention: false

fp8:
  scheme: FP8_DYNAMIC
  device: cuda:0
  max_workers: 1

int8:
  calibration_dataset: HuggingFaceH4/ultrachat_200k
  calibration_split: train_sft
  calibration_samples: 512
  max_sequence_length: 2048
  seed: 42
  smoothquant_strength: 0.8

evaluation:
  seed: 42
  temperature: 0.0
  max_tokens: 256
  max_model_len: 4096
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.90
  repeat_count: 3
  concurrency:
    - 1
    - 2
    - 4

serving:
  standard_vllm_smoke_test: true
```

Create `configs/smoke.yaml` similar to:

```yaml
model:
  source: Qwen/Qwen3.5-9B
  trust_remote_code: true
  dtype: auto

output:
  root: outputs-smoke
  fp8_dir: outputs-smoke/Qwen3.5-9B-FP8-Dynamic
  int8_dir: outputs-smoke/Qwen3.5-9B-INT8-W8A8
  results_dir: outputs-smoke/results

quantization:
  ignored_layers:
    - lm_head
    - "re:.*visual.*"
  ignore_linear_attention: false

fp8:
  scheme: FP8_DYNAMIC
  device: cuda:0
  max_workers: 1

int8:
  calibration_dataset: HuggingFaceH4/ultrachat_200k
  calibration_split: train_sft
  calibration_samples: 32
  max_sequence_length: 512
  seed: 42
  smoothquant_strength: 0.8

evaluation:
  seed: 42
  temperature: 0.0
  max_tokens: 64
  max_model_len: 2048
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.90
  repeat_count: 1
  concurrency:
    - 1

serving:
  standard_vllm_smoke_test: true
```

Print a clear warning when:

```text
calibration_samples < 128
```

Explain that a small calibration set is appropriate only for pipeline validation and not for producing the final production INT8 checkpoint.

---

# Quantization Methods

## FP8

Use LLM Compressor's model-free PTQ entry point with:

```text
FP8_DYNAMIC
```

The implementation should be based on:

```python
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub=config.model.source,
    save_directory=config.output.fp8_dir,
    scheme="FP8_DYNAMIC",
    ignore=ignored_layers,
    max_workers=config.fp8.max_workers,
    device=config.fp8.device,
)
```

FP8 dynamic quantization should use:

- Quantized FP8 weights
- Dynamic activation quantization
- No calibration dataset
- A compressed checkpoint written directly by LLM Compressor

Do not call `save_pretrained()` for the primary FP8 conversion if `model_free_ptq()` already writes the checkpoint.

## INT8

Use a calibrated W8A8 recipe with:

- SmoothQuant
- GPTQ
- INT8 weights
- Dynamic per-token INT8 activations
- Representative calibration data

The recipe should be based on:

```python
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import (
    SmoothQuantModifier,
)

recipe = [
    SmoothQuantModifier(
        smoothing_strength=config.int8.smoothquant_strength
    ),
    GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=ignored_layers,
    ),
]
```

Run quantization with:

```python
from llmcompressor import oneshot

oneshot(
    model=model,
    dataset=calibration_dataset,
    recipe=recipe,
    max_seq_length=config.int8.max_sequence_length,
    num_calibration_samples=config.int8.calibration_samples,
)
```

Then save the compressed model and processor:

```python
model.save_pretrained(
    config.output.int8_dir,
    save_compressed=True,
)

processor.save_pretrained(config.output.int8_dir)
```

For the development experiment, use UltraChat calibration data. Document that production INT8 quantization must use representative data from our private model's target domain.

---

# Ignored Modules

Start with:

```python
ignored_layers = [
    "lm_head",
    "re:.*visual.*",
]
```

Do not automatically ignore linear-attention layers.

Add this CLI option:

```bash
--ignore-linear-attention
```

When enabled, append:

```python
"re:.*linear_attn.*"
```

Before quantization, print:

- Every ignore pattern
- The modules matched by each pattern
- The total number of linear modules
- The number of modules selected for quantization
- The number of ignored vision modules
- Whether `lm_head` was found and ignored
- Whether linear-attention modules were found
- Whether linear-attention modules will be quantized

Fail before quantization if no target modules are found.

---

# Script Requirements

## 1. `scripts/inspect_model.py`

Implement a source-checkpoint inspection script.

It must:

1. Resolve a Hugging Face model ID or local checkpoint path.
2. Print the model class and architecture.
3. Confirm that the checkpoint appears to be the expected Qwen model.
4. List relevant linear modules.
5. Evaluate all ignore patterns.
6. Estimate source checkpoint disk size.
7. Verify that safetensors files or a safetensors index exist.
8. Detect whether the checkpoint appears to be a LoRA adapter.
9. Detect unsafe source/output path overlap.
10. Produce a machine-readable inspection report.

Example:

```bash
python scripts/inspect_model.py \
    --config configs/smoke.yaml
```

Fail if:

- No quantization targets are found.
- The model cannot be identified.
- Required checkpoint files are missing.
- An output directory would overwrite the source.
- The local path contains only an unmerged LoRA adapter.

---

## 2. `scripts/quantize_fp8.py`

Implement FP8 dynamic quantization using `model_free_ptq()`.

Requirements:

1. Print the effective configuration.
2. Refuse to overwrite a non-empty output directory unless `--overwrite` is supplied.
3. Support a Hugging Face model ID or local path.
4. Create experiment metadata.
5. Record:
   - Start time
   - End time
   - Elapsed time
   - Source model
   - Output directory
   - Quantization scheme
   - Ignored module patterns
   - Python version
   - Package versions
   - CUDA version
   - GPU name
   - GPU compute capability
6. Preserve or copy the tokenizer and processor assets.
7. Preserve the chat template and generation configuration.
8. Preserve vision-preprocessing assets when present.
9. Verify the resulting quantization configuration.
10. Automatically run checkpoint validation.

Save metadata as:

```text
<fp8-output>/experiment_metadata.json
```

Example:

```bash
python scripts/quantize_fp8.py \
    --config configs/smoke.yaml
```

---

## 3. `scripts/quantize_int8.py`

Implement calibrated INT8 W8A8 quantization.

Do not blindly assume that the model is text-only. Load the correct model class and processor for the checkpoint.

The implementation may use an appropriate auto class or the checkpoint-specific conditional-generation class, depending on installed Transformers support.

Prepare calibration data from UltraChat:

```python
from datasets import load_dataset

dataset = load_dataset(
    config.int8.calibration_dataset,
    split=config.int8.calibration_split,
)

dataset = dataset.shuffle(seed=config.int8.seed)
dataset = dataset.select(
    range(min(config.int8.calibration_samples, len(dataset)))
)
```

Apply the model's chat template:

```python
text = processor.apply_chat_template(
    example["messages"],
    tokenize=False,
    add_generation_prompt=False,
)
```

Tokenize the templated text without accidentally adding duplicate special tokens.

Requirements:

1. Print the complete effective configuration.
2. Warn when the calibration sample count is below 128.
3. Record calibration dataset name, split, seed, sample count, and maximum sequence length.
4. Make calibration deterministic where possible.
5. Report the number of examples successfully processed.
6. Report dropped or invalid calibration examples.
7. Apply SmoothQuant and GPTQ W8A8.
8. Save the compressed checkpoint.
9. Save the processor and tokenizer assets.
10. Write experiment metadata.
11. Automatically run checkpoint validation.

Example:

```bash
python scripts/quantize_int8.py \
    --config configs/smoke.yaml
```

---

## 4. `scripts/validate_checkpoint.py`

Implement comprehensive checkpoint validation.

A quantization command completing without an exception does not mean the checkpoint is valid.

### File validation

Confirm that the output contains:

- `config.json`
- Quantization metadata
- Safetensors files or a safetensors index
- Tokenizer files
- Processor files when required
- Chat-template information
- Vision-processor configuration when present in the source

### Tensor validation

Compare source and output tensor indexes.

Classify tensors as:

- Present in both checkpoints
- Expected transformed tensors
- Expected added scale tensors
- Missing source tensors
- Unexpected output tensors

Fail if important missing tensors match names such as:

```text
visual
embed_tokens
lm_head
norm
```

unless they are covered by an explicit and documented model-specific exception.

Ignored modules must remain present in their original precision. In particular, verify that ignored visual weights were preserved.

### MTP validation

Check whether the source checkpoint includes MTP tensors that are missing from the generated checkpoint.

When an officially supported helper is available, the script may use:

```python
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
```

The script must:

1. Detect the missing MTP condition.
2. Print the exact tensor names involved.
3. Copy only supported MTP tensors.
4. Re-run validation.
5. Record the repair in the validation report.

Do not silently repair arbitrary missing tensors.

### Load validation

Attempt:

1. Transformers configuration loading.
2. Tokenizer loading.
3. Processor loading.
4. Model metadata inspection.
5. Standard vLLM initialization.
6. At least one generation request.

Write:

```text
<checkpoint>/validation_report.json
```

The report must show pass, fail, skip, and error states separately.

Example:

```bash
python scripts/validate_checkpoint.py \
    --source Qwen/Qwen3.5-9B \
    --checkpoint outputs-smoke/Qwen3.5-9B-FP8-Dynamic
```

---

## 5. `scripts/generate_samples.py`

Generate deterministic outputs from:

- BF16
- FP8
- INT8

Use identical:

- Prompts
- Chat templates
- Seeds
- Maximum output tokens
- Sampling parameters

Use greedy decoding by default:

```text
temperature = 0
```

Include evaluation categories for:

- Factual questions
- Summarization
- Information extraction
- JSON output
- Short reasoning
- Multi-step reasoning
- Code generation
- Long-context recall
- Domain-style placeholder prompts
- Safety and refusal behavior

Write JSONL records similar to:

```json
{
  "model_variant": "fp8",
  "prompt_id": "reasoning_001",
  "category": "reasoning",
  "prompt": "Prompt text",
  "output": "Generated output",
  "latency_seconds": 1.23,
  "input_tokens": 100,
  "output_tokens": 80,
  "tokens_per_second": 65.0,
  "success": true,
  "error": null
}
```

---

## 6. `scripts/benchmark.py`

Benchmark BF16, FP8, and INT8 one at a time.

Release GPU memory and terminate the serving process between variants.

Measure:

- Checkpoint disk size
- Model startup time
- Idle GPU memory
- Peak GPU memory
- Time to first token
- End-to-end latency
- Output tokens per second
- Request throughput
- Successful request count
- Failed request count

Run:

1. Single-request latency tests.
2. Small-batch throughput tests.
3. Configurable concurrency tests.
4. Text-generation smoke tests.
5. Optional image-text smoke tests.
6. Standard vLLM checkpoint-loading tests.

Do not pass runtime quantization flags such as:

```text
--quantization fp8
```

or:

```text
--quantization int8
```

when loading an already quantized offline checkpoint. Allow the checkpoint's quantization configuration to determine the format.

Example:

```bash
python scripts/benchmark.py \
    --config configs/smoke.yaml \
    --variants bf16,fp8,int8 \
    --backend vllm
```

---

## 7. `scripts/compare_results.py`

Generate:

```text
outputs/results/comparison.json
outputs/results/comparison.csv
outputs/results/report.md
```

Include a comparison table:

| Metric | BF16 | FP8 | INT8 |
|---|---:|---:|---:|
| Checkpoint size | | | |
| Quantization time | N/A | | |
| Load time | | | |
| Idle GPU memory | | | |
| Peak GPU memory | | | |
| Time to first token | | | |
| End-to-end latency | | | |
| Output tokens/second | | | |
| Requests/second | | | |
| Exact match against BF16 | 100% | | |
| Normalized match against BF16 | 100% | | |
| Token similarity against BF16 | 100% | | |
| JSON parse success | | | |
| Checkpoint validation | | | |
| Standard vLLM test | | | |

Implement lightweight output-regression metrics:

- Exact string match
- Normalized string match
- Token-level edit similarity
- JSON parse success
- Expected-key presence
- Optional reference-answer checks

Clearly explain that lexical similarity is an experimental regression signal, not a complete model-quality evaluation.

---

## 8. `scripts/run_experiment.py`

Provide an end-to-end orchestrator.

Example:

```bash
python scripts/run_experiment.py \
    --config configs/smoke.yaml \
    --steps inspect,fp8,int8,validate,benchmark,compare
```

Support individual steps:

```bash
python scripts/run_experiment.py \
    --config configs/smoke.yaml \
    --steps fp8
```

Support:

```text
--model
--steps
--skip-existing
--overwrite
--dry-run
--ignore-linear-attention
```

The dry run must:

- Print all planned actions.
- Print source and output paths.
- Print the effective configuration.
- Avoid downloading the model.
- Avoid loading the model.
- Avoid allocating GPU memory.
- Avoid modifying output directories.

---

# Experiment Metadata

Every GPU-dependent operation should record:

- Slurm job ID from `SLURM_JOB_ID`
- Slurm node list from `SLURM_JOB_NODELIST`
- Slurm job partition from `SLURM_JOB_PARTITION`
- CUDA visible devices from `CUDA_VISIBLE_DEVICES`
- Hostname
- Python version
- PyTorch version
- Transformers version
- LLM Compressor version
- compressed-tensors version
- vLLM version
- CUDA runtime version
- NVIDIA driver version when available
- GPU name
- GPU UUID when available
- GPU compute capability
- Full command line
- Effective YAML configuration
- Git commit hash when running inside a Git repository

Do not fail merely because optional Slurm environment variables are unavailable. Record them as `null`.

---

# Checkpoint Safety Requirements

For every generated checkpoint:

1. Preserve all unquantized source tensors.
2. Preserve ignored visual weights.
3. Preserve tokenizer and processor assets.
4. Preserve special-token mappings.
5. Preserve the chat template.
6. Preserve generation configuration.
7. Preserve vision-preprocessing configuration.
8. Detect and explicitly handle MTP tensors.
9. Compare source and output tensor indexes.
10. Fail loudly if important tensors are missing.
11. Never report success solely because quantization completed.
12. Require an actual model-loading test.
13. Require at least one successful generation test.
14. Never overwrite the source checkpoint.
15. Refuse to overwrite existing output unless `--overwrite` is provided.

---

# Environment Requirements

Provide separate dependency files when quantization and vLLM have incompatible requirements:

```text
requirements-quantize.txt
requirements-vllm.txt
```

Expected packages include:

```text
torch
transformers
llmcompressor
compressed-tensors
datasets
accelerate
safetensors
pyyaml
pydantic
pandas
psutil
pynvml
pytest
vllm
```

Do not invent package versions. Document tested versions after installation and save them with each experiment.

The code should produce a clear compatibility error if an installed package does not provide a required API.

---

# Expected Usage

## Enter the allocated GPU job

```bash
srun --jobid=XXXX bash
```

## Enter the project and activate the environment

```bash
cd /PATH/
source ./venv/bin/activate
```

## Inspect the environment

```bash
python --version
nvidia-smi
echo "${SLURM_JOB_ID}"
echo "${CUDA_VISIBLE_DEVICES}"
```

## Inspect the model

```bash
python scripts/inspect_model.py \
    --config configs/smoke.yaml
```

## Run FP8 quantization

```bash
python scripts/quantize_fp8.py \
    --config configs/smoke.yaml
```

## Run INT8 quantization

```bash
python scripts/quantize_int8.py \
    --config configs/smoke.yaml
```

## Validate FP8

```bash
python scripts/validate_checkpoint.py \
    --source Qwen/Qwen3.5-9B \
    --checkpoint outputs-smoke/Qwen3.5-9B-FP8-Dynamic
```

## Validate INT8

```bash
python scripts/validate_checkpoint.py \
    --source Qwen/Qwen3.5-9B \
    --checkpoint outputs-smoke/Qwen3.5-9B-INT8-W8A8
```

## Benchmark with standard vLLM

```bash
python scripts/benchmark.py \
    --config configs/smoke.yaml \
    --variants bf16,fp8,int8 \
    --backend vllm
```

## Run the complete experiment

```bash
python scripts/run_experiment.py \
    --config configs/smoke.yaml \
    --steps inspect,fp8,int8,validate,benchmark,compare
```

---

# Migration to Our Fine-Tuned Model

The model source must be replaceable using one CLI argument:

```bash
--model /models/our-finetuned-qwen-9b
```

Example:

```bash
python scripts/run_experiment.py \
    --config configs/production-example.yaml \
    --model /models/our-finetuned-qwen-9b \
    --steps inspect,fp8,int8,validate,benchmark,compare
```

Assume the private checkpoint is a merged BF16 checkpoint.

If the path contains only a LoRA adapter:

1. Do not silently quantize it.
2. Stop with a clear error.
3. Explain that the adapter must first be merged with its base model.
4. Alternatively, explain that quantized-base-plus-LoRA serving requires a separate experiment.

For the private model:

- Replace or supplement UltraChat with private-domain calibration examples.
- Keep calibration and evaluation datasets separate.
- Use held-out prompts testing behaviors introduced by fine-tuning.
- Repeat text and multimodal tests.
- Require successful standard vLLM testing before production promotion.

---

# Acceptance Criteria

The project is complete only when:

- [ ] The source checkpoint can be inspected.
- [ ] An FP8 dynamic checkpoint can be generated.
- [ ] An INT8 W8A8 checkpoint can be generated.
- [ ] Both checkpoints contain quantization metadata.
- [ ] Both checkpoints preserve tokenizer assets.
- [ ] Both checkpoints preserve processor assets.
- [ ] Ignored visual weights are verified as present.
- [ ] Missing source tensors are detected.
- [ ] MTP tensors are handled or explicitly reported.
- [ ] Both checkpoints receive a standard vLLM load test.
- [ ] At least one generation request succeeds for each valid checkpoint.
- [ ] BF16, FP8, and INT8 use identical evaluation settings.
- [ ] Disk size, GPU memory, latency, and throughput are compared.
- [ ] Output-regression metrics are generated.
- [ ] A Markdown report is generated.
- [ ] Configuration is available through YAML and CLI overrides.
- [ ] The source can be replaced with a local fine-tuned checkpoint.
- [ ] Unit tests do not download or load the full 9B model.
- [ ] Smoke-test and production settings are clearly separated.
- [ ] Slurm job metadata is captured in experiment metadata.
- [ ] The README includes Slurm execution instructions.

---

# Implementation Priority

Implement and test in this order:

```text
1. Configuration loading and CLI overrides
2. Environment and Slurm metadata collection
3. Source checkpoint inspection
4. FP8 dynamic quantization
5. FP8 checkpoint validation
6. FP8 standard vLLM smoke test
7. INT8 calibration-data preparation
8. INT8 W8A8 quantization
9. INT8 checkpoint validation
10. INT8 standard vLLM smoke test
11. BF16, FP8, and INT8 sample generation
12. Performance benchmarking
13. Result comparison and Markdown reporting
14. Migration documentation for our fine-tuned checkpoint
```

FP8 is the expected primary candidate on Hopper/H800 hardware. INT8 should be implemented and measured as a calibrated alternative rather than assumed to be better.

Begin by creating the complete repository structure, configuration models, README, and dry-run support. Then implement FP8 before INT8. Keep every script experimental, explicit, and easy to read.