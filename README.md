# Aura quantization lab

This repository is a small harness for comparing a BF16 Qwen
checkpoint with LLM Compressor FP8 dynamic and calibrated INT8 W8A8 checkpoints.
FP8 is the first candidate for Hopper/H800. INT8 is a measured alternative, not
an assumption that lower precision will be faster or better.

The scripts never quantize in place. They refuse a non-empty output directory
unless `--overwrite` is explicit and reject source/output overlap. 
A quantized checkpoint is only marked complete after
file, tensor-index, Transformers, and (when installed) vLLM generation checks.

The Qwen3.5 embeddings and MTP prefix are ignored by default so they remain at
source precision; standard vLLM does not load an FP8 embedding scale.
Linear-attention layers remain quantization targets unless
`--ignore-linear-attention` is supplied. The branch's 3-D `conv1d` kernel is
always ignored because LLM Compressor's FP8 linear-weight scheme requires 2-D
weights; its actual 2-D projection weights are still quantized.

## Environment

Use separate environments if LLM Compressor and vLLM resolve incompatible
dependencies. Do not invent or pin versions before testing them on the cluster:

```bash
# First, acquire a GPU use srun command on hkstp cluster

python -m venv .venv-quantize
source .venv-quantize/bin/activate
pip install -r requirements-quantize.txt

python -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -r requirements-vllm.txt
```

Record the versions that actually work; every quantization run captures installed
package, CUDA, GPU, Git, command-line, and Slurm metadata.

## Slurm workflow

Run downloads, quantization, validation, and benchmarking inside a GPU allocation,
not on a login node:

```bash
cd /home/dyvm6xra/dyvm6xrauser60/aura-quantization
# source /path/to/quantization-venv/bin/activate
source .venv-quantize/bin/activate

python --version
nvidia-smi
echo "${SLURM_JOB_ID}"
echo "${CUDA_VISIBLE_DEVICES}"
```


Start with a dry run. It does not download/load a model, allocate a GPU, or create
output directories:

```bash
python scripts/run_experiment.py --config configs/smoke.yaml --dry-run
```

Then inspect and create FP8 in the quantization environment:

```bash
python scripts/inspect_model.py --config configs/smoke.yaml
python scripts/quantize_fp8.py --config configs/smoke.yaml \
  --skip-runtime-validation  # omit this when vLLM is in the same environment
```

INT8 uses only 32 UltraChat conversations in the smoke configuration. That tests
the pipeline; it is not enough for a final checkpoint:

```bash
python scripts/quantize_int8.py --config configs/smoke.yaml \
  --skip-runtime-validation  # omit this when vLLM is in the same environment
```

The INT8 recipe explicitly disables GPTQ activation ordering. LLM Compressor's
current static default is incompatible with W8A8's per-channel weight strategy
in standard vLLM; act-order is intended for grouped GPTQ schemes.

Activate the vLLM environment to validate, generate matched outputs, and benchmark:

```bash
source .venv-vllm/bin/activate

python scripts/validate_checkpoint.py \
  --source Qwen/Qwen3.5-9B \
  --checkpoint outputs-smoke/Qwen3.5-9B-FP8-Dynamic

export VLLM_WORKER_MULTIPROC_METHOD=spawn
python scripts/generate_samples.py --config configs/smoke.yaml
python scripts/benchmark.py --config configs/smoke.yaml --variants bf16,fp8,int8
python scripts/compare_results.py --config configs/smoke.yaml
```

Scripts default `VLLM_WORKER_MULTIPROC_METHOD=spawn` so vLLM does not fork after CUDA
is already initialized (which raises `Cannot re-initialize CUDA in forked subprocess`).

When runtime validation is requested but vLLM is unavailable, validation reports
`incomplete` rather than claiming that the checkpoint is valid. The explicit skip
above is for the split-environment workflow; the later vLLM validation must pass.
If standard vLLM loads and generates but its separately installed Transformers
cannot recognize Qwen3.5, the report is `pass_with_warnings` and retains that
Transformers error instead of treating it as a checkpoint failure.

Or run selected stages:

```bash
python scripts/run_experiment.py \
  --config configs/smoke.yaml \
  --steps inspect,fp8,int8,validate,generate,benchmark,compare
```

Already-quantized offline checkpoints are loaded without runtime quantization
flags. Their `config.json` determines the compression format. The scripts select
vLLM's Triton GDN prefill backend because these compute nodes do not expose
`nvcc` for FlashInfer's first-run JIT.

## Moving to the fine-tuned model

Use a finetuned BF16 checkpoint by changing one argument:

```bash
python scripts/run_experiment.py \
  --config configs/production-example.yaml \
  --model /models/our-finetuned-qwen-9b \
  --steps inspect,fp8,int8,validate,generate,benchmark,compare
```

For INT8, replace UltraChat with representative private-domain conversations and
keep calibration data separate from held-out evaluation prompts.


Run unit tests without downloading the 9B model:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
# or, after installing pytest:
pytest
```
