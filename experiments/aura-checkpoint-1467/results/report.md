# Quantization experiment report

| Metric | BF16 | FP8 | INT8 |
|---|---:|---:|---:|
| Checkpoint Size Bytes | 18839846975 | 11924285565 | 11924157536 |
| Quantization Time Seconds | N/A | 79.79 | 1400 |
| Load Time Seconds | 53.1 | 57.96 | 52.14 |
| Idle Gpu Memory Bytes | 80869130240 | 80198828032 | 81308221440 |
| Peak Gpu Memory Bytes | 80871227392 | 80205119488 | 81312415744 |
| Time To First Token Seconds | N/A | N/A | N/A |
| Single Request Latency Seconds | 1.837 | 1.482 | 1.355 |
| Max Batch Size | 4 | 4 | 4 |
| Max Batch Output Tokens Per Second | 531.7 | 653.6 | 727.3 |
| Max Batch Requests Per Second | 2.093 | 2.553 | 2.841 |
| Generation Success Rate | 1 | 1 | 1 |
| Compared Prompts | 20 | 20 | 20 |
| Exact Match Against Bf16 | 1 | 0.6 | 0.45 |
| Normalized Match Against Bf16 | 1 | 0.65 | 0.45 |
| Token Similarity Against Bf16 | 1 | 0.891 | 0.7401 |
| Json Prompts | 3 | 3 | 3 |
| Json Parse Success | 1 | 1 | 1 |
| Expected Key Presence | 1 | 1 | 1 |
| Checkpoint Validation | N/A | pass_with_warnings | pass_with_warnings |
| Standard Vllm Test | pass | pass | pass |

> Exact and token-level similarity are lightweight regression signals. They are not a complete model-quality evaluation.
> Failed or missing generations count as mismatches. JSON metrics only cover prompts with `expected_keys`.
> Throughput comes from offline vLLM batches (batch size = prompts per generate() call), not an online serving load test.
