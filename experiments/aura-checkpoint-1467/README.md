# 实验：checkpoint-1467 的 FP8 / INT8 量化

最终目标模型 `tt-simple-bench-all-64k-fix / v0-20261023-174822 / checkpoint-1467`
（`Qwen3_5ForConditionalGeneration`，合并后的 BF16，18.8 GB）的量化、评测和 demo 部署记录。

- 配置：[`configs/aura-checkpoint-1467.yaml`](../../configs/aura-checkpoint-1467.yaml)
- 结果：[`results/`](results/)（`report.md`、`comparison.{json,csv}`、`benchmark.json`、`generations.jsonl`）
- 运行时间：2026-11-09，hk01dgx014 的物理 GPU 1（H800），Slurm 作业 132149

## 评测设置

与线上推理客户端（`aura_v2_demo/main_dev/gateway/clients.py`）保持一致：

- `chat_template_kwargs: {enable_thinking: false}`：不开思考模式；
- `skip_special_tokens: false`：保留 `<|silent|>`、`<|respond|>` 等 special token；
- 贪心解码（temperature 0），每条最多 256 token，20 条评测提示词。

## 结果

| 指标 | BF16 | FP8 | INT8 |
|---|---:|---:|---:|
| 检查点大小 | 18.8 GB | 11.9 GB | 11.9 GB |
| 量化耗时 | — | 80 秒 | 23 分钟 |
| 单条请求延迟 | 1.84 秒 | 1.48 秒 | 1.36 秒 |
| 4 条一批吞吐（token/秒） | 532 | 654 | 727 |
| 与 BF16 完全一致 | 100% | 60% | 45% |
| 词级相似度 | 1.00 | 0.89 | 0.74 |
| JSON 任务 | 3/3 | 3/3 | 3/3 |
| 检查点验证 | — | pass_with_warnings | pass_with_warnings |

- 两个检查点都保留了全部 760 个源张量，359 个忽略层张量（含 333 个视觉张量）精度和形状不变。
- `pass_with_warnings` 只是因为 vLLM 环境里的 Transformers 4.57 不认识 `qwen3_5`；标准 vLLM 加载和生成都通过。
- FP8 更接近 BF16，INT8 吞吐略高但偏离更大。INT8 用 512 条 UltraChat 文本对话校准，和视频/主动对话场景不匹配。
- 显存一栏没有比较意义：vLLM 会按 `gpu_memory_utilization` 预分配。vLLM 日志显示两种量化模型的权重都占 11.23 GiB。

## 复现

在登录节点、仓库根目录下运行。所有脚本只用 `env.sh` 里的 `GPU_INDEX`（默认物理 GPU 1），启动前会核对 GPU UUID，GPU 被占用就退出。

```bash
mkdir -p runs/aura-checkpoint-1467 && nohup setsid srun --jobid=132149 --overlap -N1 -n1 experiments/aura-checkpoint-1467/run_all.sh > runs/aura-checkpoint-1467/run_all.log 2>&1 < /dev/null &
```

`run_all.sh` 依次执行 `quantize.sh fp8`、`quantize.sh int8`、`evaluate.sh`，任一步验证失败都会停止。
用 `nohup setsid` 启动，断开 SSH 也不会中断。

## 部署到 Aura demo

demo 跑在计算节点的 `aura` 容器里，容器内 `/workspace` 就是 `/home/dyvm6xra/dyvm6xrauser36/aura_demo/main`。

```bash
experiments/aura-checkpoint-1467/deploy_demo.sh int8 --replace tt-simple-bench-all-64k-fix-FP8
```

脚本会复制检查点、在 `aura_model_registry.json` 里注册（自动备份到 `backup/`），然后用 `--two_token` 启动：

| 模型 id | GPU | 端口 |
|---|---|---|
| `tt-simple-bench-all-64k-fix-FP8` | 1 | 8350 |
| `tt-simple-bench-all-64k-fix-INT8` | 1 | 8351 |

默认 `--gpu-memory-utilization 0.85` 下一张卡只放得下一个，所以换模型时要 `--replace` 掉正在运行的那个。

模型端口只在容器内可访问。用 `probe_demo.py` 对比原版和量化版的回复：

```bash
srun --jobid=132149 --overlap -N1 -n1 bash -c "docker exec -i aura python3 - BF16:8349:tt-simple-bench-all-64k-fix-demo INT8:8351:tt-simple-bench-all-64k-fix-INT8 < experiments/aura-checkpoint-1467/probe_demo.py"
```

two-token 模式的服务要求请求带 `stop_token_ids: [248070]`（`<|silent|>`）、`temperature: 0.2`、`min_tokens: 0`，否则返回 400。
