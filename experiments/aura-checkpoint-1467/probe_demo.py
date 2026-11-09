#!/usr/bin/env python3
"""Send the same chat prompts to two-token Aura model servers and print the replies side by side.

Ports are only reachable inside the container:
  srun --jobid=132149 --overlap -N1 -n1 bash -c \
    "docker exec -i aura python3 - BF16:8349:tt-simple-bench-all-64k-fix-demo INT8:8351:tt-simple-bench-all-64k-fix-INT8 \
     < experiments/aura-checkpoint-1467/probe_demo.py"
"""
import json
import sys
import urllib.error
import urllib.request

SILENT_TOKEN_ID = 248070
SYSTEM = ("You are receiving a live video stream where the final frame is the present moment. Respond only when a "
          "response is needed based on the user's message or the visual context. Otherwise, output `<|silent|>` "
          "to signify silence.")
QUESTIONS = ["你好，用一句话介绍一下你自己。", "1+1等于几？", "你是谁？"]

targets = [arg.split(":", 2) for arg in sys.argv[1:]] or [
    ["BF16", "8349", "tt-simple-bench-all-64k-fix-demo"],
]
for question in QUESTIONS:
    for label, port, model in targets:
        # Two-token servers require these sampling settings (same as the gateway).
        body = {"model": model,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
                "max_tokens": 60, "temperature": 0.2, "min_tokens": 0, "skip_special_tokens": False,
                "stop_token_ids": [SILENT_TOKEN_ID], "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            choice = json.load(urllib.request.urlopen(request, timeout=60))["choices"][0]
            print(f"{label:6s} {question!r} -> {choice['message']['content']!r} ({choice['finish_reason']})")
        except urllib.error.HTTPError as exc:
            print(f"{label:6s} {question!r} -> HTTP {exc.code}: {exc.read().decode()[:300]}")
