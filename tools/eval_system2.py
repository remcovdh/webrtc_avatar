"""Calibrate retrieval and compare System 2 language models.

Prints the best retrieval score for questions the knowledge folder can and
cannot answer (to pick `system2.min_score`), then each model's answers, speed
and GPU memory. This is the comparison that chose Llama 3.2 3B. Models load one
at a time next to the running conductor, so pause the avatar first
(see tools/README.md):
    docker compose run --rm --no-deps -T conductor python - [repo::file.gguf ...] < tools/eval_system2.py
"""

import subprocess
import sys
import time

sys.path.insert(0, "/workspace")
import torch  # noqa: F401,E402  (load before llama.cpp)

from knowledge import MarkdownKnowledge  # noqa: E402
from system2 import LlamaSystem2  # noqa: E402

MODELS = sys.argv[1:] or [
    "bartowski/Llama-3.2-3B-Instruct-GGUF::Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "unsloth/Qwen3-4B-Instruct-2507-GGUF::Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
]
ANSWERABLE = [
    "Why is the avatar so fast?",
    "Hoe luistert de avatar naar mij?",
    "What happens when I correct the avatar?",
    "Waar draait dit project op?",
    "Is my microphone audio saved?",
]
UNANSWERABLE = [
    "What is a balloon made of?",
    "Wie is de minister-president van Nederland?",
    "What is the capital of France?",
    "Kun je een grap vertellen over katten?",
]
# Related to the notes but not answered by them: a faithful model says it doesn't know.
TRAP = "Who built the avatar and when did the project start?"


def gpu_used() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    return int(out.strip() or 0)


knowledge = MarkdownKnowledge()
print("== retrieval scores (best passage)")
for question in ANSWERABLE + UNANSWERABLE + [TRAP]:
    best = knowledge.search(question, 1)[0]
    kind = "ANS " if question in ANSWERABLE else "TRAP" if question == TRAP else "NO  "
    print(f"  {kind} {best.score:.3f}  {question[:50]:50} -> {best.source}")

for model in MODELS:
    before = gpu_used()
    started = time.perf_counter()
    system2 = LlamaSystem2(model)
    load = time.perf_counter() - started
    system2.warm_up()
    print(f"\n== {model.split('::')[-1]}: load {load:.1f}s, GPU +{gpu_used() - before} MiB")
    times = []
    for question in ANSWERABLE + [TRAP]:
        text, detail = system2.answer(question, knowledge.search(question, 3))
        times.append(detail["llm_ms"])
        print(f"  [{detail['llm_ms']:5d} ms, {detail['tokens']} tok] {question[:40]:40} -> {text}")
    print(f"  median {sorted(times)[len(times) // 2]} ms")
    del system2
