"""Calibrate retrieval and compare System 2 language models.

Prints the best retrieval score for questions the knowledge folder can and
cannot answer (to pick `system2.min_score`), then each model's answers, speed
and GPU memory. This is the comparison that chose Granite 4.0 1B. Models load one
at a time next to the running conductor, so pause the avatar first
(see tools/README.md):
    docker compose run --rm --no-deps -T conductor python - [repo::file.gguf ...] < tools/eval_system2.py
"""

import subprocess
import sys
import time

import torch  # noqa: F401,E402  (load before llama.cpp)

from conductor.knowledge import MarkdownKnowledge  # noqa: E402
from conductor.system2 import LlamaSystem2  # noqa: E402

MODELS = sys.argv[1:] or [
    "ibm-granite/granite-4.0-1b-GGUF::granite-4.0-1b-Q4_K_M.gguf",
    "unsloth/Qwen3.5-2B-GGUF::Qwen3.5-2B-Q4_K_M.gguf",
]
ANSWERABLE = [
    "Why is the avatar so fast?",
    "Hoe luistert de avatar naar mij?",
    "What happens when I correct the avatar?",
    "Waar draait dit project op?",
    "Is my microphone audio saved?",
    "Hoe beslist de avatar wat hij zegt?",
    "Wat gebeurt er met mijn gesprekken?",
    "Which GPU does it run on?",
    "Kan de avatar Nederlands verstaan?",
    "How do I delete my conversation logs?",
    "Wat is het doel van dit project?",
]
UNANSWERABLE = [
    "What is a balloon made of?",
    "Wie is de minister-president van Nederland?",
    "What is the capital of France?",
    "Kun je een grap vertellen over katten?",
]
# Related to the notes but not answered by them: a faithful model says it doesn't know.
TRAPS = [
    "Who built the avatar and when did the project start?",
    "Hoeveel heeft dit project gekost en wie betaalt het?",
]


def gpu_used() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    return int(out.strip() or 0)


knowledge = MarkdownKnowledge()
print("== retrieval scores (best passage)")
for question in ANSWERABLE + UNANSWERABLE + TRAPS:
    best = knowledge.search(question, 1)[0]
    kind = "ANS " if question in ANSWERABLE else "TRAP" if question in TRAPS else "NO  "
    print(f"  {kind} {best.score:.3f}  {question[:50]:50} -> {best.source}")

for model in MODELS:
    before = gpu_used()
    started = time.perf_counter()
    system2 = LlamaSystem2(model)
    load = time.perf_counter() - started
    system2.warm_up()
    print(f"\n== {model.split('::')[-1]}: load {load:.1f}s, GPU +{gpu_used() - before} MiB")
    times = []
    for question in ANSWERABLE + TRAPS:
        text, detail = system2.answer(question, knowledge.search(question, 3))
        times.append(detail["llm_ms"])
        print(f"  [{detail['llm_ms']:5d} ms, {detail['tokens']} tok] {question[:40]:40} -> {text}")
    print(f"  median {sorted(times)[len(times) // 2]} ms")
    del system2
