"""System 2: turn retrieved knowledge into a short spoken English answer.

The language model is only a formulator: it may use the retrieved passages and
nothing else. Deciding *whether* there is anything to say happens before it is
called (see the conductor: no relevant passage, no LLM call). `LlamaSystem2`
runs a small 4-bit GGUF model in-process with llama.cpp on the GPU; another
runtime or model can replace it behind the `System2` interface.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Protocol, Sequence

from knowledge import Passage

LOG = logging.getLogger("system2")

# "<huggingface repo>::<gguf file>". Chosen by measurement (see
# LISTENING_PLAN.md): Llama 3.2 3B answered fastest (median 260 ms), shortest
# and always in English; Qwen3 4B was richer but too long for speech; Phi-4-mini
# answered Dutch questions in Dutch.
MODEL = os.getenv(
    "SYSTEM2_MODEL",
    "bartowski/Llama-3.2-3B-Instruct-GGUF::Llama-3.2-3B-Instruct-Q4_K_M.gguf",
)
MAX_TOKENS = int(os.getenv("SYSTEM2_MAX_TOKENS", "120"))

INSTRUCTIONS = """You are the voice of a friendly avatar talking out loud to the user.
Answer the user's question using ONLY the facts in the notes below.
Rules:
- Reply in English, in 1 to 3 short spoken sentences.
- No lists, no markdown, no headings, no emojis.
- Never add facts that are not in the notes.
- If the notes do not answer the question, say briefly that you don't know that yet.
The question may be in Dutch; still answer in English."""


class System2(Protocol):
    def answer(self, question: str, passages: Sequence[Passage]) -> tuple[str, dict[str, Any]]:
        """The spoken answer and details (model, timings) for the log."""
        ...


def build_messages(question: str, passages: Sequence[Passage]) -> list[dict[str, str]]:
    notes = "\n\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))
    return [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": f"Notes:\n{notes}\n\nQuestion: {question}"},
    ]


def clean_spoken(text: str) -> str:
    """Strip what a speech voice should not read out, and cut to 3 sentences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"\s*\[\d+\]", "", text)
    text = re.sub(r"[*#`>]+", "", text)
    text = text.replace("_", " ")  # "warping_spade" -> "warping spade", not "warpingspade"
    text = " ".join(text.split())
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:3]).strip()


class LlamaSystem2:
    def __init__(self, model: str = MODEL, llm: Any = None) -> None:
        self.model = model
        if llm is None:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama

            repo, filename = model.split("::")
            started = time.perf_counter()
            path = hf_hub_download(repo, filename)
            llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
            LOG.info("System 2 model %s ready in %.1fs", filename, time.perf_counter() - started)
        self.llm = llm

    def warm_up(self) -> None:
        self.answer("What is this?", [Passage("Test: this is a warm-up note.", "warm-up", 1.0)])

    def answer(self, question: str, passages: Sequence[Passage]) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        result = self.llm.create_chat_completion(
            messages=build_messages(question, passages),
            max_tokens=MAX_TOKENS,
            temperature=0.3,
        )
        raw = result["choices"][0]["message"]["content"] or ""
        usage = result.get("usage", {})
        return clean_spoken(raw), {
            "model": self.model.split("::")[-1],
            "llm_ms": round((time.perf_counter() - started) * 1000),
            "tokens": usage.get("completion_tokens"),
            "raw": raw,
        }
