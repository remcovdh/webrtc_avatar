"""System 1: fast typed decisions about what the user just said.

`decide(text, language)` returns a `Decision`: intent, emotion, whether System 2
must answer, and the topic. Each field comes from the first source that knows:

1. the correction memory: a sentence very similar to one the user corrected
   in the page (cosine similarity of multilingual-e5-small embeddings);
2. keyword rules on the ASR text (config);
3. a `Decider`: typed zero-shot decisions. `JevK5Decider` runs JevK5 v0.3 4B
   (Qwen3.5-4B + LoRA, option-logit readout) in-process with llama.cpp. It
   replaced Laya, which scored intent 11/16 and emotion 4-9/16 on our test set
   against JevK5's 15/18 and 17/18.

Classes, criteria, keyword rules and reaction phrases live in
`config/system1.json` and are re-read when the file changes. Everything that
needs a model can be injected, so the logic is testable without downloads.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

LOG = logging.getLogger("system1")

CONFIG_PATH = Path(os.getenv("SYSTEM1_CONFIG", "/workspace/config/system1.json"))
MEMORY_PATH = Path(os.getenv("SYSTEM1_MEMORY", "/workspace/results/system1/corrections.jsonl"))
# "<huggingface repo>[@<revision>]::<gguf file>" of the JevK5 decision model.
JEVK5_MODEL = os.getenv(
    "SYSTEM1_MODEL",
    "alibiserikbay/JevK5-GGUF@ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a"
    "::jevk5-4b-v0.3-Q4_K_M.gguf",
)
# Softmax temperature of the option-logit readout (JevK5 v0.3's jevk5_config.json).
JEVK5_TEMPERATURE = 1.22
FIELDS = ("intent", "emotion")


@dataclass
class Decision:
    text: str
    language: str
    intent: str
    emotion: str
    needs_system2: bool
    topic: str | None = None
    keywords: list[str] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)  # field -> memory/keyword/jevk5
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class System1(Protocol):
    def decide(self, text: str, language: str) -> Decision: ...

    def correct(self, text: str, fields: dict[str, str]) -> None:
        """Remember the user's correction; similar sentences follow it at once."""
        ...

    def options(self) -> dict[str, list[str]]:
        """The classes the page can offer for corrections."""
        ...


# ---------------------------------------------------------------------------
# Config (re-read on change)
# ---------------------------------------------------------------------------


class Config:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime: float | None = None
        self._data: dict[str, Any] = {}

    def get(self) -> dict[str, Any]:
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._data, self._mtime = data, mtime
            LOG.info("Loaded System 1 config %s", self.path)
        return self._data


# ---------------------------------------------------------------------------
# Language, keywords, topic
# ---------------------------------------------------------------------------

_DUTCH_HINTS = {
    "de", "het", "een", "ik", "je", "jij", "niet", "wat", "waar", "hoe", "kun",
    "is", "en", "van", "dat", "maar", "ook", "met", "voor", "mij", "we", "zijn",
}
_ENGLISH_HINTS = {
    "the", "a", "an", "i", "you", "not", "what", "where", "how", "can", "is",
    "and", "of", "that", "but", "also", "with", "for", "me", "we", "are",
}


def normalize_language(language: str | None, text: str) -> str:
    """"en-US"/"nl-NL" from the page win; "auto" guesses from the words."""
    if language and not language.startswith("auto"):
        return language.split("-")[0].lower()
    words = set(re.findall(r"[a-zà-ÿ']+", text.lower()))
    return "nl" if len(words & _DUTCH_HINTS) > len(words & _ENGLISH_HINTS) else "en"


def _plain(text: str) -> str:
    return " " + " ".join(re.findall(r"[\wà-ÿ']+", text.lower())) + " "


def keyword_fields(text: str, rules: list[dict[str, Any]]) -> dict[str, str]:
    """Fields decided by the first matching keyword rule (whole words).

    A rule matches on `words` anywhere, `starts` (the utterance's first words)
    or `ends` (its last characters, e.g. "?"). `max_words` limits it to short
    utterances: "Hallo" makes "Hallo!" a greeting, but not "Hallo, kun je me
    vertellen wat een ballon is?"."""
    plain = _plain(text)
    length = len(plain.split())
    stripped = text.strip()
    for rule in rules:
        if length > rule.get("max_words", 10_000):
            continue
        matched = (
            any(_plain(phrase) in plain for phrase in rule.get("words", []))
            or any(plain.startswith(_plain(phrase)) for phrase in rule.get("starts", []))
            or any(stripped.endswith(end) for end in rule.get("ends", []))
        )
        if matched:
            return {name: value for name, value in rule.items() if name in FIELDS}
    return {}


class TopicExtractor:
    """Classical, CPU-only topic picking: watch words, noun phrases, YAKE."""

    def __init__(self, nlp: dict[str, Any] | None = None) -> None:
        self._nlp = nlp

    def _model(self, language: str) -> Any:
        if self._nlp is None:
            import spacy

            self._nlp = {"en": spacy.load("en_core_web_sm"), "nl": spacy.load("nl_core_news_sm")}
        return self._nlp.get(language) or self._nlp["en"]

    def extract(self, text: str, language: str, watch_words: Sequence[str]) -> tuple[str | None, list[str]]:
        from rapidfuzz import fuzz

        for word in watch_words:
            for token in re.findall(r"[\wà-ÿ']+", text):
                if fuzz.ratio(word.lower(), token.lower()) >= 85:
                    return word, [word]

        doc = self._model(language)(text)
        best, best_score = None, 0
        for chunk in doc.noun_chunks:
            tokens = [t for t in chunk if t.pos_ not in ("DET", "PRON")]
            if not any(t.pos_ in ("NOUN", "PROPN") for t in tokens):
                continue
            score = len(tokens) + 2 * sum(t.pos_ == "PROPN" for t in tokens)
            if score > best_score:
                best, best_score = " ".join(t.text for t in tokens), score
        keywords = []
        try:
            import yake

            keywords = [k for k, _ in yake.KeywordExtractor(lan=language, n=2, top=3).extract_keywords(text)]
        except Exception:  # YAKE is only a helper
            pass
        return best, keywords


# ---------------------------------------------------------------------------
# Correction memory
# ---------------------------------------------------------------------------


class CorrectionMemory:
    """Corrections by sentence; the nearest similar sentence wins per field."""

    def __init__(self, path: Path, embed: Callable[[Sequence[str]], np.ndarray]) -> None:
        self.path, self.embed = path, embed
        self._texts: list[str] = []
        self._fields: list[dict[str, str]] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        if path.exists():
            merged: dict[str, dict[str, str]] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                merged.setdefault(record["text"], {}).update(record["fields"])
            for text, fields in merged.items():
                self._remember(text, fields)

    def _unit(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self.embed(list(texts)), dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)

    def _remember(self, text: str, fields: dict[str, str]) -> None:
        if text in self._texts:
            self._fields[self._texts.index(text)].update(fields)
            return
        vector = self._unit([text])
        self._vectors = vector if len(self._texts) == 0 else np.vstack([self._vectors, vector])
        self._texts.append(text)
        self._fields.append(dict(fields))

    def add(self, text: str, fields: dict[str, str]) -> None:
        self._remember(text, fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "text": text, "fields": fields}, ensure_ascii=False) + "\n")

    def lookup(self, text: str, threshold: float) -> tuple[dict[str, str], float]:
        if not self._texts:
            return {}, 0.0
        similarities = self._vectors @ self._unit([text])[0]
        best = int(np.argmax(similarities))
        similarity = float(similarities[best])
        return (dict(self._fields[best]), similarity) if similarity >= threshold else ({}, similarity)

    def __len__(self) -> int:
        return len(self._texts)


# ---------------------------------------------------------------------------
# Deciders: typed zero-shot decisions
# ---------------------------------------------------------------------------


class Decider(Protocol):
    def predict(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """{"answers": {name: {"choice", "probabilities", "answer_confidence"}}}
        for `choice` questions with {"instructions", "criteria": {key: text}}."""
        ...


class JevK5Decider:
    """JevK5 in-process: one forward pass per question; the answer is the
    softmax over the option letters' next-token logits (SemIf's readout)."""

    def __init__(self, model: str = JEVK5_MODEL, llm: Any = None) -> None:
        import threading

        from jevk5.prompt import LETTERS, decision_options, prompt_text

        self._letters, self._options, self._prompt = LETTERS, decision_options, prompt_text
        if llm is None:
            # PyTorch must load its CUDA libraries before llama.cpp does; the
            # other order fails later with "libtorch_cuda.so: undefined symbol:
            # ncclCommWindowRegister" when the embeddings import torch.
            import torch  # noqa: F401
            from llama_cpp import Llama

            from shared.models import download_gguf

            filename = model.split("::")[-1]
            started = time.perf_counter()
            # logits_all keeps the last position's logits readable; a small
            # context keeps that buffer small (prompts are a few hundred tokens).
            llm = Llama(
                model_path=download_gguf(model),
                n_gpu_layers=-1, n_ctx=1024, logits_all=True, verbose=False,
            )
            LOG.info("JevK5 %s loaded in %.1fs", filename, time.perf_counter() - started)
        self.llm = llm
        self._letter_ids = [
            llm.tokenize(letter.encode(), add_bos=False, special=False)[0] for letter in LETTERS
        ]
        self._lock = threading.Lock()  # one llama context, shared by sessions

    def _probabilities(self, text: str, question: dict[str, Any]) -> dict[str, float]:
        options = self._options(question)
        prompt = self._prompt(text, question["instructions"], [label for _, label in options])
        tokens = self.llm.tokenize(prompt.encode(), add_bos=False, special=True)
        with self._lock:
            self.llm.reset()
            self.llm.eval(tokens)
            logits = self.llm.scores[self.llm.n_tokens - 1]
            z = [float(logits[self._letter_ids[i]]) for i in range(len(options))]
        top = max(z)
        weights = [float(np.exp((v - top) / JEVK5_TEMPERATURE)) for v in z]
        total = sum(weights)
        return {key: w / total for (key, _), w in zip(options, weights)}

    def predict(self, text: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        answers = {}
        for name, question in questions.items():
            probabilities = self._probabilities(text, question)
            choice = max(probabilities, key=probabilities.get)
            answers[name] = {
                "choice": choice,
                "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
                "answer_confidence": probabilities[choice],
            }
        return {"answers": answers}


# ---------------------------------------------------------------------------
# System 1: correction memory > keyword rules > decider
# ---------------------------------------------------------------------------


class TypedSystem1:
    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        memory_path: Path = MEMORY_PATH,
        agent: Decider | None = None,
        embed: Callable[[Sequence[str]], np.ndarray] | None = None,
        topics: TopicExtractor | None = None,
    ) -> None:
        self.config = Config(config_path)
        self.agent = agent or JevK5Decider()
        if embed is None:
            from conductor.knowledge import load_embedder

            e5 = load_embedder()
            embed = lambda texts: e5(texts, True)  # noqa: E731 (sentence vs sentence)
        self.memory = CorrectionMemory(memory_path, embed)
        self.topics = topics or TopicExtractor()
        LOG.info("Correction memory: %d sentences", len(self.memory))

    def warm_up(self) -> None:
        """Load spaCy and run the decider once per language before the first turn."""
        started = time.perf_counter()
        for text, language in (("Hoe gaat het?", "nl-NL"), ("How are you?", "en-US")):
            self.decide(text, language)
        LOG.info("System 1 warmed up in %.1fs", time.perf_counter() - started)

    def options(self) -> dict[str, list[str]]:
        cfg = self.config.get()
        return {"intent": list(cfg["intents"]), "emotion": list(cfg["emotions"])}

    def decide(self, text: str, language: str) -> Decision:
        started = time.perf_counter()
        cfg = self.config.get()
        lang = normalize_language(language, text)
        questions = {
            "intent": {"type": "choice", "instructions": "What is the speaker doing with this utterance to a conversational assistant?", "criteria": cfg["intents"]},
            "emotion": {"type": "choice", "instructions": "Which emotion does the speaker express?", "criteria": cfg["emotions"]},
        }
        answers = self.agent.predict(text, questions)["answers"]
        values = {name: answers[name]["choice"] for name in FIELDS}
        confidence = {name: round(float(answers[name].get("answer_confidence", 0.0)), 3) for name in FIELDS}
        sources = {name: "jevk5" for name in FIELDS}
        # Low-confidence zero-shot emotions are more often wrong than right.
        if confidence["emotion"] < cfg.get("min_emotion_confidence", 0.0):
            values["emotion"], sources["emotion"] = "neutral", "default"

        for name, value in keyword_fields(text, cfg.get("keyword_rules", [])).items():
            values[name], sources[name] = value, "keyword"
        remembered, similarity = self.memory.lookup(text, cfg.get("memory_similarity", 0.9))
        for name, value in remembered.items():
            if name in FIELDS:
                values[name], sources[name] = value, f"memory ({similarity:.2f})"

        topic, keywords = self.topics.extract(text, lang, cfg.get("watch_words", []))
        return Decision(
            text=text,
            language=lang,
            intent=values["intent"],
            emotion=values["emotion"],
            needs_system2=values["intent"] in cfg.get("system2_intents", []),
            topic=topic,
            keywords=keywords,
            confidence=confidence,
            sources=sources,
            probabilities={name: answers[name].get("probabilities", {}) for name in FIELDS},
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def correct(self, text: str, fields: dict[str, str]) -> None:
        valid = self.options()
        clean = {name: value for name, value in fields.items() if name in FIELDS and value in valid[name]}
        if clean:
            self.memory.add(text, clean)
            LOG.info("Correction for %r: %s", text, clean)


# ---------------------------------------------------------------------------
# Reactions
# ---------------------------------------------------------------------------


class Reactions:
    """Pick an English reaction phrase for a decision, in rotation."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._counters: dict[str, itertools.count] = {}

    def pick(self, decision: Decision) -> str | None:
        return self.pick_for((f"{decision.intent}+{decision.emotion}", decision.intent), decision.topic)

    def pick_for(self, keys: Sequence[str], topic: str | None) -> str | None:
        """A phrase from the first key with phrases (e.g. "unknown")."""
        reactions = self.config.get().get("reactions", {})
        for key in keys:
            options = reactions.get(key, [])
            # Naming the topic is what makes the avatar sound like it
            # understood, so phrases with {topic} win whenever there is one.
            with_topic = [p for p in options if "{topic}" in p]
            without_topic = [p for p in options if "{topic}" not in p]
            chosen = with_topic if topic and with_topic else without_topic
            phrases = [p.replace("{topic}", topic or "") for p in chosen]
            if phrases:
                index = next(self._counters.setdefault(key, itertools.count()))
                return phrases[index % len(phrases)]
        return None
