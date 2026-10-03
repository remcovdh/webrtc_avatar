"""Knowledge: find the passages that can answer a question.

`MarkdownKnowledge` indexes the personal `knowledge/` folder: every Markdown
section (split at headings) is embedded on the CPU with a small multilingual
model, so Dutch questions find English pages and the other way round. The
folder is re-indexed when a file changes. Anything smarter (graph RAG, a wiki,
documents, research agents) can replace it behind the `Knowledge` interface.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from shared.models import split_revision

LOG = logging.getLogger("knowledge")

KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", "/workspace/knowledge"))
# "<huggingface repo>[@<revision>]".
EMBED_MODEL = os.getenv(
    "KNOWLEDGE_EMBED_MODEL",
    "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3",
)
MAX_SECTION_CHARS = 700


@dataclass
class Passage:
    text: str
    source: str  # "file.md#Heading"
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Knowledge(Protocol):
    def search(self, question: str, limit: int = 3) -> list[Passage]:
        """Best-matching passages, most relevant first, with a 0-1 score."""
        ...


def split_markdown(text: str, name: str) -> list[tuple[str, str]]:
    """(source, passage) per section; long sections are cut at paragraphs.
    Each passage starts with its heading path so it can stand on its own."""
    sections: list[tuple[list[str], list[str]]] = [([], [])]
    headings: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            level = len(match.group(1))
            headings = headings[: level - 1] + [match.group(2).strip()]
            sections.append((list(headings), []))
        else:
            sections[-1][1].append(line)

    passages = []
    for path, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        title = " > ".join(path)
        source = f"{name}#{path[-1]}" if path else name
        chunk = ""
        for paragraph in re.split(r"\n\s*\n", body):
            paragraph = " ".join(paragraph.split())
            if chunk and len(chunk) + len(paragraph) > MAX_SECTION_CHARS:
                passages.append((source, f"{title}: {chunk}" if title else chunk))
                chunk = ""
            chunk = f"{chunk} {paragraph}".strip()
        if chunk:
            passages.append((source, f"{title}: {chunk}" if title else chunk))
    return passages


_EMBEDDER: Callable[[Sequence[str], bool], np.ndarray] | None = None


def load_embedder() -> Callable[[Sequence[str], bool], np.ndarray]:
    """The shared CPU embedding model (also used by System 1's correction
    memory), loaded once per process."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        repo, revision = split_revision(EMBED_MODEL)
        model = SentenceTransformer(repo, revision=revision, device="cpu")
        LOG.info("Embedding model %s loaded in %.1fs", EMBED_MODEL, time.perf_counter() - started)

        def embed(texts: Sequence[str], is_query: bool) -> np.ndarray:
            # E5 models expect these prefixes.
            prefix = "query: " if is_query else "passage: "
            return model.encode([prefix + t for t in texts], normalize_embeddings=True)

        _EMBEDDER = embed
    return _EMBEDDER


class MarkdownKnowledge:
    def __init__(
        self,
        folder: Path = KNOWLEDGE_DIR,
        embed: Callable[[Sequence[str], bool], np.ndarray] | None = None,
    ) -> None:
        self.folder = folder
        self._embed = embed or self._load_model()
        self._signature: tuple = ()
        self._sources: list[str] = []
        self._texts: list[str] = []
        self._vectors = np.zeros((0, 1), dtype=np.float32)

    @staticmethod
    def _load_model() -> Callable[[Sequence[str], bool], np.ndarray]:
        return load_embedder()

    def _files(self) -> list[Path]:
        return sorted(
            p for p in self.folder.glob("**/*.md")
            if not p.name.startswith("_") and p.name.lower() != "readme.md"
        )

    def _refresh(self) -> None:
        files = self._files()
        signature = tuple((str(p), p.stat().st_mtime, p.stat().st_size) for p in files)
        if signature == self._signature:
            return
        started = time.perf_counter()
        sources, texts = [], []
        for path in files:
            name = str(path.relative_to(self.folder))
            for source, text in split_markdown(path.read_text(encoding="utf-8"), name):
                sources.append(source)
                texts.append(text)
        self._vectors = (
            np.asarray(self._embed(texts, False), dtype=np.float32)
            if texts else np.zeros((0, 1), dtype=np.float32)
        )
        self._sources, self._texts, self._signature = sources, texts, signature
        LOG.info(
            "Indexed %d passages from %d pages in %.1fs",
            len(texts), len(files), time.perf_counter() - started,
        )

    def search(self, question: str, limit: int = 3) -> list[Passage]:
        self._refresh()
        if not self._texts:
            return []
        query = np.asarray(self._embed([question], True), dtype=np.float32)[0]
        scores = self._vectors @ query
        best = np.argsort(-scores)[:limit]
        return [Passage(self._texts[i], self._sources[i], round(float(scores[i]), 3)) for i in best]
