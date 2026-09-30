"""Conversation review (M5): what went well, what went wrong, what to improve.

Run by hand with `./review.sh` (it pauses the avatar so the review model fits on
the GPU). Reads the conversation logs written since the previous review and
writes `results/review/<timestamp>.md` plus a `.json` with the suggestions.
Nothing is changed automatically: you decide what to apply to
`config/system1.json` and the `knowledge/` folder.

The facts (counts, unsure decisions, corrections, knowledge gaps, turns that
went wrong) are computed directly from the logs. A local language model
(Qwen3 4B, the agreed review model) then proposes knowledge pages, better
class descriptions and new classes, and diagnoses the turns that went wrong.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

LOG = logging.getLogger("review")

RESULTS = Path(os.getenv("REVIEW_RESULTS", "/workspace/results"))
CONFIG_PATH = Path(os.getenv("SYSTEM1_CONFIG", "/workspace/config/system1.json"))
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", "/workspace/knowledge"))
REVIEW_MODEL = os.getenv(
    "REVIEW_MODEL",
    "unsloth/Qwen3-4B-Instruct-2507-GGUF::Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
)
UNSURE_BELOW = float(os.getenv("REVIEW_UNSURE_BELOW", "0.6"))
REPEAT_SIMILARITY = float(os.getenv("REVIEW_REPEAT_SIMILARITY", "0.88"))
# Intents that suggest the avatar's previous reply missed the mark.
PUSHBACK_INTENTS = {"disagreement", "stop"}


# ---------------------------------------------------------------------------
# Facts from the logs
# ---------------------------------------------------------------------------


@dataclass
class Facts:
    turns: list[dict[str, Any]] = field(default_factory=list)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    unsure: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    went_wrong: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)


def load_records(folder: Path, since: str | None) -> list[dict[str, Any]]:
    """All log lines (turns and corrections) newer than `since`, in time order."""
    records = []
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since is None or record.get("time", "") > since:
                records.append(record)
    return sorted(records, key=lambda r: r.get("time", ""))


def gather_facts(
    records: Sequence[dict[str, Any]],
    similarity: Callable[[str, str], float] | None = None,
) -> Facts:
    facts = Facts()
    by_turn: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if "correction" in record:
            correction = record["correction"]
            turn = by_turn.get((record.get("session"), correction.get("turn")), {})
            decided = turn.get("decision", {})
            facts.corrections.append({
                "text": correction.get("text"),
                "was": {k: decided.get(k) for k in correction.get("fields", {})},
                "now": correction.get("fields", {}),
            })
            continue
        if "turn" in record and "text" in record:
            facts.turns.append(record)
            by_turn[(record.get("session"), record.get("turn"))] = record

    intents, sources, emotions = Counter(), Counter(), Counter()
    ignored = Counter()
    for turn in facts.turns:
        decision = turn.get("decision") or {}
        if not decision:  # turns from before System 1 (the M2 echo)
            intents["no decision (echo)"] += 1
            continue
        intents[decision.get("intent", "?")] += 1
        emotions[decision.get("emotion", "?")] += 1
        sources[(decision.get("sources") or {}).get("intent", "?").split(" ")[0]] += 1
        for item in turn.get("ignored", []):
            ignored[item.get("reason", "?")] += 1
        confidence = (decision.get("confidence") or {}).get("intent", 1.0)
        # Only the models' own choices count (keyword rules and the memory are
        # certain by definition); older logs were decided by Laya.
        if (decision.get("sources") or {}).get("intent") in ("jevk5", "laya") and confidence < UNSURE_BELOW:
            facts.unsure.append({
                "text": turn["text"], "intent": decision.get("intent"),
                "confidence": confidence,
                "probabilities": (decision.get("probabilities") or {}).get("intent", {}),
            })
        if turn.get("knowledge_gap"):
            best = (turn.get("retrieved") or [{}])[0]
            facts.gaps.append({"question": turn["text"], "topic": decision.get("topic"),
                               "best_source": best.get("source"), "best_score": best.get("score")})

    # A turn went wrong if the user pushed back right after it, or asked
    # almost the same thing again in the same session.
    sessions: dict[str, list[dict[str, Any]]] = {}
    for turn in facts.turns:
        sessions.setdefault(turn.get("session", ""), []).append(turn)
    for turns in sessions.values():
        for previous, current in zip(turns, turns[1:]):
            intent = (current.get("decision") or {}).get("intent")
            reason = None
            if intent in PUSHBACK_INTENTS:
                reason = f"user answered with '{intent}'"
            elif similarity is not None and similarity(previous["text"], current["text"]) >= REPEAT_SIMILARITY:
                reason = "user asked almost the same again"
            if reason:
                facts.went_wrong.append({
                    "user": previous["text"], "avatar": previous.get("reply"),
                    "next_user": current["text"], "reason": reason,
                })

    facts.counts = {
        "turns": len(facts.turns),
        "sessions": len(sessions),
        "intents": dict(intents.most_common()),
        "emotions": dict(emotions.most_common()),
        "intent_sources": dict(sources.most_common()),
        "answered_from_knowledge": sum(1 for t in facts.turns if t.get("llm")),
        "knowledge_gaps": len(facts.gaps),
        "corrections": len(facts.corrections),
        "ignored_speech": dict(ignored),
    }
    return facts


# ---------------------------------------------------------------------------
# Suggestions from the review model
# ---------------------------------------------------------------------------


class ReviewModel:
    """Qwen3 4B through llama.cpp, asked for small JSON answers."""

    def __init__(self, model: str = REVIEW_MODEL, llm: Any = None) -> None:
        if llm is None:
            import torch  # noqa: F401  (load before llama.cpp, see system1.JevK5Decider)
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama

            repo, filename = model.split("::")
            llm = Llama(model_path=hf_hub_download(repo, filename), n_gpu_layers=-1, n_ctx=8192, verbose=False)
        self.llm = llm

    def ask(self, instructions: str, material: Any) -> Any:
        result = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": instructions + "\nAnswer with JSON only."},
                {"role": "user", "content": json.dumps(material, ensure_ascii=False, indent=1)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1500,
        )
        text = result["choices"][0]["message"]["content"] or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            LOG.warning("Review model gave no valid JSON: %r", text[:200])
            return {}


KNOWLEDGE_TASK = """You help maintain the knowledge folder of a talking avatar.
Below are questions users asked that the folder could not answer, and the pages
that exist. Group related questions and propose new pages (or additions to an
existing page). Return {"pages": [{"file": "name.md", "title": "...",
"questions": ["..."], "what_to_write": "one or two sentences"}]}.
Ignore questions that are not worth answering (tests, nonsense)."""

CLASSES_TASK = """You tune the classifier ("System 1") of a talking avatar. It
labels each user utterance with an intent and an emotion, choosing between the
classes and descriptions below. You get the user's corrections (what the model
said versus what was right) and utterances the model was unsure about.
Propose sharper descriptions for classes that were confused, and new classes
only if several utterances clearly need one. Return {"description_changes":
[{"field": "intent|emotion", "class": "...", "new_description": "...",
"why": "..."}], "new_classes": [{"field": "intent|emotion", "class": "...",
"description": "...", "examples": ["..."], "reaction": "a short English reply
phrase"}]}. Return empty lists if nothing needs to change."""

WRONG_TASK = """You review a talking avatar's conversations. Each item is a
user utterance, the avatar's reply, and what the user said next, which
suggests the reply missed the mark. For each item give a short diagnosis and
a concrete fix (a better reply phrase, a class description, a knowledge page).
Return {"diagnoses": [{"user": "...", "problem": "...", "fix": "..."}]}."""


def suggest(model: ReviewModel, facts: Facts, config: dict[str, Any], pages: list[str]) -> dict[str, Any]:
    suggestions: dict[str, Any] = {}
    if facts.gaps:
        suggestions["knowledge"] = model.ask(KNOWLEDGE_TASK, {"unanswered": facts.gaps, "existing_pages": pages})
    if facts.corrections or facts.unsure:
        suggestions["classes"] = classes = model.ask(CLASSES_TASK, {
            "intents": config.get("intents", {}),
            "emotions": config.get("emotions", {}),
            "corrections": facts.corrections,
            "unsure": facts.unsure[:30],
        })
        # The model sometimes copies the "intent|emotion" placeholder; an
        # existing class belongs to exactly one field.
        for change in classes.get("description_changes", []) if isinstance(classes, dict) else []:
            for name in ("intent", "emotion"):
                if change.get("class") in config.get(f"{name}s", {}):
                    change["field"] = name
    if facts.went_wrong:
        suggestions["went_wrong"] = model.ask(WRONG_TASK, {"items": facts.went_wrong[:20]})
    return suggestions


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render(facts: Facts, suggestions: dict[str, Any], since: str | None, model: str) -> str:
    c = facts.counts
    lines = [
        f"# Conversation review {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Conversations since {since or 'the beginning'}: {c['turns']} turns in "
        f"{c['sessions']} sessions. Review model: {model.split('::')[-1]}. "
        "Nothing was changed automatically.",
        "",
        "## Numbers",
        "",
        f"- Intents: {', '.join(f'{k} {v}' for k, v in c['intents'].items()) or '-'}",
        f"- Emotions: {', '.join(f'{k} {v}' for k, v in c['emotions'].items()) or '-'}",
        f"- Intent decided by: {', '.join(f'{k} {v}' for k, v in c['intent_sources'].items()) or '-'}",
        f"- Answered from the knowledge folder: {c['answered_from_knowledge']}; knowledge gaps: {c['knowledge_gaps']}",
        f"- Corrections: {c['corrections']}; ignored speech: {c['ignored_speech'] or '-'}",
        "",
    ]

    def section(title: str, rows: list[str], empty: str) -> None:
        lines.extend([f"## {title}", ""] + (rows or [empty]) + [""])

    section("Knowledge gaps", [
        f"- “{g['question']}” (topic {g['topic'] or '-'}; best match {g['best_source']} {g['best_score']})"
        for g in facts.gaps], "None.")
    pages = (suggestions.get("knowledge") or {}).get("pages", [])
    section("Suggested knowledge pages", ([
        "The review model may fill in facts itself; write pages from what you know "
        "to be true, since the avatar repeats them as fact.", ""] if pages else []) + [
        f"- **{p.get('file')}** — {p.get('title')}: {p.get('what_to_write')} "
        f"(answers: {'; '.join(p.get('questions', []))})" for p in pages], "No suggestions.")
    section("Corrections", [
        f"- “{k['text']}”: {k['was']} → {k['now']}" for k in facts.corrections], "None.")
    section("Unsure decisions", [
        f"- “{u['text']}”: {u['intent']} ({u['confidence']:.2f})" for u in facts.unsure[:30]], "None.")
    classes = suggestions.get("classes") or {}
    section("Suggested class descriptions", [
        f"- {d.get('field')} **{d.get('class')}**: “{d.get('new_description')}” — {d.get('why')}"
        for d in classes.get("description_changes", [])], "No suggestions.")
    section("Suggested new classes", [
        f"- {n.get('field')} **{n.get('class')}**: {n.get('description')} "
        f"(e.g. {'; '.join(n.get('examples', []))}; reply: “{n.get('reaction')}”)"
        for n in classes.get("new_classes", [])], "No suggestions.")
    diagnoses = {d.get("user"): d for d in (suggestions.get("went_wrong") or {}).get("diagnoses", [])}
    section("Turns that went wrong", [
        f"- “{w['user']}” → avatar: “{w['avatar']}” → user: “{w['next_user']}” ({w['reason']})"
        + (f"\n  - Problem: {diagnoses[w['user']].get('problem')} Fix: {diagnoses[w['user']].get('fix')}"
           if w["user"] in diagnoses else "")
        for w in facts.went_wrong], "None detected.")
    lines.append("Apply suggestions by editing `config/system1.json` and the `knowledge/` folder.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="review every log, not only new ones")
    parser.add_argument("--no-model", action="store_true", help="facts only, no review model")
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out = RESULTS / "review"
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    since = None if args.all or not state_path.exists() else json.loads(state_path.read_text()).get("last")
    records = load_records(RESULTS / "conversations", since)

    similarity = None
    try:
        from knowledge import load_embedder

        embed = load_embedder()
        similarity = lambda a, b: float(embed([a], True)[0] @ embed([b], True)[0])  # noqa: E731
    except Exception:
        LOG.warning("No embedding model; repeated questions are not detected")
    facts = gather_facts(records, similarity)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    pages = sorted(p.name for p in KNOWLEDGE_DIR.glob("*.md") if not p.name.startswith("_"))
    suggestions: dict[str, Any] = {}
    if not args.no_model and (facts.gaps or facts.corrections or facts.unsure or facts.went_wrong):
        started = time.perf_counter()
        suggestions = suggest(ReviewModel(), facts, config, pages)
        LOG.info("Review model finished in %.0fs", time.perf_counter() - started)

    stamp = time.strftime("%Y%m%d-%H%M")
    report = out / f"{stamp}.md"
    report.write_text(render(facts, suggestions, since, REVIEW_MODEL), encoding="utf-8")
    (out / f"{stamp}.json").write_text(json.dumps(
        {"since": since, "counts": facts.counts, "suggestions": suggestions}, ensure_ascii=False, indent=1
    ), encoding="utf-8")
    if records:
        state_path.write_text(json.dumps({"last": records[-1].get("time")}))
    print(report)


if __name__ == "__main__":
    main()
