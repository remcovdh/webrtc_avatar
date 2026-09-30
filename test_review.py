"""Build-time tests for the conversation review (a fake review model)."""

import json
import tempfile
import unittest
from pathlib import Path

from review import ReviewModel, gather_facts, load_records, render, suggest


def turn(session, n, text, intent, source="jevk5", confidence=0.9, reply="ok", **extra):
    return {
        "time": f"2026-09-30T10:00:{n:02d}", "session": session, "turn": n, "text": text,
        "decision": {"intent": intent, "emotion": "neutral", "topic": None,
                     "sources": {"intent": source}, "confidence": {"intent": confidence},
                     "probabilities": {"intent": {intent: confidence}}},
        "reply": reply, **extra,
    }


RECORDS = [
    turn("a", 1, "Hello there", "greeting"),
    turn("a", 2, "What is a balloon", "question", source="keyword", reply="I don't know that yet.",
         knowledge_gap=True, retrieved=[{"source": "x.md#y", "score": 0.7}]),
    turn("a", 3, "No, that is not what I asked", "disagreement"),
    turn("a", 4, "I wonder about the weather", "remark", confidence=0.4),
    {"time": "2026-09-30T10:00:05", "session": "a",
     "correction": {"turn": 4, "text": "I wonder about the weather", "fields": {"intent": "question"}}},
    turn("b", 1, "Tell me about Devoteam", "request"),
    turn("b", 2, "Tell me about Devoteam please", "request"),
]


class FakeModel(ReviewModel):
    def __init__(self):
        self.asked = []

    def ask(self, instructions, material):
        self.asked.append(instructions.split("\n")[0])
        if "knowledge folder" in instructions:
            return {"pages": [{"file": "balloons.md", "title": "Balloons", "questions": ["What is a balloon"],
                               "what_to_write": "What balloons are made of."}]}
        if "classifier" in instructions:
            return {"description_changes": [{"field": "intent", "class": "question",
                                              "new_description": "asks or wonders about something",
                                              "why": "'I wonder' was a question"}], "new_classes": []}
        return {"diagnoses": [{"user": "What is a balloon", "problem": "no knowledge", "fix": "add a page"}]}


class ReviewTests(unittest.TestCase):
    def setUp(self):
        same = lambda a, b: 0.95 if a.startswith("Tell me") and b.startswith("Tell me") else 0.1  # noqa: E731
        self.facts = gather_facts(RECORDS, same)

    def test_counts_gaps_unsure_and_corrections(self):
        c = self.facts.counts
        self.assertEqual((c["turns"], c["sessions"], c["knowledge_gaps"], c["corrections"]), (6, 2, 1, 1))
        self.assertEqual(c["intent_sources"], {"jevk5": 5, "keyword": 1})
        self.assertEqual([u["text"] for u in self.facts.unsure], ["I wonder about the weather"])
        self.assertEqual(self.facts.corrections[0]["was"], {"intent": "remark"})
        self.assertEqual(self.facts.corrections[0]["now"], {"intent": "question"})

    def test_pushback_and_repeated_questions_flag_the_previous_turn(self):
        reasons = {w["user"]: w["reason"] for w in self.facts.went_wrong}
        self.assertEqual(reasons["What is a balloon"], "user answered with 'disagreement'")
        self.assertEqual(reasons["Tell me about Devoteam"], "user asked almost the same again")
        self.assertEqual(len(reasons), 2)

    def test_report_shows_facts_and_suggestions(self):
        model = FakeModel()
        suggestions = suggest(model, self.facts, {"intents": {}, "emotions": {}}, ["the-avatar.md"])
        self.assertEqual(len(model.asked), 3)
        report = render(self.facts, suggestions, None, "r::Qwen.gguf")
        for expected in ("balloons.md", "asks or wonders about something",
                         "Problem: no knowledge", "“What is a balloon”", "Nothing was changed automatically"):
            self.assertIn(expected, report)

    def test_only_records_after_the_last_review_are_read(self):
        folder = Path(tempfile.mkdtemp())
        (folder / "20260930-a.jsonl").write_text("\n".join(json.dumps(r) for r in RECORDS[:3]) + "\n")
        self.assertEqual(len(load_records(folder, None)), 3)
        self.assertEqual([r["turn"] for r in load_records(folder, "2026-09-30T10:00:01")], [2, 3])


if __name__ == "__main__":
    unittest.main()
