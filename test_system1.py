"""Build-time tests for System 1 (a fake Laya agent; real spaCy models)."""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from system1 import (
    Config,
    LayaSystem1,
    Reactions,
    TopicExtractor,
    keyword_fields,
    normalize_language,
)

CONFIG = Path(__file__).with_name("system1.json")
if not CONFIG.exists():
    CONFIG = Path(__file__).parent / "config" / "system1.json"


class FakeAgent:
    """Answers like Laya: a fixed intent/emotion with the given confidences."""

    def __init__(self, intent="remark", emotion="joyful", emotion_confidence=0.9):
        self.intent, self.emotion, self.emotion_confidence = intent, emotion, emotion_confidence

    def predict(self, text, questions):
        return {"answers": {
            "intent": {"choice": self.intent, "answer_confidence": 0.8, "probabilities": {self.intent: 0.8}},
            "emotion": {"choice": self.emotion, "answer_confidence": self.emotion_confidence, "probabilities": {}},
        }}


def bag_of_words(texts):
    """Tiny deterministic embedding: word counts over a fixed vocabulary."""
    vocabulary = ["hoe", "laat", "is", "het", "what", "time", "weather", "weer", "morgen", "your", "name"]
    return np.array([[text.lower().count(word) + 1e-3 for word in vocabulary] for text in texts], dtype=np.float32)


class NoTopics(TopicExtractor):
    def extract(self, text, language, watch_words):
        return None, []


class System1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())
        self.config = self.folder / "system1.json"
        shutil.copy(CONFIG, self.config)
        self.memory = self.folder / "corrections.jsonl"

    def make(self, agent=None) -> LayaSystem1:
        return LayaSystem1(self.config, self.memory, agent or FakeAgent(), bag_of_words, NoTopics())

    def test_laya_answers_when_nothing_else_knows(self) -> None:
        d = self.make(FakeAgent("question", "neutral")).decide("Hoe laat is het?", "nl-NL")
        self.assertEqual((d.intent, d.emotion, d.language), ("question", "neutral", "nl"))
        self.assertTrue(d.needs_system2)
        self.assertEqual(d.sources, {"intent": "laya", "emotion": "laya"})

    def test_low_confidence_emotion_becomes_neutral(self) -> None:
        d = self.make(FakeAgent("remark", "surprised", 0.4)).decide("I see", "en-US")
        self.assertEqual((d.emotion, d.sources["emotion"]), ("neutral", "default"))

    def test_keywords_beat_laya(self) -> None:
        d = self.make(FakeAgent("greeting")).decide("Wacht even, stop maar.", "nl-NL")
        self.assertEqual((d.intent, d.sources["intent"]), ("stop", "keyword"))
        self.assertFalse(d.needs_system2)

    def test_a_correction_applies_at_once_to_similar_sentences_only(self) -> None:
        system1 = self.make(FakeAgent("greeting"))
        system1.correct("Kun je iets vertellen over het weer morgen?", {"intent": "request"})
        similar = system1.decide("Kun je iets over het weer van morgen vertellen?", "nl-NL")
        self.assertEqual(similar.intent, "request")
        self.assertTrue(similar.sources["intent"].startswith("memory"))
        other = system1.decide("What is your name?", "en-US")
        self.assertEqual((other.intent, other.sources["intent"]), ("greeting", "laya"))

    def test_corrections_survive_a_restart_and_invalid_classes_are_ignored(self) -> None:
        system1 = self.make()
        system1.correct("Hoe laat is het?", {"intent": "question", "emotion": "furious"})
        restarted = self.make()
        d = restarted.decide("Hoe laat is het?", "nl-NL")
        self.assertEqual((d.intent, d.emotion), ("question", "joyful"))
        self.assertEqual(len(restarted.memory), 1)

    def test_config_edits_apply_without_restart(self) -> None:
        system1 = self.make(FakeAgent("remark"))
        self.assertNotIn("joke", system1.options()["intent"])
        data = json.loads(self.config.read_text())
        data["intents"]["joke"] = "tells a joke"
        time.sleep(0.01)
        self.config.write_text(json.dumps(data))
        os.utime(self.config, (time.time() + 5, time.time() + 5))
        self.assertIn("joke", system1.options()["intent"])


class HelperTests(unittest.TestCase):
    def test_language_from_the_page_wins_and_auto_guesses(self) -> None:
        self.assertEqual(normalize_language("nl-NL", "hello"), "nl")
        self.assertEqual(normalize_language("auto", "Kun je me vertellen wat het is?"), "nl")
        self.assertEqual(normalize_language("auto", "Can you tell me what it is?"), "en")

    def test_keyword_rules_match_whole_words_in_short_utterances(self) -> None:
        rules = [{"words": ["stop"], "intent": "stop"}, {"words": ["hallo"], "intent": "greeting", "max_words": 6}]
        self.assertEqual(keyword_fields("Please stop now", rules), {"intent": "stop"})
        self.assertEqual(keyword_fields("It stopped raining", rules), {})
        self.assertEqual(keyword_fields("Hallo!", rules), {"intent": "greeting"})
        self.assertEqual(keyword_fields("Hallo, kun je me vertellen wat een ballon is?", rules), {})

    def test_reactions_prefer_intent_and_emotion_and_skip_missing_topics(self) -> None:
        from system1 import Decision

        reactions = Reactions(Config(CONFIG))
        joyful = Decision("x", "en", "remark", "joyful", False)
        self.assertEqual(reactions.pick(joyful), "Glad you like it!")
        no_topic = Decision("x", "en", "question", "neutral", True)
        self.assertEqual(reactions.pick(no_topic), "Good question. I can't answer that yet, but soon I can.")
        topic = Decision("x", "en", "question", "neutral", True, topic="ballon")
        self.assertIn("ballon", reactions.pick(topic))

    def test_topics_from_spacy_and_watch_words(self) -> None:
        topics = TopicExtractor()
        self.assertEqual(topics.extract("Waar is een ballon van gemaakt?", "nl", [])[0], "ballon")
        self.assertEqual(
            topics.extract("Can you tell me about its cloud projects?", "en", [])[0], "cloud projects"
        )
        # ASR misspelling still hits the watch word.
        self.assertEqual(topics.extract("Wat doet Devotiem?", "nl", ["Devoteam"])[0], "Devoteam")


if __name__ == "__main__":
    unittest.main()
