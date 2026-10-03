"""Build-time tests for knowledge retrieval and System 2 (no model downloads)."""

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from conductor.app import ConversationResponder, Turn
from conductor.knowledge import MarkdownKnowledge, Passage, split_markdown
from conductor.system2 import LlamaSystem2, build_messages, clean_spoken
from test_system1 import CONFIG, FakeAgent, NoTopics, bag_of_words
from conductor.system1 import TypedSystem1

VOCABULARY = ["tensorrt", "fast", "frame", "listen", "microphone", "privacy", "balloon"]


def keyword_embed(texts, is_query):
    vectors = np.array([[t.lower().count(w) + 1e-3 for w in VOCABULARY] for t in texts], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class FakeLlm:
    def __init__(self) -> None:
        self.calls = 0

    def create_chat_completion(self, messages, max_tokens, temperature):
        self.calls += 1
        return {
            "choices": [{"message": {"content": "**TensorRT** made it fast [1]. It renders a frame in 27 ms. More. Even more."}}],
            "usage": {"completion_tokens": 20},
        }


class KnowledgeTests(unittest.TestCase):
    def test_sections_carry_their_heading_and_long_ones_are_split(self) -> None:
        long = "\n\n".join(["word " * 80] * 3)
        passages = split_markdown(f"# Avatar\n\n## Speed\nTensorRT is fast.\n\n## Long\n{long}", "a.md")
        self.assertEqual(passages[0], ("a.md#Speed", "Avatar > Speed: TensorRT is fast."))
        self.assertGreater(sum(source == "a.md#Long" for source, _ in passages), 1)

    def test_search_finds_the_right_section_and_reindexes_on_change(self) -> None:
        folder = Path(tempfile.mkdtemp())
        (folder / "a.md").write_text("# A\n## Speed\nTensorRT makes every frame fast.\n## Ears\nThe microphone lets it listen.\n")
        (folder / "_draft.md").write_text("# Draft\nballoon balloon balloon\n")
        knowledge = MarkdownKnowledge(folder, keyword_embed)
        best = knowledge.search("How does it listen with the microphone?", limit=1)[0]
        self.assertEqual(best.source, "a.md#Ears")
        self.assertLess(knowledge.search("balloon", limit=1)[0].score, 0.5)  # drafts ignored
        time.sleep(0.01)
        (folder / "b.md").write_text("# Toys\nA balloon is a toy.\n")
        self.assertEqual(knowledge.search("balloon", limit=1)[0].source, "b.md#Toys")


class System2Tests(unittest.TestCase):
    def test_spoken_output_is_cleaned_and_short(self) -> None:
        text, detail = LlamaSystem2("repo::model.gguf", FakeLlm()).answer("Why fast?", [Passage("TensorRT", "a", 0.9)])
        self.assertEqual(text, "TensorRT made it fast. It renders a frame in 27 ms. More.")
        self.assertEqual(detail["model"], "model.gguf")
        self.assertEqual(clean_spoken("<think>hmm</think> Yes."), "Yes.")
        self.assertEqual(clean_spoken("The warping_spade network [2] ."), "The warping spade network.")

    def test_prompt_numbers_the_notes_and_keeps_the_question(self) -> None:
        messages = build_messages("Waarom?", [Passage("Fact one", "a", 0.9), Passage("Fact two", "b", 0.8)])
        self.assertIn("[1] Fact one", messages[1]["content"])
        self.assertIn("Question: Waarom?", messages[1]["content"])
        # The reminder is the last thing the model reads.
        self.assertTrue(messages[1]["content"].endswith(
            "(Answer in English, using only the notes. If the notes do not contain "
            "the answer, say you don't know that yet.)"
        ))
        self.assertIn("ONLY the facts", messages[0]["content"])


class ConversationResponderTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = Path(tempfile.mkdtemp())
        config = folder / "system1.json"
        shutil.copy(CONFIG, config)
        system1 = TypedSystem1(config, folder / "corrections.jsonl", FakeAgent("question", "neutral"), bag_of_words, NoTopics())
        pages = folder / "knowledge"
        pages.mkdir()
        (pages / "a.md").write_text("# Avatar\n## Speed\nTensorRT makes every frame fast.\n")
        self.llm = FakeLlm()
        self.responder = ConversationResponder(
            system1, MarkdownKnowledge(pages, keyword_embed), LlamaSystem2("r::m.gguf", self.llm)
        )

    def ask(self, text):
        turn = Turn(text, "spk0", [1], "silence")
        filler, detail = self.responder.reply(turn, "en-US")
        self.assertTrue(detail["needs_answer"])
        return filler, *self.responder.answer(turn, detail)

    def test_relevant_knowledge_is_rephrased_by_the_llm(self) -> None:
        filler, answer, detail = self.ask("Why is TensorRT so fast?")
        self.assertEqual(filler, "Good question, let me think.")
        self.assertTrue(answer.startswith("TensorRT made it fast."))
        self.assertEqual(self.llm.calls, 1)
        self.assertEqual(detail["retrieved"][0]["source"], "a.md#Speed")

    def test_a_knowledge_gap_skips_the_llm_and_says_so(self) -> None:
        _, answer, detail = self.ask("What is a balloon?")
        self.assertEqual(self.llm.calls, 0)
        self.assertTrue(detail["knowledge_gap"])
        self.assertIn(answer, ["I don't know that yet.", "That's not something I know about yet."])


if __name__ == "__main__":
    unittest.main()
