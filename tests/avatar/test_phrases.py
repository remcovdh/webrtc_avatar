"""Tests for splitting a text into phrases."""

import unittest

from avatar.config import load
from avatar.phrases import split_phrases

SETTINGS = load({}).settings


class SplitPhrasesTests(unittest.TestCase):
    def test_short_greeting_merges_with_second_phrase(self) -> None:
        phrases = split_phrases(
            "Hello! This is my first neural streaming avatar test with Chatterbox. "
            "This remains a separate phrase.",
            SETTINGS,
        )
        self.assertEqual(
            phrases[0],
            "Hello! This is my first neural streaming avatar test with Chatterbox.",
        )
        self.assertEqual(len(phrases), 2)

    def test_normal_opening_sentence_is_not_forced_into_second(self) -> None:
        phrases = split_phrases(
            "This opening sentence is already comfortably long. This is phrase two.",
            SETTINGS,
        )
        self.assertEqual(phrases[0], "This opening sentence is already comfortably long.")
        self.assertEqual(len(phrases), 2)

    def test_greeting_stays_separate_when_merging_is_off(self) -> None:
        settings = load({"AVATAR_MERGE_SHORT_OPENING_PHRASE": "false"}).settings
        phrases = split_phrases("Hello! This is the second sentence of the text.", settings)
        self.assertEqual(phrases[0], "Hello!")

    def test_long_sentence_is_cut_at_the_limits_without_losing_words(self) -> None:
        text = " ".join(f"word{index}" for index in range(80))
        phrases = split_phrases(text, SETTINGS)
        self.assertEqual(" ".join(phrases), text)
        self.assertGreater(len(phrases), 2)
        self.assertLessEqual(len(phrases[0]), SETTINGS.phrase_first_target_chars + 10)
        for phrase in phrases:
            self.assertLessEqual(len(phrase), SETTINGS.phrase_max_chars + 10)

    def test_whitespace_is_normalised_and_empty_text_gives_nothing(self) -> None:
        self.assertEqual(split_phrases("  \n  ", SETTINGS), [])
        self.assertEqual(split_phrases("One   two.\nThree.", SETTINGS)[0], "One two. Three.")


if __name__ == "__main__":
    unittest.main()
