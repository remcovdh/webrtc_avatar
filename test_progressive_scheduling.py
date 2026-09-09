"""Build-time tests for v2M phrase merging and adaptive TTS prefetch."""

import ast
from pathlib import Path
import unittest


class ProgressiveSchedulingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path(__file__).with_name("server.py").read_text(encoding="utf-8")
        tree = ast.parse(cls.source)
        split_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_split_phrases"
        )
        namespace = {
            "PROGRESSIVE_PHRASE_MODE": True,
            "PHRASE_FIRST_TARGET_CHARS": 48,
            "PHRASE_TARGET_CHARS": 100,
            "PHRASE_MAX_CHARS": 160,
            "MERGE_SHORT_OPENING_PHRASE": True,
            "PHRASE_MIN_FIRST_CHARS": 24,
        }
        ast.fix_missing_locations(split_node)
        exec(compile(ast.Module(body=[split_node], type_ignores=[]), "server.py", "exec"), namespace)
        cls.split_phrases = staticmethod(namespace["_split_phrases"])

    def test_short_greeting_merges_with_second_phrase(self) -> None:
        phrases = self.split_phrases(
            "Hello! This is my first neural streaming avatar test with Chatterbox. "
            "This remains a separate phrase."
        )
        self.assertEqual(
            phrases[0],
            "Hello! This is my first neural streaming avatar test with Chatterbox.",
        )
        self.assertEqual(len(phrases), 2)

    def test_normal_opening_sentence_is_not_forced_into_second(self) -> None:
        phrases = self.split_phrases(
            "This opening sentence is already comfortably long. This is phrase two."
        )
        self.assertEqual(phrases[0], "This opening sentence is already comfortably long.")
        self.assertEqual(len(phrases), 2)

    def test_prefetch_callback_occurs_after_first_window_append(self) -> None:
        tree = ast.parse(self.source)
        render_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_render_phrase_in_windows"
        )
        render_source = ast.get_source_segment(self.source, render_node) or ""
        append_at = render_source.index("playback.append_video_window(frames)")
        callback_at = render_source.index("prefetch_callback()")
        self.assertLess(append_at, callback_at)
        self.assertIn(
            "playback.buffered_seconds >= prefetch_min_buffer_seconds",
            render_source,
        )

    def test_eager_policy_remains_available_for_comparison(self) -> None:
        self.assertIn('TTS_PREFETCH_POLICY == "eager"', self.source)


if __name__ == "__main__":
    unittest.main()
