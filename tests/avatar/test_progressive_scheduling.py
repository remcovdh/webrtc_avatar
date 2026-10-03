"""Build-time tests for v2M phrase merging and adaptive TTS prefetch."""

import ast
from pathlib import Path
import unittest


class ProgressiveSchedulingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (Path(__file__).resolve().parents[2] / "src/avatar/server.py").read_text(encoding="utf-8")

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
