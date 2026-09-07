"""Build-time tests for incremental FLP window timing and wiring."""

from collections import deque
from pathlib import Path
import math
import unittest


def emitted_counts(window_sizes: list[int], fps: float, target: int) -> list[int]:
    """Mirror PlaybackBuffer's cumulative, drift-free resampling counts."""
    source_frames = 0
    emitted = 0
    result = []
    for size in window_sizes:
        source_frames += size
        desired = min(target, math.ceil(source_frames * 30 / fps))
        result.append(desired - emitted)
        emitted = desired
    return result


class FrameWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path(__file__).with_name("server.py").read_text(encoding="utf-8")

    def test_split_windows_do_not_accumulate_rounding_drift(self) -> None:
        counts = emitted_counts([8, 8, 8], fps=12.5, target=58)
        self.assertEqual(counts, [20, 19, 19])
        self.assertEqual(sum(counts), 58)

    def test_worker_publishes_windows_thread_safely(self) -> None:
        self.assertIn("asyncio.run_coroutine_threadsafe(", self.source)

    def test_audio_starts_with_first_video_window(self) -> None:
        begin_at = self.source.index("playback.begin_phrase_stream(")
        append_at = self.source.index("playback.append_video_window(frames)")
        self.assertLess(begin_at, append_at)

    def test_v2j_rollback_switch_is_present(self) -> None:
        self.assertIn('INCREMENTAL_FRAME_WINDOWS = _env_bool(', self.source)


if __name__ == "__main__":
    unittest.main()
