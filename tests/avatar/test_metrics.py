"""Tests for the per-phrase metrics the page and the benchmark read."""

import logging
import re
import unittest
from pathlib import Path

from avatar.metrics import _RENDER_FIELDS, log_phrase, phrase_metrics

ROOT = Path(__file__).resolve().parents[2]


def sample(**changes):
    render_detail = {name: 1 for name in _RENDER_FIELDS}
    render_detail.update(animation_region="all", first_window_ms=120)
    arguments = dict(
        index=2, count=3, phrase="Second phrase.", voice_mode="preset-clone",
        tts_seconds=0.9, tts_wait_ms=300, tts_prefetched=True,
        tts_detail={"provider": "chatterbox", "reference_used": True, "headers_ms": 5,
                    "first_byte_ms": 6, "download_ms": 1, "finalize_ms": 1, "bytes": 100},
        prefetch_policy="adaptive", next_prefetch_started=True,
        next_prefetch_buffer_seconds=0.53333, render_seconds=1.2345,
        render_detail=render_detail, adaptive_stride=True,
        stride_reason="buffer-healthy", buffer_before_seconds=1.23456,
        media_seconds=2.0004, buffered_seconds=0.8,
        played_ms={"underrun_ms": 20.0, "video_underrun_ms": 33.3,
                   "video_dropped_ms": 0.0, "speech_hold_ms": 40.0},
        speech_lead_seconds=0.25, fps_estimate=28.123456, first_ready_ms=None,
    )
    arguments.update(changes)
    return phrase_metrics(**arguments)


class PhraseMetricsTests(unittest.TestCase):
    def test_overlap_is_synthesis_time_not_spent_waiting(self) -> None:
        self.assertEqual(sample()["tts_overlap_ms"], 600)
        self.assertEqual(sample(tts_wait_ms=2000)["tts_overlap_ms"], 0)

    def test_values_are_rounded_for_display(self) -> None:
        metrics = sample()
        self.assertEqual(metrics["render_ms"], 1234)
        self.assertEqual(metrics["render_buffer_before_seconds"], 1.235)
        self.assertEqual(metrics["next_tts_prefetch_buffer_seconds"], 0.533)
        self.assertEqual(metrics["video_underrun_ms"], 33)
        self.assertEqual(metrics["speech_lead_ms"], 250)
        self.assertEqual(metrics["render_fps_estimate"], 28.123)
        self.assertIsNone(sample(next_prefetch_buffer_seconds=None)[
            "next_tts_prefetch_buffer_seconds"])

    def test_the_benchmark_finds_its_columns(self) -> None:
        source = (ROOT / "scripts/benchmark_avatar.py").read_text(encoding="utf-8")
        preferred = source[source.index("    preferred = ["):]
        preferred = preferred[:preferred.index("]")]
        columns = set(re.findall(r'"([a-z_]+)"', preferred))
        added_by_the_client = {"run_index", "warmup", "requested_mode", "client_wall_ms"}
        self.assertEqual(columns - added_by_the_client - set(sample()), set())

    def test_fields_of_the_removed_legacy_renderer_are_gone(self) -> None:
        for name in ("render_backend", "render_decode_ms", "render_pipeline_reported_ms"):
            self.assertNotIn(name, sample())

    def test_the_log_line_formats(self) -> None:
        with self.assertLogs("test-metrics", level="INFO") as captured:
            log_phrase(logging.getLogger("test-metrics"), sample())
        self.assertIn("Phrase 2/3 timings: voice=preset-clone tts=0.900s", captured.output[0])
        self.assertIn("prefetch_buffer=0.533s", captured.output[0])


if __name__ == "__main__":
    unittest.main()
