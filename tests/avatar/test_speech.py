"""Tests for the speaking flow, with stand-ins for the renderer and the TTS.

No models and no GPU: the stand-in renderer produces numbered frames on a
worker thread exactly as the real one hands over windows.
"""

import asyncio
import json
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from avatar.config import load
from avatar.playback import IdleFrame, PlaybackBuffer
from avatar.speech import Speaker, select_render_stride
from avatar.tts_client import VoiceRequest

ROOT = Path(__file__).resolve().parents[2]
THREE_PHRASES = (
    "This opening sentence is already comfortably long. "
    "This is the second phrase of the text. And this is the third one."
)


class FakeChannel:
    readyState = "open"

    def __init__(self) -> None:
        self.events: list[dict] = []

    def send(self, message: str) -> None:
        self.events.append(json.loads(message))

    def of(self, kind: str) -> list[dict]:
        return [event for event in self.events if event["type"] == kind]


class FakeTts:
    def __init__(self, timeline: list[str], seconds: float = 0.4) -> None:
        self.timeline = timeline
        self.seconds = seconds
        self.texts: list[str] = []
        self.voice_error: str | None = None

    def resolve_voice(self, mode: str) -> VoiceRequest:
        if self.voice_error:
            raise ValueError(self.voice_error)
        return VoiceRequest(mode or "design")

    async def wait_until_ready(self) -> float:
        return 0.0

    async def synthesize(self, text, voice, pcm_path, wav_path):
        self.texts.append(text)
        self.timeline.append(f"tts:{len(self.texts)}")
        await asyncio.sleep(0)
        detail = {
            "headers_ms": 1, "first_byte_ms": 1, "download_ms": 0, "finalize_ms": 0,
            "total_ms": 1, "bytes": 2, "voice_mode": voice.mode,
            "reference_used": False, "reference_bytes": 0, "provider": "chatterbox",
        }
        return np.ones(int(self.seconds * 24_000), dtype=np.int16), detail


class FakeRenderer:
    """10 motion frames at 25 fps per phrase, in windows of 4."""

    def __init__(self, timeline: list[str], frames: int = 10) -> None:
        self.timeline = timeline
        self.frames = frames
        self.fps_estimate = 30.0
        self.calls: list[dict] = []
        self.fail: str | None = None
        self.threads: set[int] = set()

    def measured(self, effective_fps: float) -> None:
        self.fps_estimate = effective_fps or self.fps_estimate

    def measured_at_warmup(self, effective_fps: float) -> None:
        self.fps_estimate = max(self.fps_estimate, effective_fps)

    def render(self, wav_path, render_stride=None, reset_motion_reference=True,
               window_callback=None):
        if self.fail:
            raise RuntimeError(self.fail)
        phrase = len(self.calls) + 1
        self.calls.append({"stride": render_stride, "reset": reset_motion_reference,
                           "windows": window_callback is not None})
        self.threads.add(threading.get_ident())
        stride = render_stride or 1
        ids = list(range(0, self.frames, stride))
        frames = [np.full((2, 2, 3), phrase * 20 + i, dtype=np.uint8) for i in ids]
        fps = 25.0 / stride
        windows = 0
        if window_callback is not None:
            time.sleep(0.02)  # the first window takes a moment, as on the GPU
            for start in range(0, len(frames), 4):
                windows += 1
                self.timeline.append(f"window:{phrase}.{windows}")
                window_callback(frames[start:start + 4], fps, {
                    "index": windows, "render_ms": 1,
                    "expected_rendered_frames": len(frames),
                    "motion_frames": self.frames,
                })
        self.timeline.append(f"rendered:{phrase}")
        detail = {
            "motion_ms": 1, "frame_loop_ms": 1, "pipeline_ms": 2, "total_ms": 2,
            "render_stride": stride, "motion_frames": self.frames, "frames": len(frames),
            "source_fps": 25.0, "playback_fps": fps, "effective_fps": 40.0,
            "motion_reference_reset": reset_motion_reference,
            "persistent_phrase_motion": True, "relative_motion": True,
            "animation_region": "all", "driving_multiplier": 1.0, "normalize_lip": True,
            "eye_retargeting": False, "lip_retargeting": False,
            "incremental_windows": window_callback is not None,
            "window_size": 4 if window_callback is not None else 0,
            "window_count": windows, "window_max_ms": 1, "window_mean_ms": 1,
        }
        return ([] if window_callback is not None else frames), fps, detail


class SpeakerCase(unittest.IsolatedAsyncioTestCase):
    def build(self, **environment: str) -> Speaker:
        environment.setdefault("AVATAR_RESULTS_ROOT", tempfile.mkdtemp())
        self.timeline: list[str] = []
        self.tts = FakeTts(self.timeline)
        self.renderer = FakeRenderer(self.timeline)
        self.idle = IdleFrame(np.full((2, 2, 3), 255, dtype=np.uint8))
        self.channel = FakeChannel()
        self.playback = PlaybackBuffer(self.idle)
        self.shown: list[int] = []
        return Speaker(load(environment).settings, self.renderer, self.tts, self.idle)

    async def play(self) -> None:
        """Stands in for the browser: consume audio and video as fast as possible."""
        while True:
            self.playback.next_audio()
            value = int(self.playback.next_video()[0, 0, 0])
            if value != 255 and (not self.shown or self.shown[-1] != value):
                self.shown.append(value)
            await asyncio.sleep(0)

    async def speak(self, speaker: Speaker, text: str = THREE_PHRASES) -> None:
        player = asyncio.create_task(self.play())
        try:
            await asyncio.wait_for(
                speaker.speak(text, "", self.playback, self.channel), timeout=10
            )
        finally:
            player.cancel()


class SpeakTests(SpeakerCase):
    async def test_three_phrases_are_planned_spoken_and_reported(self) -> None:
        speaker = self.build()
        await self.speak(speaker)
        kinds = [event["type"] for event in self.channel.events]
        self.assertEqual(kinds[0], "plan")
        self.assertEqual(kinds[-2:], ["summary", "ready"])
        self.assertEqual(len(self.channel.of("plan")[0]["phrases"]), 3)
        self.assertEqual(self.tts.texts, self.channel.of("plan")[0]["phrases"])
        metrics = self.channel.of("metrics")
        self.assertEqual([m["chunk"] for m in metrics], [1, 2, 3])
        self.assertIsNotNone(metrics[0]["first_ready_ms"])
        self.assertIsNone(metrics[1]["first_ready_ms"])
        self.assertEqual(self.channel.of("summary")[0]["chunks"], 3)
        self.assertFalse(self.playback.busy)
        self.assertFalse(speaker.lock.locked())

    async def test_frames_reach_the_browser_in_order(self) -> None:
        speaker = self.build()
        await self.speak(speaker)
        self.assertEqual(self.shown, sorted(self.shown))
        self.assertGreaterEqual(self.shown[0], 20)   # phrase 1
        self.assertGreaterEqual(self.shown[-1], 60)  # phrase 3
        # Rendering ran on a worker thread, not on the event loop.
        self.assertNotIn(threading.get_ident(), self.renderer.threads)

    async def test_only_the_first_phrase_resets_the_motion_reference(self) -> None:
        speaker = self.build()
        await self.speak(speaker)
        self.assertEqual([call["reset"] for call in self.renderer.calls], [True, False, False])

    async def test_every_phrase_resets_without_persistent_motion(self) -> None:
        speaker = self.build(AVATAR_PERSISTENT_PHRASE_MOTION="false")
        await self.speak(speaker)
        self.assertEqual([call["reset"] for call in self.renderer.calls], [True, True, True])

    async def test_whole_phrase_rendering_without_windows(self) -> None:
        speaker = self.build(AVATAR_PROFILE="benchmark-visual")
        await self.speak(speaker)
        self.assertEqual([call["windows"] for call in self.renderer.calls], [False] * 3)
        self.assertEqual(len(self.channel.of("playing")), 1)
        self.assertEqual(self.shown, sorted(self.shown))

    async def test_the_page_finds_every_field_it_shows(self) -> None:
        speaker = self.build()
        await self.speak(speaker)
        page = (ROOT / "src/avatar/index.html").read_text(encoding="utf-8")
        timing = page[page.index("function appendTiming"):page.index("function updateVoiceControls")]
        wanted = set(re.findall(r"message\.([a-z_]+)", timing))
        self.assertGreater(len(wanted), 20)
        self.assertEqual(wanted - set(self.channel.of("metrics")[0]), set())


class PrefetchTests(SpeakerCase):
    def first(self, marker: str) -> int:
        return self.timeline.index(marker)

    async def test_adaptive_waits_for_the_first_video_window(self) -> None:
        speaker = self.build(AVATAR_TTS_PREFETCH_MIN_BUFFER_SECONDS="0")
        await self.speak(speaker)
        self.assertLess(self.first("window:1.1"), self.first("tts:2"))
        self.assertLess(self.first("tts:2"), self.first("rendered:1"))
        metrics = self.channel.of("metrics")
        self.assertTrue(metrics[0]["next_tts_prefetch_started"])
        self.assertTrue(metrics[1]["tts_prefetched"])
        self.assertFalse(metrics[2]["next_tts_prefetch_started"])  # nothing follows

    async def test_adaptive_does_not_start_on_a_shallow_buffer(self) -> None:
        speaker = self.build(AVATAR_TTS_PREFETCH_MIN_BUFFER_SECONDS="60")
        await self.speak(speaker)
        self.assertLess(self.first("rendered:1"), self.first("tts:2"))
        self.assertFalse(self.channel.of("metrics")[0]["next_tts_prefetch_started"])

    async def test_eager_starts_before_rendering(self) -> None:
        speaker = self.build(AVATAR_TTS_PREFETCH_POLICY="eager")
        await self.speak(speaker)
        self.assertLess(self.first("tts:2"), self.first("window:1.1"))
        self.assertEqual(self.channel.of("metrics")[0]["tts_prefetch_policy"], "eager")

    async def test_off_synthesizes_each_phrase_after_the_previous_render(self) -> None:
        speaker = self.build(AVATAR_TTS_PREFETCH="false")
        await self.speak(speaker)
        self.assertLess(self.first("rendered:1"), self.first("tts:2"))
        self.assertLess(self.first("rendered:2"), self.first("tts:3"))
        self.assertEqual(self.channel.of("metrics")[0]["tts_prefetch_policy"], "off")


class RefusalTests(SpeakerCase):
    async def error(self, speaker: Speaker, text: str = "Hello there, world.") -> str:
        await self.speak(speaker, text)
        self.assertEqual(self.tts.texts, [])
        return self.channel.of("error")[0]["message"]

    async def test_failed_startup_is_reported_on_every_request(self) -> None:
        speaker = self.build()
        speaker.unavailable = "No face was detected"
        self.assertEqual(await self.error(speaker), "No face was detected")

    async def test_busy_playback_empty_text_and_bad_voice_are_refused(self) -> None:
        speaker = self.build()
        self.assertIn("Enter text", await self.error(speaker, "   "))
        self.channel.events.clear()
        self.tts.voice_error = "Preset voice is not configured"
        self.assertIn("not configured", await self.error(speaker))
        self.channel.events.clear()
        self.playback.producing = True
        self.assertIn("Wait for the current speech", await self.error(speaker))

    async def test_a_render_failure_is_reported_and_leaves_things_clean(self) -> None:
        speaker = self.build()
        self.renderer.fail = "CUDA out of memory"
        await self.speak(speaker)
        self.assertEqual(self.channel.of("error")[0]["message"], "CUDA out of memory")
        self.assertFalse(self.playback.busy)
        self.assertFalse(speaker.lock.locked())
        self.assertEqual(self.channel.of("ready"), [])


class WarmupTests(SpeakerCase):
    async def test_the_idle_image_is_the_chosen_warmup_frame(self) -> None:
        speaker = self.build(AVATAR_WARMUP_IDLE_FRAME_INDEX="3")
        result = await speaker.warm_up()
        self.assertTrue(result.complete)
        self.assertIsNone(result.error)
        self.assertEqual(int(self.idle.frame[0, 0, 0]), 23)  # phrase 1, frame 3
        self.assertEqual((self.idle.source, self.idle.index), ("startup-warmup-neural-frame", 3))
        self.assertEqual(result.metrics["idle_frame_index"], 3)
        self.assertEqual(self.tts.texts, ["Hello."])
        self.assertEqual(self.renderer.fps_estimate, 40.0)  # raised by the measurement
        # A buffer created before the warm-up shows the new idle image too.
        self.assertEqual(int(self.playback.next_video()[0, 0, 0]), 23)

    async def test_an_index_past_the_end_takes_the_last_frame(self) -> None:
        speaker = self.build(AVATAR_WARMUP_IDLE_FRAME_INDEX="99")
        await speaker.warm_up()
        self.assertEqual(self.idle.index, 9)

    async def test_no_frames_is_a_reported_non_fatal_error(self) -> None:
        speaker = self.build()
        self.renderer.frames = 0
        result = await speaker.warm_up()
        self.assertFalse(result.complete)
        self.assertIn("no frame", result.error)
        self.assertEqual(int(self.idle.frame[0, 0, 0]), 255)  # the portrait stays
        self.assertEqual(self.idle.source, "original-avatar")

    async def test_the_portrait_stays_when_the_neural_idle_frame_is_off(self) -> None:
        speaker = self.build(AVATAR_USE_NEURAL_IDLE_FRAME="false")
        result = await speaker.warm_up()
        self.assertTrue(result.complete)
        self.assertEqual(int(self.idle.frame[0, 0, 0]), 255)


class RenderStrideTests(unittest.TestCase):
    def test_adaptive_stride_follows_the_buffer(self) -> None:
        settings = load({}).settings  # stride 1, catch-up 2 below 0.75 s
        self.assertEqual(select_render_stride(settings, 1, 0.0), (1, "first-phrase-quality"))
        self.assertEqual(select_render_stride(settings, 2, 0.5), (2, "low-buffer-catchup"))
        self.assertEqual(select_render_stride(settings, 2, 0.75), (1, "buffer-healthy"))

    def test_fixed_stride_ignores_the_buffer(self) -> None:
        fixed = load({"AVATAR_ADAPTIVE_RENDER_STRIDE": "false"}).settings
        self.assertEqual(select_render_stride(fixed, 3, 0.0), (1, "fixed"))
        same = load({"AVATAR_CATCHUP_RENDER_STRIDE": "1"}).settings
        self.assertEqual(select_render_stride(same, 3, 0.0), (1, "fixed"))


if __name__ == "__main__":
    unittest.main()
