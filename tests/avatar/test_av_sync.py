"""Tests for audio-clocked playback: the buffer, the speech gate, keep-alive noise."""

import unittest

import numpy as np

from avatar.playback import AUDIO_RATE, AUDIO_SAMPLES, VIDEO_FPS, IdleFrame, PlaybackBuffer
from avatar.tracks import keepalive_amplitude, keepalive_noise

RENDER_FPS = 12.5


def numbered_frames(start: int, count: int) -> list[np.ndarray]:
    """Tiny frames whose pixel value identifies the rendered frame index."""
    return [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(start, start + count)]


def play_audio(playback: PlaybackBuffer, seconds: float) -> None:
    for _ in range(round(seconds * AUDIO_RATE / AUDIO_SAMPLES)):
        playback.next_audio()


def pcm(seconds: float) -> np.ndarray:
    return np.ones(int(seconds * 24_000), dtype=np.int16)


class AudioClockedVideoTests(unittest.TestCase):
    def test_buffered_video_plays_every_frame_in_step_with_audio(self) -> None:
        playback = PlaybackBuffer()
        playback.begin()
        playback.begin_phrase_stream(pcm(1.0), RENDER_FPS, 13)
        playback.append_video_window(numbered_frames(0, 13))
        playback.finish_phrase_stream()

        shown = []
        for tick in range(VIDEO_FPS):
            # Audio runs ahead of the video clock by at most one video frame.
            while playback.playhead_seconds < tick / VIDEO_FPS:
                playback.next_audio()
            shown.append(int(playback.next_video()[0, 0, 0]))

        self.assertEqual(playback.video_dropped, 0)
        self.assertEqual(shown[0], 0)
        self.assertEqual(shown, sorted(shown))

    def test_late_render_skips_ahead_instead_of_lagging_the_voice(self) -> None:
        playback = PlaybackBuffer()
        playback.begin()
        playback.begin_phrase_stream(pcm(2.0), RENDER_FPS, 25)
        playback.append_video_window(numbered_frames(0, 8))
        playback.next_video()

        # Rendering stalls while 1.2 s of speech plays.
        play_audio(playback, 1.2)
        playback.append_video_window(numbered_frames(8, 17))
        frame = int(playback.next_video()[0, 0, 0])

        # Render frame 15 belongs at 1.2 s (15 / 12.5 fps); the old buffer
        # would have shown frame 1 here, roughly a second behind the voice.
        self.assertEqual(frame, 15)
        self.assertGreater(playback.video_dropped, 0)

    def test_next_phrase_starts_where_previous_audio_ended(self) -> None:
        playback = PlaybackBuffer()
        playback.begin()
        first = playback.begin_phrase_stream(pcm(0.5), RENDER_FPS, 7)
        playback.append_video_window(numbered_frames(0, 7))
        playback.finish_phrase_stream()
        play_audio(playback, 0.6)  # phrase audio plus a TTS gap of silence
        playback.next_video()

        playback.begin_phrase_stream(pcm(0.5), RENDER_FPS, 7)
        playback.append_video_window(numbered_frames(100, 7))

        # Phrase audio is padded to whole 20 ms WebRTC chunks.
        self.assertAlmostEqual(
            playback.video[0][0], first, delta=AUDIO_SAMPLES / AUDIO_RATE
        )
        self.assertEqual(int(playback.next_video()[0, 0, 0]), 100)


    def test_lip_sync_offset_shows_the_mouth_later(self) -> None:
        playback = PlaybackBuffer(lip_sync_offset_ms=200.0)
        playback.begin()
        playback.begin_phrase_stream(pcm(1.0), RENDER_FPS, 13)
        playback.append_video_window(numbered_frames(0, 13))
        play_audio(playback, 0.62)
        # 0.62 s of audio with the mouth 0.2 s later shows the 0.42 s
        # moment: render frame 5 (12.5 fps).
        self.assertEqual(int(playback.next_video()[0, 0, 0]), 5)


    def test_offset_video_tail_plays_out_after_the_voice_ends(self) -> None:
        playback = PlaybackBuffer(lip_sync_offset_ms=80.0)
        playback.begin()
        # 0.8 s fills whole 20 ms chunks, so no padding hides the tail.
        playback.begin_phrase_stream(pcm(0.8), RENDER_FPS, 10)
        playback.append_video_window(numbered_frames(0, 10))
        playback.finish_phrase_stream()
        playback.finish()
        play_audio(playback, 1.2)  # 0.8 s of speech, then silence
        for _ in range(VIDEO_FPS):
            playback.next_video()
        # Without catching up, the last 80 ms of frames never became due
        # and the request waited on `busy` forever.
        self.assertFalse(playback.busy)

    def test_split_windows_do_not_accumulate_rounding_drift(self) -> None:
        # 24 frames at 12.5 fps are 1.92 s = 58 frames at 30 fps. Appending
        # them as three windows must give exactly those 58, not 3 x 20.
        playback = PlaybackBuffer()
        playback.begin()
        playback.begin_phrase_stream(pcm(1.92), RENDER_FPS, 24)
        counts = []
        for start in (0, 8, 16):
            before = len(playback.video)
            playback.append_video_window(numbered_frames(start, 8))
            counts.append(len(playback.video) - before)
        self.assertEqual(counts, [20, 19, 19])

    def test_audio_waits_for_the_first_video_window(self) -> None:
        # Phrase audio is queued before the renderer has produced anything.
        # Speech must not start on the idle image.
        idle = IdleFrame(np.full((2, 2, 3), 99, dtype=np.uint8))
        playback = PlaybackBuffer(idle)
        playback.begin()
        playback.begin_phrase_stream(pcm(1.0), RENDER_FPS, 13)
        play_audio(playback, 0.3)
        self.assertEqual(playback.playhead_seconds, 0.0)
        self.assertFalse(playback.next_audio().any())
        self.assertEqual(int(playback.next_video()[0, 0, 0]), 99)

        playback.append_video_window(numbered_frames(0, 8))
        self.assertTrue(playback.next_audio().any())
        self.assertEqual(int(playback.next_video()[0, 0, 0]), 0)

    def test_idle_image_follows_the_shared_holder(self) -> None:
        idle = IdleFrame(np.full((2, 2, 3), 1, dtype=np.uint8))
        playback = PlaybackBuffer(idle)
        self.assertEqual(int(playback.next_video()[0, 0, 0]), 1)
        idle.frame = np.full((2, 2, 3), 2, dtype=np.uint8)  # warm-up replaces it
        self.assertEqual(int(playback.next_video()[0, 0, 0]), 2)

    def test_begin_resets_every_counter(self) -> None:
        playback = PlaybackBuffer()
        playback.begin()
        playback.begin_phrase_stream(pcm(0.5), RENDER_FPS, 7, render_rate=0.5)
        playback.append_video_window(numbered_frames(0, 7))
        play_audio(playback, 0.3)
        playback.clear()
        fresh = PlaybackBuffer()
        skip = {"idle", "video", "audio", "last_video"}
        for name, value in vars(fresh).items():
            if name not in skip:
                self.assertEqual(getattr(playback, name), value, name)


class KeepaliveNoiseTests(unittest.TestCase):
    def test_silence_becomes_inaudible_noise_and_speech_is_untouched(self) -> None:
        rng = np.random.default_rng(0)
        silence = np.zeros((1, AUDIO_SAMPLES), dtype=np.int16)
        amplitude = keepalive_amplitude(-60.0)
        noise = keepalive_noise(silence, rng, amplitude)
        self.assertTrue(noise.any())
        rms_dbfs = 20 * np.log10(np.sqrt(np.mean(noise.astype(float) ** 2)) / 32767)
        self.assertLess(rms_dbfs, -55)
        speech = np.full((1, AUDIO_SAMPLES), 1000, dtype=np.int16)
        self.assertIs(keepalive_noise(speech, rng, amplitude), speech)
        self.assertIs(keepalive_noise(silence, rng, keepalive_amplitude(None)), silence)


class SpeechStartGateTests(unittest.TestCase):
    def test_slow_render_holds_speech_until_enough_video_exists(self) -> None:
        playback = PlaybackBuffer()
        playback.begin()
        # Half real time with 0.64 s windows:
        # (0.64 + (2 - 0.64) * 0.5) * 1.15 = 1.518 s of video first.
        playback.begin_phrase_stream(
            pcm(2.0), RENDER_FPS, 25, render_rate=0.5, window_seconds=0.64
        )
        self.assertAlmostEqual(playback.speech_lead_seconds, 1.518)

        playback.append_video_window(numbered_frames(0, 8))  # 0.64 s
        playback.append_video_window(numbered_frames(8, 8))  # 1.28 s
        play_audio(playback, 0.5)
        self.assertEqual(playback.playhead_seconds, 0.0)
        self.assertEqual(playback.speech_holds, 25)
        self.assertEqual(int(playback.next_video()[0, 0, 0]), 0)

        playback.append_video_window(numbered_frames(16, 8))  # 1.92 s
        play_audio(playback, 0.5)
        self.assertAlmostEqual(playback.playhead_seconds, 0.5)

    def test_real_time_render_or_unknown_rate_does_not_hold(self) -> None:
        for rate in (None, 1.0, 1.4):
            playback = PlaybackBuffer()
            playback.begin()
            playback.begin_phrase_stream(pcm(1.0), RENDER_FPS, 13, render_rate=rate)
            playback.append_video_window(numbered_frames(0, 2))
            play_audio(playback, 0.2)
            self.assertAlmostEqual(playback.playhead_seconds, 0.2)
            self.assertEqual(playback.speech_holds, 0)

    def test_finished_render_releases_a_short_phrase(self) -> None:
        playback = PlaybackBuffer()
        playback.begin()
        playback.begin_phrase_stream(pcm(1.0), RENDER_FPS, 13, render_rate=0.1)
        playback.append_video_window(numbered_frames(0, 5))
        playback.finish_phrase_stream()
        play_audio(playback, 0.2)
        self.assertAlmostEqual(playback.playhead_seconds, 0.2)


if __name__ == "__main__":
    unittest.main()
