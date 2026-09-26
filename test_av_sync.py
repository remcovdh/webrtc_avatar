"""Build-time tests for audio-clocked playback and eye/mouth motion scaling.

These import the real server module, so they run inside the avatar image.
"""

import unittest

import numpy as np

import server
from server import AUDIO_RATE, AUDIO_SAMPLES, VIDEO_FPS, PlaybackBuffer

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


class ExpressionMotionScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = {"exp": np.zeros((1, 21, 3), dtype=np.float32)}
        self.motion = {"exp": np.ones((1, 21, 3), dtype=np.float32), "t": 7}

    def test_scales_eye_and_lip_keypoints_independently(self) -> None:
        scaled = server._adjust_driving_motion(
            self.motion, self.reference, eye_scale=0.25, lip_scale=1.5,
            lip_mode="relative",
        )
        eyes = server.EYE_EXPRESSION_INDICES
        lips = server.LIP_EXPRESSION_INDICES
        others = [index for index in range(21) if index not in eyes + lips]

        np.testing.assert_allclose(scaled["exp"][:, eyes, :], 0.25)
        np.testing.assert_allclose(scaled["exp"][:, lips, :], 1.5)
        np.testing.assert_allclose(scaled["exp"][:, others, :], 1.0)
        self.assertEqual(scaled["t"], 7)
        np.testing.assert_allclose(self.motion["exp"], 1.0)  # input untouched

    def test_scale_one_or_missing_reference_is_a_no_op(self) -> None:
        scale = server._adjust_driving_motion
        self.assertIs(
            scale(self.motion, self.reference, eye_scale=1.0, lip_scale=1.0,
                  lip_mode="relative"),
            self.motion,
        )
        self.assertIs(
            scale(self.motion, None, np.zeros((1, 21, 3)), eye_scale=0.0),
            self.motion,
        )

    def test_head_scale_damps_rotation_and_translation(self) -> None:
        def pose(pitch: float, t: float) -> dict:
            angle = np.array([[pitch]], dtype=np.float32)
            zero = np.zeros((1, 1), dtype=np.float32)
            return {
                "exp": np.zeros((1, 21, 3), dtype=np.float32),
                "pitch": angle, "yaw": zero, "roll": zero,
                "R": server.get_rotation_matrix(angle, zero, zero),
                "t": np.full((1, 3), t, dtype=np.float32),
            }

        adjusted = server._adjust_driving_motion(
            pose(4.0, 0.1), pose(0.0, 0.0), eye_scale=1.0, lip_scale=1.0,
            lip_mode="relative", head_scale=0.25,
        )
        np.testing.assert_allclose(adjusted["pitch"], 1.0)
        np.testing.assert_allclose(adjusted["t"], 0.025, rtol=1e-6)
        np.testing.assert_allclose(adjusted["R"], pose(1.0, 0.0)["R"], atol=1e-6)

    def test_absolute_lips_make_flp_output_joyvasa_mouth(self) -> None:
        source = np.full((1, 21, 3), 0.5, dtype=np.float32)
        reference = {"exp": np.full((1, 21, 3), 0.2, dtype=np.float32)}
        driving = {"exp": np.full((1, 21, 3), 0.9, dtype=np.float32)}
        adjusted = server._adjust_driving_motion(
            driving, reference, source, eye_scale=1.0, lip_mode="absolute"
        )
        # FLP relative motion: source + (driving - reference).
        animated = source + (adjusted["exp"] - reference["exp"])
        lips = server.LIP_EXPRESSION_INDICES
        others = [index for index in range(21) if index not in lips]
        np.testing.assert_allclose(animated[:, lips, :], 0.9, rtol=1e-6)
        np.testing.assert_allclose(animated[:, others, :], 1.2, rtol=1e-6)


class TensorRTWarpingTests(unittest.TestCase):
    def test_disabled_or_unavailable_tensorrt_keeps_the_cuda_session(self) -> None:
        class Predictor:
            onnx_model = "cuda-session"

        class Model:
            predictor = Predictor()
            kwargs = {"model_path": "/nonexistent/warping_spade.onnx"}

        loaded = type("Loaded", (), {"model_dict": {"warping_spade": Model()}})()
        original = server.AVATAR_TENSORRT, server.ort.get_available_providers
        try:
            server.AVATAR_TENSORRT = False
            server._enable_tensorrt_warping(loaded)
            self.assertEqual(Model.predictor.onnx_model, "cuda-session")

            server.AVATAR_TENSORRT = True
            server.ort.get_available_providers = lambda: ["CUDAExecutionProvider"]
            server._enable_tensorrt_warping(loaded)
            self.assertEqual(Model.predictor.onnx_model, "cuda-session")
            self.assertEqual(server.warping_backend, "cuda")
        finally:
            server.AVATAR_TENSORRT, server.ort.get_available_providers = original


if __name__ == "__main__":
    unittest.main()
