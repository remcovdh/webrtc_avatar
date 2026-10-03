"""Tests for the renderer: motion adjustment, FLP parameters, the frame loop.

They import FasterLivePortrait, so they run inside the avatar image; the frame
loop is tested with a stand-in pipeline and needs no GPU.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from avatar import renderer
from avatar.config import load
from avatar.renderer import Renderer, adjust_driving_motion, flp_infer_params


class ExpressionMotionScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = {"exp": np.zeros((1, 21, 3), dtype=np.float32)}
        self.motion = {"exp": np.ones((1, 21, 3), dtype=np.float32), "t": 7}

    def test_scales_eye_and_lip_keypoints_independently(self) -> None:
        scaled = adjust_driving_motion(
            self.motion, self.reference, eye_scale=0.25, lip_scale=1.5,
            lip_mode="relative", head_scale=1.0,
        )
        eyes = renderer.EYE_EXPRESSION_INDICES
        lips = renderer.LIP_EXPRESSION_INDICES
        others = [index for index in range(21) if index not in eyes + lips]

        np.testing.assert_allclose(scaled["exp"][:, eyes, :], 0.25)
        np.testing.assert_allclose(scaled["exp"][:, lips, :], 1.5)
        np.testing.assert_allclose(scaled["exp"][:, others, :], 1.0)
        self.assertEqual(scaled["t"], 7)
        np.testing.assert_allclose(self.motion["exp"], 1.0)  # input untouched

    def test_scale_one_or_missing_reference_is_a_no_op(self) -> None:
        scale = adjust_driving_motion
        self.assertIs(
            scale(self.motion, self.reference, eye_scale=1.0, lip_scale=1.0,
                  lip_mode="relative", head_scale=1.0),
            self.motion,
        )
        self.assertIs(
            scale(self.motion, None, np.zeros((1, 21, 3)), eye_scale=0.0,
                  lip_scale=1.0, lip_mode="absolute", head_scale=0.3),
            self.motion,
        )
        # Without relative motion FLP ignores the reference, so nothing is scaled.
        self.assertIs(
            scale(self.motion, self.reference, eye_scale=0.0, lip_scale=1.0,
                  lip_mode="relative", head_scale=1.0, relative_motion=False),
            self.motion,
        )

    def test_head_scale_damps_rotation_and_translation(self) -> None:
        def pose(pitch: float, t: float) -> dict:
            angle = np.array([[pitch]], dtype=np.float32)
            zero = np.zeros((1, 1), dtype=np.float32)
            return {
                "exp": np.zeros((1, 21, 3), dtype=np.float32),
                "pitch": angle, "yaw": zero, "roll": zero,
                "R": renderer.get_rotation_matrix(angle, zero, zero),
                "t": np.full((1, 3), t, dtype=np.float32),
            }

        adjusted = adjust_driving_motion(
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
        adjusted = adjust_driving_motion(
            driving, reference, source, eye_scale=1.0, lip_scale=1.0,
            lip_mode="absolute", head_scale=1.0,
        )
        # FLP relative motion: source + (driving - reference).
        animated = source + (adjusted["exp"] - reference["exp"])
        lips = renderer.LIP_EXPRESSION_INDICES
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
        without = Renderer(load({"AVATAR_TENSORRT": "false"}).settings)
        without._enable_tensorrt_warping(loaded)
        self.assertEqual(Model.predictor.onnx_model, "cuda-session")
        self.assertEqual(without.warping_backend, "cuda")

        unavailable = Renderer(load({}).settings)
        with patch.object(
            renderer.ort, "get_available_providers", lambda: ["CUDAExecutionProvider"]
        ):
            unavailable._enable_tensorrt_warping(loaded)
        self.assertEqual(Model.predictor.onnx_model, "cuda-session")
        self.assertEqual(unavailable.warping_backend, "cuda")


class SourceCropTests(unittest.TestCase):
    def test_crop_past_the_photo_edge_has_no_black_border(self) -> None:
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        # A face near the top-left corner: the 2.3x crop leaves the photo.
        rng = np.random.default_rng(0)
        pts = (rng.random((106, 2)) * 20 + 5).astype(np.float32)
        crop = renderer.crop_source_image(image, pts, dsize=64, scale=2.3)
        self.assertEqual(crop["img_crop"].shape, (64, 64, 3))
        self.assertEqual(int(crop["img_crop"].min()), 200)
        # Upright by default: the crop transform has no rotation component.
        np.testing.assert_allclose(crop["M_o2c"][0, 1], 0.0, atol=1e-6)


class InferParamsTests(unittest.TestCase):
    def test_settings_reach_faster_live_portrait(self) -> None:
        params = flp_infer_params(load({
            "AVATAR_ANIMATION_REGION": "lip",
            "AVATAR_DRIVING_MULTIPLIER": "1.3",
            "AVATAR_NORMALIZE_LIP": "false",
            "AVATAR_EYE_RETARGETING": "true",
            "AVATAR_LIP_RETARGETING": "true",
            "AVATAR_RELATIVE_MOTION": "false",
        }).settings)
        self.assertEqual(params["animation_region"], "lip")
        self.assertEqual(params["driving_multiplier"], 1.3)
        self.assertFalse(params["flag_normalize_lip"])
        self.assertTrue(params["flag_eye_retargeting"])
        self.assertTrue(params["flag_lip_retargeting"])
        self.assertFalse(params["flag_relative_motion"])

    def test_defaults_keep_the_crop_output_and_relative_motion(self) -> None:
        params = flp_infer_params(load({}).settings)
        self.assertFalse(params["flag_pasteback"])
        self.assertTrue(params["flag_relative_motion"])
        self.assertTrue(params["flag_stitching"])


class FakePipeline:
    """Stands in for FasterLivePortrait + JoyVASA: 10 motion frames at 25 fps."""

    is_source_video = False

    def __init__(self, frames: int = 10) -> None:
        exp = np.zeros((1, 21, 3), dtype=np.float32)
        zero = np.zeros((1, 1), dtype=np.float32)
        pose = {
            "exp": exp, "pitch": zero, "yaw": zero, "roll": zero,
            "t": np.zeros((1, 3), dtype=np.float32),
            "R": renderer.get_rotation_matrix(zero, zero, zero),
        }
        self.motion = [{**pose, "id": index} for index in range(frames)]
        self.joyvasa_pipe = self
        self.src_imgs = ["source-image"]
        self.src_infos = [[[{"exp": exp}]]]
        self.R_d_0 = None
        self.x_d_0_info = None
        self.calls: list[tuple[int, bool]] = []

    def gen_motion_sequence(self, wav_path: str) -> dict:
        return {"motion": self.motion, "output_fps": 25.0}

    def run_with_pkl(self, driving, src_img, src_info, first_frame=False):
        self.calls.append((driving[0]["id"], first_frame))
        if first_frame:  # FLP stores its motion reference on the first frame
            self.R_d_0, self.x_d_0_info = "set", driving[0]
        return [np.full((4, 4, 3), driving[0]["id"], dtype=np.uint8)]


def renderer_with(pipeline: FakePipeline, **environment: str) -> Renderer:
    built = Renderer(load(environment).settings)
    built.pipeline = pipeline
    return built


class RenderLoopTests(unittest.TestCase):
    def test_not_loaded_is_reported(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            Renderer(load({}).settings).render(Path("speech.wav"))

    def test_only_a_reset_phrase_marks_its_first_frame(self) -> None:
        pipeline = FakePipeline(4)
        built = renderer_with(pipeline)
        built.render(Path("a.wav"), reset_motion_reference=True)
        self.assertEqual(pipeline.calls, [(0, True), (1, False), (2, False), (3, False)])
        pipeline.calls.clear()
        # A later phrase of the same utterance keeps the first one's reference.
        built.render(Path("b.wav"), reset_motion_reference=False)
        self.assertEqual([first for _frame, first in pipeline.calls], [False] * 4)

    def test_stride_skips_motion_frames_and_lowers_playback_fps(self) -> None:
        pipeline = FakePipeline(10)
        frames, fps, detail = renderer_with(pipeline).render(Path("a.wav"), render_stride=2)
        self.assertEqual([frame for frame, _first in pipeline.calls], [0, 2, 4, 6, 8])
        self.assertEqual((len(frames), fps), (5, 12.5))
        self.assertEqual(frames[0].shape, (512, 512, 3))  # letterboxed
        self.assertEqual(
            (detail["render_stride"], detail["frames"], detail["motion_frames"]),
            (2, 5, 10),
        )
        self.assertFalse(detail["incremental_windows"])

    def test_default_stride_comes_from_the_settings(self) -> None:
        pipeline = FakePipeline(6)
        _frames, fps, detail = renderer_with(
            pipeline, AVATAR_RENDER_STRIDE="3", AVATAR_CATCHUP_RENDER_STRIDE="3"
        ).render(Path("a.wav"))
        self.assertEqual(detail["render_stride"], 3)
        self.assertAlmostEqual(fps, 25 / 3)

    def test_windows_are_published_in_order_with_a_short_tail(self) -> None:
        pipeline = FakePipeline(10)
        windows: list[tuple[list[int], float, dict]] = []

        def collect(frames, fps, info):
            windows.append(([int(frame[256, 256, 0]) for frame in frames], fps, info))

        frames, _fps, detail = renderer_with(
            pipeline, AVATAR_RENDER_WINDOW_FRAMES="4"
        ).render(Path("a.wav"), window_callback=collect)
        self.assertEqual(frames, [])  # everything went through the callback
        self.assertEqual([ids for ids, _fps, _info in windows], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
        self.assertEqual([info["index"] for _ids, _fps, info in windows], [1, 2, 3])
        self.assertTrue(all(info["expected_rendered_frames"] == 10 for _i, _f, info in windows))
        self.assertEqual(
            (detail["window_count"], detail["window_size"], detail["frames"]), (3, 4, 10)
        )
        self.assertTrue(detail["incremental_windows"])

    def catch_up_run(self, wanted: list[bool], **environment: str):
        """Render 16 frames in windows of 4; `wanted[n]` answers window n+1."""
        pipeline = FakePipeline(16)
        shown: list[list[int]] = []

        def collect(frames, fps, info):
            shown.append([int(frame[256, 256, 0]) for frame in frames])
            return wanted[info["index"] - 1]

        environment.setdefault("AVATAR_RENDER_WINDOW_FRAMES", "4")
        _frames, fps, detail = renderer_with(pipeline, **environment).render(
            Path("a.wav"), window_callback=collect
        )
        return pipeline, shown, fps, detail

    def test_a_window_that_must_catch_up_renders_every_second_frame(self) -> None:
        pipeline, shown, fps, detail = self.catch_up_run([True, False, True, False])
        self.assertEqual(shown, [
            [0, 1, 2, 3],        # full
            [4, 4, 6, 6],        # catching up: each rendered frame shown twice
            [8, 9, 10, 11],      # full again
            [12, 12, 14, 14],
        ])
        self.assertEqual(fps, 25.0)  # the stream's frame rate does not change
        self.assertEqual([frame for frame, _first in pipeline.calls],
                         [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 14])
        self.assertEqual((detail["frames"], detail["held_frames"]), (16, 4))

    def test_no_catch_up_without_adaptive_stride_or_at_the_catch_up_stride(self) -> None:
        always = [True] * 4
        _p, shown, _fps, detail = self.catch_up_run(
            always, AVATAR_ADAPTIVE_RENDER_STRIDE="false"
        )
        self.assertEqual(detail["held_frames"], 0)
        self.assertEqual(shown[1], [4, 5, 6, 7])
        # A phrase already rendered at the catch-up stride cannot go lower.
        pipeline = FakePipeline(16)
        _frames, _fps, detail = renderer_with(pipeline).render(
            Path("a.wav"), render_stride=2, window_callback=lambda *_: True
        )
        self.assertEqual(detail["held_frames"], 0)

    def test_speed_estimate_is_blended_and_warmup_only_raises_it(self) -> None:
        built = Renderer(load({"AVATAR_EXPECTED_RENDER_FPS": "10"}).settings)
        built.measured_at_warmup(4.0)
        self.assertEqual(built.fps_estimate, 10.0)
        built.measured_at_warmup(30.0)
        self.assertEqual(built.fps_estimate, 30.0)
        built.measured(20.0)
        self.assertEqual(built.fps_estimate, 25.0)
        built.measured(0.0)  # a phrase without a measurement changes nothing
        self.assertEqual(built.fps_estimate, 25.0)




if __name__ == "__main__":
    unittest.main()
