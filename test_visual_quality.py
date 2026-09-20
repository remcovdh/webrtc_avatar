"""Regression checks for visual-quality configuration and recorded benchmarks."""

from pathlib import Path
import unittest


class VisualQualitySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parent
        cls.server = (root / "server.py").read_text(encoding="utf-8")
        cls.benchmark = (root / "benchmark_avatar.py").read_text(encoding="utf-8")
        cls.compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
        cls.quality_runner = (root / "quality_benchmark.sh").read_text(
            encoding="utf-8"
        )

    def test_flp_visual_controls_are_wired(self) -> None:
        expected = [
            "cfg.infer_params.animation_region = AVATAR_ANIMATION_REGION",
            "cfg.infer_params.driving_multiplier = AVATAR_DRIVING_MULTIPLIER",
            "cfg.infer_params.flag_normalize_lip = AVATAR_NORMALIZE_LIP",
            "cfg.infer_params.flag_eye_retargeting = AVATAR_EYE_RETARGETING",
            "cfg.infer_params.flag_lip_retargeting = AVATAR_LIP_RETARGETING",
        ]
        for value in expected:
            with self.subTest(value=value):
                self.assertIn(value, self.server)

    def test_compose_exposes_every_visual_control(self) -> None:
        for name in [
            "AVATAR_ANIMATION_REGION",
            "AVATAR_DRIVING_MULTIPLIER",
            "AVATAR_NORMALIZE_LIP",
            "AVATAR_EYE_RETARGETING",
            "AVATAR_LIP_RETARGETING",
        ]:
            with self.subTest(name=name):
                self.assertIn(f'{name}: "${{{name}:-', self.compose)

    def test_benchmark_can_record_each_run(self) -> None:
        self.assertIn("from aiortc.contrib.media import MediaRecorder", self.benchmark)
        self.assertIn('run.add_argument(\n        "--record-media"', self.benchmark)
        self.assertIn("await self.recorder.stop()", self.benchmark)

    def test_quality_profiles_separate_runtime_from_stride_one(self) -> None:
        self.assertIn("QUALITY_BENCHMARK_PROFILE:-runtime", self.quality_runner)
        self.assertIn("render_stride=2", self.quality_runner)
        self.assertIn("incremental_windows=true", self.quality_runner)
        self.assertIn("render_stride=1", self.quality_runner)
        self.assertIn("incremental_windows=false", self.quality_runner)


if __name__ == "__main__":
    unittest.main()
