"""Regression checks for visual-quality configuration and recorded benchmarks."""

from pathlib import Path
import unittest


class VisualQualitySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.benchmark = (root / "scripts/benchmark_avatar.py").read_text(encoding="utf-8")
        cls.compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
        cls.quality_runner = (root / "scripts/quality_benchmark.sh").read_text(
            encoding="utf-8"
        )

    def test_compose_exposes_every_visual_control(self) -> None:
        for name in [
            "AVATAR_ANIMATION_REGION",
            "AVATAR_DRIVING_MULTIPLIER",
            "AVATAR_NORMALIZE_LIP",
            "AVATAR_EYE_RETARGETING",
            "AVATAR_LIP_RETARGETING",
        ]:
            with self.subTest(name=name):
                self.assertIn(f"      - {name}\n", self.compose)

    def test_benchmark_can_record_each_run(self) -> None:
        self.assertIn("from aiortc.contrib.media import MediaRecorder", self.benchmark)
        self.assertIn('run.add_argument(\n        "--record-media"', self.benchmark)
        self.assertIn("await self.recorder.stop()", self.benchmark)

    def test_quality_profiles_separate_runtime_from_stride_one(self) -> None:
        self.assertIn("QUALITY_BENCHMARK_PROFILE:-runtime", self.quality_runner)
        self.assertIn('avatar_profile="benchmark-${quality_profile}"', self.quality_runner)
        self.assertIn('AVATAR_PROFILE="${avatar_profile}"', self.quality_runner)


if __name__ == "__main__":
    unittest.main()
