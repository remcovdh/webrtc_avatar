"""Build-time guards for the neural startup/idle frame wiring."""

from pathlib import Path
import unittest


class NeuralIdleFrameSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path(__file__).with_name("server.py").read_text(encoding="utf-8")

    def test_idle_frame_reuses_warmup_output(self) -> None:
        self.assertIn("BASE_AVATAR = frames[selected_index].copy()", self.source)

    def test_empty_warmup_frames_fail_safely(self) -> None:
        self.assertIn("if not frames:", self.source)

    def test_health_reports_idle_source(self) -> None:
        self.assertIn('"idle_frame_source": idle_frame_source', self.source)


if __name__ == "__main__":
    unittest.main()
