"""Build-time guards for the phrase-continuity wiring."""

from pathlib import Path
import unittest


class MotionContinuitySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (Path(__file__).resolve().parents[2] / "src/avatar/server.py").read_text(encoding="utf-8")

    def test_only_first_phrase_resets_when_persistence_enabled(self) -> None:
        self.assertIn(
            "index == 1 or not PERSISTENT_PHRASE_MOTION",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
