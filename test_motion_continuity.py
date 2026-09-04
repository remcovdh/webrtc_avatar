"""Build-time guards for the phrase-continuity wiring."""

from pathlib import Path
import unittest


class MotionContinuitySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path(__file__).with_name("server.py").read_text(encoding="utf-8")

    def test_first_frame_is_gated_by_request_reset(self) -> None:
        self.assertIn(
            "first_frame=reset_motion_reference and frame_index == 0",
            self.source,
        )

    def test_only_first_phrase_resets_when_persistence_enabled(self) -> None:
        self.assertIn(
            "index == 1 or not PERSISTENT_PHRASE_MOTION",
            self.source,
        )

    def test_relative_motion_is_configurable(self) -> None:
        self.assertIn(
            "cfg.infer_params.flag_relative_motion = AVATAR_RELATIVE_MOTION",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
