"""Build-time guards for the TTS boundary."""

from pathlib import Path
import unittest


class TtsSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).parent
        cls.server = (root / "server.py").read_text(encoding="utf-8")
        cls.adapter = (root / "chatterbox_api.py").read_text(encoding="utf-8")
        cls.compose = (root / "docker-compose.yml").read_text(encoding="utf-8")

    def test_chatterbox_normalizes_to_24khz_int16(self) -> None:
        self.assertIn("OUTPUT_RATE = 24_000", self.adapter)
        self.assertIn(".to(torch.int16)", self.adapter)

    def test_compose_runs_the_chatterbox_target(self) -> None:
        self.assertIn("target: chatterbox", self.compose)
        self.assertIn('command: ["chatterbox"]', self.compose)

    def test_voice_modes_are_builtin_and_clone(self) -> None:
        self.assertIn(
            "VOICE_MODES = {VOICE_MODE_DESIGN, VOICE_MODE_PRESET_CLONE}",
            self.server,
        )


if __name__ == "__main__":
    unittest.main()
