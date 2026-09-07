"""Build-time guards for the configurable TTS boundary."""

from pathlib import Path
import unittest


class ConfigurableTtsSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).parent
        cls.server = (root / "server.py").read_text(encoding="utf-8")
        cls.adapter = (root / "chatterbox_api.py").read_text(encoding="utf-8")
        cls.compose = (root / "docker-compose.yml").read_text(encoding="utf-8")

    def test_generic_url_keeps_legacy_breeze_fallback(self) -> None:
        self.assertIn('"TTS_URL",', self.server)
        self.assertIn('os.getenv("BREEZE_TTS_URL"', self.server)

    def test_chatterbox_normalizes_to_24khz_int16(self) -> None:
        self.assertIn("OUTPUT_RATE = 24_000", self.adapter)
        self.assertIn(".to(torch.int16)", self.adapter)

    def test_compose_has_one_dynamic_tts_service(self) -> None:
        self.assertIn('target: "${TTS_PROVIDER:-chatterbox}"', self.compose)
        self.assertIn('command: ["${TTS_PROVIDER:-chatterbox}"]', self.compose)
        self.assertNotIn("  breeze-tts:\n", self.compose)

    def test_chatterbox_does_not_expose_breeze_direction_mode(self) -> None:
        self.assertIn('if TTS_PROVIDER == "breeze" else []', self.server)


if __name__ == "__main__":
    unittest.main()
