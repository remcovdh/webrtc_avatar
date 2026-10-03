"""Tests for the TTS client: voice modes and the synthesize round trip."""

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from avatar import tts_client
from avatar.config import load
from avatar.tts_client import TtsClient, VoiceRequest


def client_with_preset(folder: Path, content: bytes | None = b"RIFF....") -> TtsClient:
    preset = folder / "voice.wav"
    if content is not None:
        preset.write_bytes(content)
    return TtsClient(load({"AVATAR_PRESET_AUDIO_PATH": str(preset)}).settings)


class VoiceModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())

    def test_default_mode_clones_the_reference_recording(self) -> None:
        client = client_with_preset(self.folder)
        voice = client.resolve_voice("")
        self.assertEqual(voice.mode, "preset-clone")
        self.assertEqual(voice.reference_path, self.folder / "voice.wav")

    def test_design_is_the_builtin_voice_without_a_reference(self) -> None:
        client = client_with_preset(self.folder)
        self.assertEqual(client.resolve_voice("design"), VoiceRequest(mode="design"))

    def test_aliases_and_case_are_accepted(self) -> None:
        client = client_with_preset(self.folder)
        for name in ("preset", "clone", " Preset-Clone "):
            self.assertEqual(client.resolve_voice(name).mode, "preset-clone")

    def test_missing_or_empty_reference_is_explained(self) -> None:
        for content, reason in ((None, "Missing"), (b"", "empty")):
            client = client_with_preset(Path(tempfile.mkdtemp()), content)
            self.assertEqual(client.preset_configuration()[0], False)
            with self.assertRaisesRegex(ValueError, reason):
                client.resolve_voice("preset-clone")
            # The built-in voice needs no reference.
            self.assertEqual(client.resolve_voice("design").mode, "design")

    def test_removed_and_unknown_modes_are_rejected(self) -> None:
        client = client_with_preset(self.folder)
        for name in ("preset-direction", "robot"):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                client.resolve_voice(name)


class SynthesizeTests(unittest.IsolatedAsyncioTestCase):
    """The HTTP round trip against a stand-in for the TTS service."""

    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())
        self.requests: list[httpx.Request] = []
        self.reply = httpx.Response(200, content=np.arange(5, dtype="<i2").tobytes())

        def handler(request: httpx.Request) -> httpx.Response:
            request.read()
            self.requests.append(request)
            return self.reply

        real = httpx.AsyncClient
        patcher = patch.object(
            tts_client.httpx,
            "AsyncClient",
            lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_pcm_comes_back_as_samples_and_a_wav_file(self) -> None:
        client = client_with_preset(self.folder)
        pcm, detail = await client.synthesize(
            "Hello.", client.resolve_voice("design"),
            self.folder / "speech.pcm", self.folder / "speech.wav",
        )
        np.testing.assert_array_equal(pcm, np.arange(5))
        with wave.open(str(self.folder / "speech.wav")) as wav_file:
            self.assertEqual(
                (wav_file.getframerate(), wav_file.getnchannels(), wav_file.getnframes()),
                (24_000, 1, 5),
            )
        self.assertEqual(detail["bytes"], 10)
        self.assertEqual(detail["provider"], "chatterbox")
        self.assertFalse(detail["reference_used"])
        body = self.requests[0].content
        self.assertIn(b'name="text"', body)
        self.assertIn(b'name="seed"', body)
        self.assertNotIn(b"ref_audio", body)
        self.assertTrue(str(self.requests[0].url).endswith("/v1/audio/speech"))

    async def test_clone_mode_sends_the_reference_recording(self) -> None:
        client = client_with_preset(self.folder, b"reference-bytes")
        _pcm, detail = await client.synthesize(
            "Hello.", client.resolve_voice("preset-clone"),
            self.folder / "speech.pcm", self.folder / "speech.wav",
        )
        self.assertTrue(detail["reference_used"])
        self.assertEqual(detail["reference_bytes"], len(b"reference-bytes"))
        self.assertIn(b"reference-bytes", self.requests[0].content)

    async def test_a_stray_last_byte_is_dropped(self) -> None:
        self.reply = httpx.Response(200, content=b"\x01\x00\x02\x00\x03")
        client = client_with_preset(self.folder)
        pcm, _detail = await client.synthesize(
            "Hi.", VoiceRequest("design"),
            self.folder / "speech.pcm", self.folder / "speech.wav",
        )
        np.testing.assert_array_equal(pcm, [1, 2])

    async def test_service_errors_and_empty_audio_raise(self) -> None:
        client = client_with_preset(self.folder)
        for reply, message in (
            (httpx.Response(503, text="loading"), "TTS failed .503.: loading"),
            (httpx.Response(200, content=b""), "empty audio"),
        ):
            self.reply = reply
            with self.assertRaisesRegex(RuntimeError, message):
                await client.synthesize(
                    "Hi.", VoiceRequest("design"),
                    self.folder / "speech.pcm", self.folder / "speech.wav",
                )

    async def test_readiness_follows_the_health_endpoint(self) -> None:
        client = client_with_preset(self.folder)
        self.assertTrue(await client.is_ready())
        self.reply = httpx.Response(503)
        self.assertFalse(await client.is_ready())
        self.assertTrue(str(self.requests[-1].url).endswith("/health"))


if __name__ == "__main__":
    unittest.main()
