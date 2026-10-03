"""Client for the TTS service (Chatterbox behind `tts/chatterbox_api.py`).

The service takes a text, optionally with a reference recording whose voice it
clones, and returns raw 24 kHz mono PCM.
"""

from __future__ import annotations

import asyncio
import logging
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from avatar.config import AvatarSettings

LOG = logging.getLogger("avatar.tts")

PROVIDER = "chatterbox"
HEALTH_PATH = "/health"
SEED = 42
# How long the warm-up waits for the TTS service, and how often it checks.
STARTUP_WAIT_SECONDS = 300.0
STARTUP_POLL_SECONDS = 2.0

# "design" is Chatterbox's built-in voice; "preset-clone" clones the reference
# recording.
VOICE_MODE_DESIGN = "design"
VOICE_MODE_PRESET_CLONE = "preset-clone"
VOICE_MODES = {VOICE_MODE_DESIGN, VOICE_MODE_PRESET_CLONE}
_ALIASES = {"preset": VOICE_MODE_PRESET_CLONE, "clone": VOICE_MODE_PRESET_CLONE}


@dataclass(frozen=True)
class VoiceRequest:
    mode: str
    reference_path: Path | None = None


class TtsClient:
    def __init__(self, settings: AvatarSettings) -> None:
        self.url = settings.tts_url.rstrip("/")
        self.default_voice_mode = settings.default_voice_mode
        # Reference recording for the cloned voice (preset-clone mode).
        self.preset_audio_path = Path(settings.preset_audio_path)

    def preset_configuration(self) -> tuple[bool, str]:
        """Whether the reference recording is usable, and why not."""
        if not self.preset_audio_path.is_file():
            return False, f"Missing {self.preset_audio_path}"
        if self.preset_audio_path.stat().st_size == 0:
            return False, f"Preset audio is empty: {self.preset_audio_path}"
        return True, "ready"

    def resolve_voice(self, mode: str) -> VoiceRequest:
        normalized = (mode or self.default_voice_mode).strip().lower()
        normalized = _ALIASES.get(normalized, normalized)
        if normalized not in VOICE_MODES:
            raise ValueError(
                f"Voice mode {normalized!r} is unavailable; "
                f"choose {', '.join(sorted(VOICE_MODES))}."
            )
        if normalized == VOICE_MODE_DESIGN:
            return VoiceRequest(mode=normalized)

        configured, problem = self.preset_configuration()
        if not configured:
            raise ValueError(
                f"Preset voice is not configured: {problem}. Add a clean WAV, "
                "then recreate the avatar service."
            )
        return VoiceRequest(mode=normalized, reference_path=self.preset_audio_path)

    async def is_ready(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.url}{HEALTH_PATH}")
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def synthesize(
        self,
        text: str,
        voice: VoiceRequest,
        pcm_path: Path,
        wav_path: Path,
    ) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
        """Call the TTS service and normalize its raw 24 kHz PCM."""
        request_started = time.perf_counter()
        form = {
            "text": (None, text),
            "seed": (None, str(SEED)),
        }
        reference_bytes = 0
        if voice.reference_path is not None:
            payload = voice.reference_path.read_bytes()
            reference_bytes = len(payload)
            form["ref_audio"] = (
                voice.reference_path.name,
                payload,
                "audio/wav",
            )

        timeout = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=15.0)
        headers_ready = request_started
        first_byte_at: float | None = None
        body_complete = request_started
        bytes_received = 0

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{self.url}/v1/audio/speech", files=form
            ) as response:
                headers_ready = time.perf_counter()
                if response.is_error:
                    detail = (await response.aread()).decode("utf-8", errors="replace")[:300]
                    raise RuntimeError(
                        f"{PROVIDER} TTS failed ({response.status_code}): {detail}"
                    )
                with pcm_path.open("wb") as output:
                    async for chunk in response.aiter_bytes():
                        if first_byte_at is None:
                            first_byte_at = time.perf_counter()
                        bytes_received += len(chunk)
                        output.write(chunk)
                body_complete = time.perf_counter()

        finalize_started = time.perf_counter()
        raw = pcm_path.read_bytes()
        if len(raw) < 2:
            raise RuntimeError(f"{PROVIDER} returned an empty audio stream")
        if len(raw) % 2:
            raw = raw[:-1]
        pcm = np.frombuffer(raw, dtype="<i2").copy()
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24_000)
            wav_file.writeframes(pcm.tobytes())
        complete = time.perf_counter()
        first_byte_at = first_byte_at or body_complete
        return pcm, {
            "headers_ms": round((headers_ready - request_started) * 1000),
            "first_byte_ms": round((first_byte_at - request_started) * 1000),
            "download_ms": round((body_complete - first_byte_at) * 1000),
            "finalize_ms": round((complete - finalize_started) * 1000),
            "total_ms": round((complete - request_started) * 1000),
            "bytes": bytes_received,
            "voice_mode": voice.mode,
            "reference_used": voice.reference_path is not None,
            "reference_bytes": reference_bytes,
            "provider": PROVIDER,
        }

    async def wait_until_ready(self) -> float:
        """Wait for the TTS service before startup warm-up; returns the wait."""
        started = time.perf_counter()
        deadline = started + STARTUP_WAIT_SECONDS
        last_problem = "no response"
        announced = False

        while True:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{self.url}{HEALTH_PATH}")
                if response.status_code < 500:
                    waited = time.perf_counter() - started
                    if announced:
                        LOG.info("%s became ready after %.3fs", PROVIDER, waited)
                    return waited
                last_problem = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_problem = f"{type(exc).__name__}: {exc}"

            now = time.perf_counter()
            if now >= deadline:
                raise TimeoutError(
                    f"{PROVIDER} was not ready at {self.url} after "
                    f"{STARTUP_WAIT_SECONDS:.1f}s ({last_problem})"
                )
            if not announced:
                LOG.info(
                    "Waiting up to %.1fs for %s at %s before startup warm-up",
                    STARTUP_WAIT_SECONDS,
                    PROVIDER,
                    self.url,
                )
                announced = True
            await asyncio.sleep(min(STARTUP_POLL_SECONDS, deadline - now))
