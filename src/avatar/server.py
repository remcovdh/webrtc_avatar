"""The avatar's web server: the page, health, and the WebRTC offer.

This module only wires the parts together:

    config      settings, profiles, validation
    tts_client  speech from the TTS service
    renderer    face frames from speech (JoyVASA + FasterLivePortrait)
    speech      one text -> phrases -> speech -> frames -> playback
    playback    per-browser audio/video queue on one clock
    session     one browser connection (WebRTC, microphone, messages)

Start it with `python -m avatar.serve`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Import torch before ONNX Runtime so ORT can reuse the CUDA 12 / cuDNN 9
# libraries shipped with the PyTorch image.
import torch
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from avatar import config as avatar_config
from avatar.playback import IdleFrame
from avatar.renderer import Renderer, load_avatar
from avatar.session import Session
from avatar.speech import Speaker
from avatar.tts_client import (
    PROVIDER as TTS_PROVIDER,
    VOICE_MODE_DESIGN,
    VOICE_MODE_PRESET_CLONE,
    VOICE_MODES,
    TtsClient,
)

LOG = logging.getLogger("avatar")
SERVER_BUILD = "neural-avatar-v2y-structure"
ROOT = Path(__file__).resolve().parent

# The one loaded configuration (defaults, profile, overrides; see config.py).
CONFIG = avatar_config.load()
SETTINGS = CONFIG.settings
logging.basicConfig(
    level=SETTINGS.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG.info(
    "Configuration: profile=%s hash=%s overrides=%s",
    CONFIG.profile,
    CONFIG.config_hash,
    CONFIG.summary()["config_overrides"] or "none",
)
if CONFIG.experiment:
    LOG.warning("EXPERIMENT RUN: a diagnosis or rollback switch is overridden")
for _warning in CONFIG.warnings:
    LOG.warning("Configuration: %s", _warning)


class Runtime:
    """The parts of the running avatar and what is known about their state."""

    def __init__(self) -> None:
        self.tts = TtsClient(SETTINGS)
        self.renderer = Renderer(SETTINGS)
        # The image shown while idle; every playback buffer reads it.
        self.idle = IdleFrame()
        self.speaker = Speaker(SETTINGS, self.renderer, self.tts, self.idle)
        # Why the models could not be loaded, if they could not.
        self.startup_error: str | None = None
        self.sessions: set[Session] = set()

    async def start(self) -> None:
        """Load the models and warm them up; a failure is kept, not raised, so
        the server still starts and /health can explain what is wrong."""
        try:
            self.idle.frame = load_avatar(Path(SETTINGS.image_path))
            LOG.info("Loading FasterLivePortrait and source portrait")
            await asyncio.to_thread(self.renderer.load)
            await self.speaker.warm_up()
            LOG.info("Avatar pipeline is ready")
        except Exception as exc:
            self.startup_error = self.speaker.unavailable = str(exc)
            LOG.exception("Avatar pipeline initialization failed")

    async def stop(self) -> None:
        await asyncio.gather(
            *(session.close() for session in tuple(self.sessions)),
            return_exceptions=True,
        )
        self.sessions.clear()


RUNTIME = Runtime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await RUNTIME.start()
    yield
    await RUNTIME.stop()


app = FastAPI(title="Streaming Avatar Test", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


# /health key -> settings field, for the settings reported as they are.
_HEALTH_SETTINGS = {
    "incremental_frame_windows": "incremental_frame_windows",
    "render_window_frames": "render_window_frames",
    "render_stride": "render_stride",
    "adaptive_render_stride": "adaptive_render_stride",
    "catchup_render_stride": "catchup_render_stride",
    "catchup_buffer_seconds": "catchup_buffer_seconds",
    "neural_idle_frame_enabled": "use_neural_idle_frame",
    "tts_prefetch": "tts_prefetch",
    "tts_prefetch_policy": "tts_prefetch_policy",
    "tts_prefetch_min_buffer_seconds": "tts_prefetch_min_buffer_seconds",
    "persistent_phrase_motion": "persistent_phrase_motion",
    "relative_motion": "relative_motion",
    "animation_region": "animation_region",
    "driving_multiplier": "driving_multiplier",
    "normalize_lip": "normalize_lip",
    "eye_retargeting": "eye_retargeting",
    "lip_retargeting": "lip_retargeting",
    "eye_motion_scale": "eye_motion_scale",
    "lip_motion_scale": "lip_motion_scale",
    "lip_motion_mode": "lip_motion_mode",
    "head_motion_scale": "head_motion_scale",
    "lip_sync_offset_ms": "lip_sync_offset_ms",
    "audio_keepalive_dbfs": "audio_keepalive_dbfs",
    "crop_rotation": "crop_rotation",
    "speech_start_gate": "speech_start_gate",
    "speech_start_safety": "speech_start_safety",
    "default_voice_mode": "default_voice_mode",
    "phrase_first_target_chars": "phrase_first_target_chars",
    "merge_short_opening_phrase": "merge_short_opening_phrase",
    "phrase_min_first_chars": "phrase_min_first_chars",
    "phrase_target_chars": "phrase_target_chars",
    "phrase_max_chars": "phrase_max_chars",
}


@app.get("/health")
async def health() -> JSONResponse:
    runtime = RUNTIME
    tts_ready = await runtime.tts.is_ready()
    warmup = runtime.speaker.warmup
    providers = ort.get_available_providers()
    preset_configured, preset_problem = runtime.tts.preset_configuration()

    def rounded(value: float | None) -> float | None:
        return round(value, 3) if value is not None else None

    body: dict[str, Any] = {
        "server_build": SERVER_BUILD,
        **CONFIG.summary(),
        "ok": runtime.startup_error is None and tts_ready,
        "avatar_ready": runtime.startup_error is None,
        "error": runtime.startup_error,
        # Environment
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "onnxruntime_version": ort.__version__,
        "onnx_providers": providers,
        "cuda_provider": "CUDAExecutionProvider" in providers,
        # TTS
        "tts_ready": tts_ready,
        "tts_provider": TTS_PROVIDER,
        "tts_url": runtime.tts.url,
        "tts_prefetch_depth": 1 if SETTINGS.tts_prefetch else 0,
        "voice_modes": sorted(VOICE_MODES),
        "preset_voice_configured": preset_configured,
        "preset_voice_status": preset_problem,
        # Start-up
        "startup_warmup_complete": warmup.complete,
        "startup_warmup_seconds": rounded(warmup.seconds),
        "startup_warmup_metrics": warmup.metrics,
        "startup_warmup_error": warmup.error,
        "tts_startup_wait_seconds": rounded(warmup.tts_wait_seconds),
        "idle_frame_source": runtime.idle.source,
        "idle_frame_index": runtime.idle.index,
        # Rendering
        "warping_backend": runtime.renderer.warping_backend,
        "render_fps_estimate": round(runtime.renderer.fps_estimate, 3),
        # Listening
        "listener_enabled": bool(SETTINGS.listener_socket.strip()),
        "conductor_enabled": bool(SETTINGS.conductor_socket.strip()),
    }
    for key, name in _HEALTH_SETTINGS.items():
        body[key] = getattr(SETTINGS, name)
    return JSONResponse(body, status_code=200 if body["ok"] else 503)


@app.get("/client-config")
async def client_config() -> JSONResponse:
    """What the page needs before it connects: ICE servers and voice modes."""
    tts = RUNTIME.tts
    preset_configured, preset_problem = tts.preset_configuration()
    default_mode = SETTINGS.default_voice_mode
    if default_mode != VOICE_MODE_DESIGN and not preset_configured:
        default_mode = VOICE_MODE_DESIGN

    voice_modes = [
        {
            "id": VOICE_MODE_DESIGN,
            "label": "Chatterbox built-in voice",
            "description": "Use Chatterbox's built-in voice.",
            "available": True,
        },
        {
            "id": VOICE_MODE_PRESET_CLONE,
            "label": "Preset voice (clone)",
            "description": "Clone the voice of the reference recording.",
            "available": preset_configured,
        },
    ]
    return JSONResponse(
        {
            "iceServers": json.loads(SETTINGS.ice_servers_json),
            "voiceModes": voice_modes,
            "defaultVoiceMode": default_mode,
            "presetVoiceConfigured": preset_configured,
            "presetVoiceStatus": preset_problem,
            "presetVoiceFilename": (
                tts.preset_audio_path.name if preset_configured else None
            ),
            "ttsProvider": TTS_PROVIDER,
        }
    )


@app.post("/offer")
async def offer(request: Request) -> JSONResponse:
    """A browser connects: create its session and answer its SDP offer."""
    params = await request.json()
    if not isinstance(params.get("sdp"), str) or params.get("type") != "offer":
        raise HTTPException(400, "Expected an SDP offer")

    session = Session(
        SETTINGS,
        RUNTIME.speaker,
        avatar_config={"server_build": SERVER_BUILD, **CONFIG.summary()},
        on_closed=RUNTIME.sessions.discard,
    )
    RUNTIME.sessions.add(session)
    description = await session.answer(params["sdp"], params["type"])
    return JSONResponse({"sdp": description.sdp, "type": description.type})
