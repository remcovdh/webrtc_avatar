"""Progressive WebRTC avatar server using Chatterbox TTS and FasterLivePortrait.

Each request is divided into short phrases. The first completed phrase starts
playing immediately while later phrases are synthesized and rendered.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import shutil
import tempfile
import time
import uuid
import wave
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

# Import torch before ONNX Runtime so ORT can reuse the CUDA 12 / cuDNN 9
# libraries shipped with the PyTorch image.
import torch
import onnxruntime as ort
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    AudioStreamTrack,
    VideoStreamTrack,
)
from av import AudioFrame, AudioResampler, VideoFrame
from aiortc.mediastreams import MediaStreamError
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from omegaconf import OmegaConf
import src.pipelines.faster_live_portrait_pipeline as flp_pipeline
from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline
from src.utils.utils import get_rotation_matrix

from avatar import config as avatar_config
from avatar.conductor_client import ConductorClient
from avatar.phrases import split_phrases
from avatar.renderer import Renderer, load_avatar
from avatar.playback import AUDIO_RATE, AUDIO_SAMPLES, VIDEO_FPS, IdleFrame, PlaybackBuffer
from avatar.speech import Speaker, send_event
from avatar.tracks import AvatarAudioTrack, AvatarVideoTrack
from avatar.tts_client import PROVIDER as TTS_PROVIDER
from avatar.tts_client import (
    VOICE_MODE_DESIGN,
    VOICE_MODE_PRESET_CLONE,
    VOICE_MODES,
    TtsClient,
    VoiceRequest,
)
from avatar.listener_client import SocketListener
from shared.listener_protocol import SAMPLE_RATE as LISTENER_SAMPLE_RATE
from shared.listener_protocol import TranscriptEvent
from src.pipelines.joyvasa_audio_to_motion_pipeline import (
    JoyVASAAudio2MotionPipeline,
)

LOG = logging.getLogger("avatar")
SERVER_BUILD = "neural-avatar-v2y-structure"

# All settings come from avatar_config (defaults, profiles, validation). The
# module constants below are that one loaded configuration under the names the
# rest of this file uses.
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

ROOT = Path(__file__).resolve().parent
AVATAR_PATH = Path(SETTINGS.image_path)
CONFIG_PATH = Path(SETTINGS.flp_config_path)
RESULTS_ROOT = Path(SETTINGS.results_root)
TTS = TtsClient(SETTINGS)
RENDERER = Renderer(SETTINGS)
TTS_DEFAULT_VOICE_MODE = SETTINGS.default_voice_mode
# Guidance scale of JoyVASA's audio-to-motion diffusion.
JOYVASA_CFG_SCALE = 2.8
MAX_TEXT_LENGTH = 500
INCREMENTAL_FRAME_WINDOWS = SETTINGS.incremental_frame_windows
RENDER_WINDOW_FRAMES = SETTINGS.render_window_frames
RENDER_STRIDE = SETTINGS.render_stride
ADAPTIVE_RENDER_STRIDE = SETTINGS.adaptive_render_stride
CATCHUP_RENDER_STRIDE = SETTINGS.catchup_render_stride
CATCHUP_BUFFER_SECONDS = SETTINGS.catchup_buffer_seconds
# FLP can render slower than real time. Rather than letting the mouth fall
# behind the voice (or skipping frames to catch up), hold each phrase's speech
# until enough video is rendered that the rest arrives in time.
SPEECH_START_GATE = SETTINGS.speech_start_gate
# Initial rendered-frames-per-second estimate; replaced by measurements.
EXPECTED_RENDER_FPS = SETTINGS.expected_render_fps
SPEECH_START_SAFETY = SETTINGS.speech_start_safety
# Render FLP's warping_spade (the per-frame bottleneck) through ONNX Runtime's
# TensorRT provider in FP16: ~27 ms instead of ~97 ms per frame on the RTX
# 5080. Engines are cached, so only the first start spends ~30 s building.
AVATAR_TENSORRT = SETTINGS.tensorrt
TENSORRT_CACHE_DIR = SETTINGS.tensorrt_cache
# Unix socket of listener/worker.py (VAD + streaming ASR + diarization). Empty
# disables listening; the avatar then works as before.
LISTENER_SOCKET = SETTINGS.listener_socket.strip()
# Unix socket of conductor/app.py, which owns the conversation (turn-taking,
# replies, logging). Empty disables it; typed text still works.
CONDUCTOR_SOCKET = SETTINGS.conductor_socket.strip()
# When set, every rendered phrase saves its WAV and JoyVASA motion here so
# motion problems can be analysed offline.
DEBUG_DUMP_DIR = SETTINGS.debug_dump_dir.strip()
# Spoken once at startup so the first user request finds every model loaded;
# its first frame also becomes the idle image.
WARMUP_TEXT = "Hello."
USE_NEURAL_IDLE_FRAME = SETTINGS.use_neural_idle_frame
WARMUP_IDLE_FRAME_INDEX = SETTINGS.warmup_idle_frame_index
TTS_PREFETCH = SETTINGS.tts_prefetch
TTS_PREFETCH_POLICY = SETTINGS.tts_prefetch_policy
TTS_PREFETCH_MIN_BUFFER_SECONDS = SETTINGS.tts_prefetch_min_buffer_seconds
PERSISTENT_PHRASE_MOTION = SETTINGS.persistent_phrase_motion
AVATAR_RELATIVE_MOTION = SETTINGS.relative_motion
AVATAR_ANIMATION_REGION = SETTINGS.animation_region
AVATAR_DRIVING_MULTIPLIER = SETTINGS.driving_multiplier
AVATAR_NORMALIZE_LIP = SETTINGS.normalize_lip
AVATAR_EYE_RETARGETING = SETTINGS.eye_retargeting
# Scales JoyVASA's eye keypoint motion relative to the utterance's first
# frame without touching mouth, brow or head motion: 1.0 keeps the generated
# motion, 0.0 keeps the portrait's own eyes (no gaze drift, but no blinks).
AVATAR_EYE_MOTION_SCALE = SETTINGS.eye_motion_scale
# Scales JoyVASA's mouth keypoint motion the same way. Above 1.0 opens the
# mouth further for the same audio without amplifying eyes or head motion.
AVATAR_LIP_MOTION_SCALE = SETTINGS.lip_motion_scale
# Scales JoyVASA's head rotation and translation around the utterance's first
# pose. Its pitch drifted 1-5 degrees upwards and roll up to 5 degrees during
# speech, so the avatar looked above the camera with a tilted frame.
AVATAR_HEAD_MOTION_SCALE = SETTINGS.head_motion_scale
# FLP rotates the source crop so the face is upright. With a slightly tilted
# face that turned the whole output frame, with black wedges where the crop
# left the photo. FLP never passes its own flag_do_rot to crop_image, so the
# server sets it (see _crop_source_image). false keeps the crop upright and
# the face keeps its own tilt.
AVATAR_CROP_ROTATION = SETTINGS.crop_rotation
# How JoyVASA's mouth keypoints reach the portrait. `relative` applies their
# change since the utterance's first frame to the portrait's own closed-smile
# lips, which pressed and sucked the lips in; `absolute` uses JoyVASA's mouth
# shapes directly while eyes and head stay relative. The lip motion scale only
# applies in relative mode.
AVATAR_LIP_MOTION_MODE = SETTINGS.lip_motion_mode
# Shifts the video against the audio clock: positive shows the mouth later.
# JoyVASA's mouth leads the loudness by ~40-160 ms (median ~80 ms).
AVATAR_LIP_SYNC_OFFSET_MS = SETTINGS.lip_sync_offset_ms
# FasterLivePortrait's own "eyes" and "lip" animation-region keypoints.
EYE_EXPRESSION_INDICES = [11, 13, 15, 16, 18]
LIP_EXPRESSION_INDICES = [6, 12, 14, 17, 19, 20]
AVATAR_LIP_RETARGETING = SETTINGS.lip_retargeting
MERGE_SHORT_OPENING_PHRASE = SETTINGS.merge_short_opening_phrase
PHRASE_MIN_FIRST_CHARS = SETTINGS.phrase_min_first_chars
PHRASE_FIRST_TARGET_CHARS = SETTINGS.phrase_first_target_chars
PHRASE_TARGET_CHARS = SETTINGS.phrase_target_chars
PHRASE_MAX_CHARS = SETTINGS.phrase_max_chars

pcs: set[RTCPeerConnection] = set()
startup_error: str | None = None















# The image shown while idle; every playback buffer reads it.
IDLE_FRAME = IdleFrame()
SPEAKER = Speaker(SETTINGS, RENDERER, TTS, IDLE_FRAME)












































@asynccontextmanager
async def lifespan(_app: FastAPI):
    global startup_error
    try:
        IDLE_FRAME.frame = load_avatar(Path(SETTINGS.image_path))
        LOG.info("Loading FasterLivePortrait and source portrait")
        await asyncio.to_thread(RENDERER.load)
        await SPEAKER.warm_up()
        LOG.info("Avatar pipeline is ready")
    except Exception as exc:
        startup_error = SPEAKER.unavailable = str(exc)
        LOG.exception("Avatar pipeline initialization failed")

    yield

    await asyncio.gather(*(pc.close() for pc in tuple(pcs)), return_exceptions=True)
    pcs.clear()


app = FastAPI(title="Streaming Avatar Test", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    tts_ready = await TTS.is_ready()
    warmup = SPEAKER.warmup

    providers = ort.get_available_providers()
    preset_configured, preset_problem = TTS.preset_configuration()
    body = {
        "server_build": SERVER_BUILD,
        **CONFIG.summary(),
        "ok": startup_error is None and tts_ready,
        "avatar_ready": startup_error is None,
        "tts_ready": tts_ready,
        "tts_provider": TTS_PROVIDER,
        "tts_url": TTS.url,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "onnxruntime_version": ort.__version__,
        "onnx_providers": providers,
        "cuda_provider": "CUDAExecutionProvider" in providers,
        "incremental_frame_windows": INCREMENTAL_FRAME_WINDOWS,
        "render_window_frames": RENDER_WINDOW_FRAMES,
        "adaptive_render_stride": ADAPTIVE_RENDER_STRIDE,
        "catchup_render_stride": CATCHUP_RENDER_STRIDE,
        "catchup_buffer_seconds": CATCHUP_BUFFER_SECONDS,
        "render_stride": RENDER_STRIDE,
        "startup_warmup_complete": warmup.complete,
        "startup_warmup_seconds": (
            round(warmup.seconds, 3) if warmup.seconds is not None else None
        ),
        "startup_warmup_metrics": warmup.metrics,
        "startup_warmup_error": warmup.error,
        "neural_idle_frame_enabled": USE_NEURAL_IDLE_FRAME,
        "idle_frame_source": IDLE_FRAME.source,
        "idle_frame_index": IDLE_FRAME.index,
        "tts_startup_wait_seconds": (
            round(warmup.tts_wait_seconds, 3)
            if warmup.tts_wait_seconds is not None
            else None
        ),
        "tts_prefetch": TTS_PREFETCH,
        "tts_prefetch_depth": 1 if TTS_PREFETCH else 0,
        "tts_prefetch_policy": TTS_PREFETCH_POLICY,
        "tts_prefetch_min_buffer_seconds": TTS_PREFETCH_MIN_BUFFER_SECONDS,
        "persistent_phrase_motion": PERSISTENT_PHRASE_MOTION,
        "relative_motion": AVATAR_RELATIVE_MOTION,
        "animation_region": AVATAR_ANIMATION_REGION,
        "driving_multiplier": AVATAR_DRIVING_MULTIPLIER,
        "normalize_lip": AVATAR_NORMALIZE_LIP,
        "eye_retargeting": AVATAR_EYE_RETARGETING,
        "eye_motion_scale": AVATAR_EYE_MOTION_SCALE,
        "lip_motion_scale": AVATAR_LIP_MOTION_SCALE,
        "lip_motion_mode": AVATAR_LIP_MOTION_MODE,
        "head_motion_scale": AVATAR_HEAD_MOTION_SCALE,
        "lip_sync_offset_ms": AVATAR_LIP_SYNC_OFFSET_MS,
        "audio_keepalive_dbfs": SETTINGS.audio_keepalive_dbfs,
        "listener_enabled": bool(LISTENER_SOCKET),
        "conductor_enabled": bool(CONDUCTOR_SOCKET),
        "crop_rotation": AVATAR_CROP_ROTATION,
        "warping_backend": RENDERER.warping_backend,
        "speech_start_gate": SPEECH_START_GATE,
        "speech_start_safety": SPEECH_START_SAFETY,
        "render_fps_estimate": round(RENDERER.fps_estimate, 3),
        "lip_retargeting": AVATAR_LIP_RETARGETING,
        "default_voice_mode": TTS_DEFAULT_VOICE_MODE,
        "voice_modes": sorted(VOICE_MODES),
        "preset_voice_configured": preset_configured,
        "preset_voice_status": preset_problem,
        "phrase_first_target_chars": PHRASE_FIRST_TARGET_CHARS,
        "merge_short_opening_phrase": MERGE_SHORT_OPENING_PHRASE,
        "phrase_min_first_chars": PHRASE_MIN_FIRST_CHARS,
        "phrase_target_chars": PHRASE_TARGET_CHARS,
        "phrase_max_chars": PHRASE_MAX_CHARS,
        "error": startup_error,
    }
    return JSONResponse(body, status_code=200 if body["ok"] else 503)


@app.get("/client-config")
async def client_config() -> JSONResponse:
    ice_servers = json.loads(SETTINGS.ice_servers_json)
    preset_configured, preset_problem = TTS.preset_configuration()
    default_mode = TTS_DEFAULT_VOICE_MODE
    if default_mode not in VOICE_MODES:
        default_mode = VOICE_MODE_DESIGN
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
            "iceServers": ice_servers,
            "voiceModes": voice_modes,
            "defaultVoiceMode": default_mode,
            "presetVoiceConfigured": preset_configured,
            "presetVoiceStatus": preset_problem,
            "presetVoiceFilename": (
                TTS.preset_audio_path.name if preset_configured else None
            ),
            "ttsProvider": TTS_PROVIDER,
        }
    )


@app.post("/offer")
async def offer(request: Request) -> JSONResponse:
    params = await request.json()
    if not isinstance(params.get("sdp"), str) or params.get("type") != "offer":
        raise HTTPException(400, "Expected an SDP offer")

    pc = RTCPeerConnection()
    pcs.add(pc)
    playback = PlaybackBuffer(
        IDLE_FRAME, SETTINGS.lip_sync_offset_ms, SETTINGS.speech_start_safety
    )
    jobs: set[asyncio.Task[Any]] = set()
    peer_channel: dict[str, Any] = {}
    listener = SocketListener(LISTENER_SOCKET) if LISTENER_SOCKET else None
    conductor = (
        ConductorClient(
            CONDUCTOR_SOCKET,
            avatar_config={"server_build": SERVER_BUILD, **CONFIG.summary()},
        )
        if CONDUCTOR_SOCKET
        else None
    )

    def run_job(coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        jobs.add(task)
        task.add_done_callback(jobs.discard)

    async def speak(text: str, voice_mode: str) -> None:
        """Every utterance of the avatar, so the conductor can ignore the
        avatar's own voice while it talks (half-duplex)."""
        if conductor is not None:
            await conductor.on_avatar_speaking(True)
        try:
            await SPEAKER.speak(
                text, voice_mode, playback, peer_channel.get("channel")
            )
        finally:
            if conductor is not None:
                await conductor.on_avatar_speaking(False)

    say_queue: asyncio.Queue[str] = asyncio.Queue()

    async def speak_queued() -> None:
        """Speak the conductor's lines in order, e.g. a filler then the answer."""
        while True:
            text = await say_queue.get()
            while playback.busy:  # typed text may still be playing
                await asyncio.sleep(0.05)
            await speak(text, TTS_DEFAULT_VOICE_MODE)

    class PeerOutput:
        """`AvatarOutput` for this browser session."""

        async def say(self, text: str) -> None:
            if text:
                await say_queue.put(text)

        async def show(self, state: dict[str, Any]) -> None:
            channel = peer_channel.get("channel")
            if channel is not None:
                await send_event(channel, "conversation", **state)

    if conductor is not None:
        run_job(speak_queued())
        await conductor.start(PeerOutput())

    async def forward_transcript(event: TranscriptEvent) -> None:
        channel = peer_channel.get("channel")
        if channel is None:
            return
        await send_event(
            channel,
            "transcript",
            kind=event.type,
            utterance=event.utterance,
            text=event.text,
            start=round(event.start, 2),
            end=round(event.end, 2),
            speaker=event.speaker,
            # Lets the page (and the M2 conductor) ignore the avatar's own
            # voice picked up by the microphone.
            avatar_speaking=playback.busy,
        )
        if conductor is not None:
            await conductor.on_transcript(asdict(event))

    async def listen(track: Any) -> None:
        """Browser microphone -> 16 kHz mono PCM -> listener."""
        assert listener is not None
        await listener.start(forward_transcript)
        resampler = AudioResampler(
            format="s16", layout="mono", rate=LISTENER_SAMPLE_RATE
        )
        try:
            while True:
                frame = await track.recv()
                for resampled in resampler.resample(frame):
                    await listener.feed(resampled.to_ndarray().tobytes())
        except MediaStreamError:
            pass
        finally:
            await listener.close()

    @pc.on("track")
    def on_track(track: Any) -> None:
        if track.kind != "audio" or listener is None:
            return
        LOG.info("Microphone track received; streaming it to the listener")
        task = asyncio.create_task(listen(track))
        jobs.add(task)
        task.add_done_callback(jobs.discard)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        LOG.info("Peer state: %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed"}:
            for job in tuple(jobs):
                job.cancel()
            if conductor is not None:
                await conductor.close()
            await pc.close()
            pcs.discard(pc)

    @pc.on("datachannel")
    def on_datachannel(channel: Any) -> None:
        LOG.info("Data channel connected: %s", channel.label)
        peer_channel["channel"] = channel

        ready_announced = False

        def announce_ready() -> None:
            """Send ready even when aiortc reports the channel already open."""
            nonlocal ready_announced
            if ready_announced or getattr(channel, "readyState", None) != "open":
                return
            ready_announced = True
            task = asyncio.create_task(send_event(channel, "ready", message="Ready"))
            jobs.add(task)
            task.add_done_callback(jobs.discard)

        @channel.on("open")
        def on_open() -> None:
            announce_ready()

        # A remotely-created RTCDataChannel can already be open when the
        # datachannel callback runs, in which case its open event is not seen by
        # this newly attached handler. Check the state on the next loop turn.
        asyncio.get_running_loop().call_soon(announce_ready)

        @channel.on("message")
        def on_message(message: Any) -> None:
            try:
                payload = json.loads(message) if isinstance(message, str) else {}
            except json.JSONDecodeError:
                payload = {"text": str(message)}

            if payload.get("type") == "push_to_talk":
                if conductor is not None:
                    run_job(conductor.on_push_to_talk(bool(payload.get("pressed"))))
                return

            if payload.get("type") == "correction":
                if conductor is not None:
                    run_job(conductor.on_correction({
                        "turn": payload.get("turn"),
                        "text": str(payload.get("text", "")),
                        "fields": payload.get("fields") or {},
                    }))
                return

            if payload.get("type") == "listen_language":
                if conductor is not None:
                    run_job(conductor.on_language(str(payload.get("language", ""))))
                if listener is not None:
                    task = asyncio.create_task(
                        listener.set_language(str(payload.get("language", "")))
                    )
                    jobs.add(task)
                    task.add_done_callback(jobs.discard)
                return

            text = str(payload.get("text", "")).strip()
            voice_mode = str(
                payload.get("voice_mode", TTS_DEFAULT_VOICE_MODE)
            ).strip()
            if not text:
                return
            if len(text) > MAX_TEXT_LENGTH:
                task = asyncio.create_task(
                    send_event(
                        channel,
                        "error",
                        message=f"Text is limited to {MAX_TEXT_LENGTH} characters.",
                    )
                )
            else:
                task = asyncio.create_task(speak(text, voice_mode))
            jobs.add(task)
            task.add_done_callback(jobs.discard)

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    )
    # aiortc gives every sender its own random msid stream, and browsers only
    # lip-sync (via RTCP sender reports) tracks that share one stream.
    stream_id = str(uuid.uuid4())
    for track in (
        AvatarVideoTrack(playback),
        AvatarAudioTrack(playback, SETTINGS.audio_keepalive_dbfs),
    ):
        pc.addTrack(track)._stream_id = stream_id
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return JSONResponse(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )
