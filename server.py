"""Progressive WebRTC avatar server using Breeze TTS 2 and FasterLivePortrait.

Each request is divided into short phrases. The first completed phrase starts
playing immediately while later phrases are synthesized and rendered.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import tempfile
import time
import wave
from collections import deque
from contextlib import asynccontextmanager
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
from av import AudioFrame, VideoFrame
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from omegaconf import OmegaConf
from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline

LOG = logging.getLogger("avatar")
SERVER_BUILD = "neural-avatar-v2a-warmup-metrics"
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ROOT = Path(__file__).resolve().parent
AVATAR_PATH = Path(os.getenv("AVATAR_PATH", "/workspace/inputs/avatar.jpg"))
CONFIG_PATH = Path(
    os.getenv(
        "FLP_CONFIG_PATH",
        "/workspace/FasterLivePortrait/configs/onnx_infer.yaml",
    )
)
RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "/workspace/results"))
TTS_URL = os.getenv("BREEZE_TTS_URL", "http://127.0.0.1:7860").rstrip("/")
TTS_INSTRUCTION = os.getenv(
    "BREEZE_VOICE_INSTRUCTION",
    "A warm, clear, natural voice with a calm conversational delivery.",
)
TTS_CFG_SCALE = float(os.getenv("BREEZE_CFG_SCALE", "4"))
TTS_SEED = int(os.getenv("BREEZE_SEED", "42"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "500"))
AVATAR_PASTE_BACK = _env_bool("AVATAR_PASTE_BACK", False)
STARTUP_WARMUP = _env_bool("STARTUP_WARMUP", True)
WARMUP_TEXT = os.getenv("WARMUP_TEXT", "Hello.").strip() or "Hello."
PROGRESSIVE_PHRASE_MODE = _env_bool("PROGRESSIVE_PHRASE_MODE", True)
PHRASE_FIRST_TARGET_CHARS = max(
    8, int(os.getenv("PHRASE_FIRST_TARGET_CHARS", "48"))
)
PHRASE_TARGET_CHARS = max(16, int(os.getenv("PHRASE_TARGET_CHARS", "100")))
PHRASE_MAX_CHARS = max(
    PHRASE_TARGET_CHARS, int(os.getenv("PHRASE_MAX_CHARS", "160"))
)
VIDEO_FPS = 30
AUDIO_RATE = 48_000
AUDIO_SAMPLES = 960  # 20 ms at 48 kHz

pcs: set[RTCPeerConnection] = set()
pipeline: GradioLivePortraitPipeline | None = None
startup_error: str | None = None
inference_lock = asyncio.Lock()
warmup_complete = False
warmup_seconds: float | None = None
warmup_metrics: dict[str, Any] = {}
warmup_error: str | None = None

# JoyVASA's official motion checkpoint stores its configuration as an
# argparse.Namespace. PyTorch 2.6+ blocks that class by default when loading
# weights. Allowlist only this known, inert configuration container while
# retaining weights_only=True for every other checkpoint global.
torch.serialization.add_safe_globals([argparse.Namespace])


def _letterbox(image: np.ndarray, size: int = 512) -> np.ndarray:
    """Fit an image into a square without stretching it."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _load_avatar() -> np.ndarray:
    if not AVATAR_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {AVATAR_PATH}. Mount ./inputs and add inputs/avatar.jpg."
        )
    image = cv2.imread(str(AVATAR_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode {AVATAR_PATH}")
    return _letterbox(image)


BASE_AVATAR = np.zeros((512, 512, 3), dtype=np.uint8)


def _initialize_pipeline() -> GradioLivePortraitPipeline:
    """Load FasterLivePortrait and precompute the source portrait."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"FasterLivePortrait config missing: {CONFIG_PATH}")

    LOG.info(
        "Runtime: torch=%s torch_cuda=%s onnxruntime=%s providers=%s",
        torch.__version__,
        torch.version.cuda,
        ort.__version__,
        ort.get_available_providers(),
    )

    cfg = OmegaConf.load(CONFIG_PATH)
    # Paste-back invokes torchgeometry's GPU matrix inverse for every frame.
    # On memory-constrained GPUs shared with TTS, cuSOLVER handle creation can
    # fail even though motion generation succeeded. The crop output already
    # contains the complete animated face and is the better WebRTC default.
    cfg.infer_params.flag_pasteback = AVATAR_PASTE_BACK
    cfg.infer_params.flag_relative_motion = False
    cfg.infer_params.flag_stitching = True
    cfg.infer_params.animation_region = "all"
    cfg.infer_params.cfg_scale = float(os.getenv("JOYVASA_CFG_SCALE", "2.8"))

    loaded = GradioLivePortraitPipeline(cfg=cfg, is_animal=False)
    if not loaded.prepare_source(str(AVATAR_PATH), realtime=False):
        raise ValueError(
            "No face was detected in avatar.jpg. Use a clear, front-facing portrait."
        )
    return loaded


def _split_phrases(text: str) -> list[str]:
    """Split text into low-latency phrases without losing punctuation."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if not PROGRESSIVE_PHRASE_MODE:
        return [normalized]

    phrases: list[str] = []
    current: list[str] = []
    current_length = 0
    closing_marks = "\"'”’)]}"

    for word in normalized.split(" "):
        current.append(word)
        current_length += len(word) + (1 if len(current) > 1 else 0)
        ending = word.rstrip(closing_marks)
        terminal = ending.endswith((".", "!", "?", ";", ":"))
        soft_break = ending.endswith(",")
        target = PHRASE_FIRST_TARGET_CHARS if not phrases else PHRASE_TARGET_CHARS

        if (
            terminal
            or current_length >= PHRASE_MAX_CHARS
            or current_length >= target
            or (soft_break and current_length >= max(12, target // 2))
        ):
            phrases.append(" ".join(current))
            current = []
            current_length = 0

    if current:
        tail = " ".join(current)
        if (
            phrases
            and len(tail) < 12
            and len(phrases[-1]) + 1 + len(tail) <= PHRASE_MAX_CHARS
        ):
            phrases[-1] = f"{phrases[-1]} {tail}"
        else:
            phrases.append(tail)

    return phrases


class PlaybackBuffer:
    """Per-peer playback state consumed by the two WebRTC tracks."""

    def __init__(self) -> None:
        self.video: deque[np.ndarray] = deque()
        self.audio: deque[np.ndarray] = deque()
        self.producing = False
        self.started = False
        self.last_video = BASE_AVATAR
        self.audio_underruns = 0
        self.video_underruns = 0

    @property
    def busy(self) -> bool:
        return self.producing or bool(self.video or self.audio)

    @property
    def buffered_seconds(self) -> float:
        video_seconds = len(self.video) / VIDEO_FPS
        audio_seconds = len(self.audio) * AUDIO_SAMPLES / AUDIO_RATE
        return min(video_seconds, audio_seconds)

    def begin(self) -> None:
        self.clear()
        self.producing = True

    def append(self, frames: list[np.ndarray], fps: float, pcm_24k: np.ndarray) -> float:
        if not frames:
            raise ValueError("The animation renderer returned no video frames")
        if fps <= 0:
            fps = 25.0

        pcm_48k = np.repeat(pcm_24k.astype(np.int16, copy=False), 2)
        audio_duration = len(pcm_48k) / AUDIO_RATE
        video_duration = len(frames) / fps
        duration = max(audio_duration, video_duration)

        target_count = max(1, math.ceil(duration * VIDEO_FPS))
        video_indices = np.minimum(
            (np.arange(target_count) * fps / VIDEO_FPS).astype(int),
            len(frames) - 1,
        )
        self.video.extend(frames[index] for index in video_indices)

        required_samples = max(len(pcm_48k), math.ceil(duration * AUDIO_RATE))
        padded = np.pad(pcm_48k, (0, required_samples - len(pcm_48k)))
        for offset in range(0, len(padded), AUDIO_SAMPLES):
            chunk = padded[offset : offset + AUDIO_SAMPLES]
            if len(chunk) < AUDIO_SAMPLES:
                chunk = np.pad(chunk, (0, AUDIO_SAMPLES - len(chunk)))
            self.audio.append(chunk.reshape(1, -1))
        self.started = True
        return duration

    def finish(self) -> None:
        self.producing = False

    def next_video(self) -> np.ndarray:
        if self.video:
            self.last_video = self.video.popleft()
            return self.last_video
        if self.started and (self.producing or self.audio):
            if self.producing:
                self.video_underruns += 1
            return self.last_video
        return BASE_AVATAR

    def next_audio(self) -> np.ndarray:
        if self.audio:
            return self.audio.popleft()
        if self.started and self.producing:
            self.audio_underruns += 1
        return np.zeros((1, AUDIO_SAMPLES), dtype=np.int16)

    def clear(self) -> None:
        self.video.clear()
        self.audio.clear()
        self.producing = False
        self.started = False
        self.last_video = BASE_AVATAR
        self.audio_underruns = 0
        self.video_underruns = 0


class AvatarVideoTrack(VideoStreamTrack):
    def __init__(self, playback: PlaybackBuffer) -> None:
        super().__init__()
        self.playback = playback

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(self.playback.next_video(), format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


class AvatarAudioTrack(AudioStreamTrack):
    def __init__(self, playback: PlaybackBuffer) -> None:
        super().__init__()
        self.playback = playback
        self._start: float | None = None
        self._timestamp = 0

    async def recv(self) -> AudioFrame:
        if self._start is None:
            self._start = time.time()
        else:
            self._timestamp += AUDIO_SAMPLES
            target = self._start + self._timestamp / AUDIO_RATE
            await asyncio.sleep(max(0, target - time.time()))

        frame = AudioFrame.from_ndarray(
            self.playback.next_audio(), format="s16", layout="mono"
        )
        frame.sample_rate = AUDIO_RATE
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, AUDIO_RATE)
        return frame


async def _send_event(channel: Any, event_type: str, **payload: Any) -> None:
    if getattr(channel, "readyState", None) == "open":
        channel.send(json.dumps({"type": event_type, **payload}))


async def _synthesize(
    text: str,
    instruction: str,
    pcm_path: Path,
    wav_path: Path,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Call Breeze and report request, first-byte, transfer and finalize time."""
    form = {
        "text": (None, text),
        "instruction": (None, instruction),
        "cfg_scale": (None, str(TTS_CFG_SCALE)),
        "seed": (None, str(TTS_SEED)),
    }
    timeout = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=15.0)
    request_started = time.perf_counter()
    headers_ready = request_started
    first_byte_at: float | None = None
    body_complete = request_started
    bytes_received = 0

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", f"{TTS_URL}/v1/audio/speech", files=form
        ) as response:
            headers_ready = time.perf_counter()
            if response.is_error:
                detail = (await response.aread()).decode("utf-8", errors="replace")[:300]
                raise RuntimeError(
                    f"Breeze TTS failed ({response.status_code}): {detail}"
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
        raise RuntimeError("Breeze returned an empty audio stream")
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
    }


def _render_animation(
    wav_path: Path,
    output_dir: Path,
) -> tuple[list[np.ndarray], float, dict[str, float | int]]:
    if pipeline is None:
        raise RuntimeError(startup_error or "FasterLivePortrait is not ready")

    render_started = time.perf_counter()
    original_path, crop_path, reported_elapsed = pipeline.run_audio_driving(
        str(wav_path), str(AVATAR_PATH), save_dir=str(output_dir)
    )
    pipeline_seconds = time.perf_counter() - render_started
    video_path = original_path if AVATAR_PASTE_BACK else crop_path
    LOG.info(
        "Using %s animation output",
        "full-frame paste-back" if AVATAR_PASTE_BACK else "face crop",
    )
    decode_started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: list[np.ndarray] = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(_letterbox(frame))
    capture.release()
    decode_seconds = time.perf_counter() - decode_started
    return frames, fps, {
        "pipeline_ms": round(pipeline_seconds * 1000),
        "decode_ms": round(decode_seconds * 1000),
        "total_ms": round((time.perf_counter() - render_started) * 1000),
        "pipeline_reported_ms": round(float(reported_elapsed or 0) * 1000),
        "frames": len(frames),
        "source_fps": round(fps, 3),
    }


async def _create_clip(
    text: str,
    instruction: str,
    playback: PlaybackBuffer,
    channel: Any,
) -> None:
    if startup_error:
        await _send_event(channel, "error", message=startup_error)
        return
    if playback.busy:
        await _send_event(channel, "error", message="Wait for the current speech to finish.")
        return

    phrases = _split_phrases(text)
    if not phrases:
        await _send_event(channel, "error", message="Enter text to speak.")
        return

    playback.begin()
    request_started = time.perf_counter()
    total_tts = 0.0
    total_render = 0.0

    await _send_event(
        channel,
        "plan",
        message=f"Prepared {len(phrases)} progressive phrase{'s' if len(phrases) != 1 else ''}",
        phrases=phrases,
    )

    if inference_lock.locked():
        await _send_event(channel, "status", phase="queued", message="Queued behind another request…")

    try:
        async with inference_lock:
            RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="avatar-", dir=RESULTS_ROOT) as tmp:
                tmp_dir = Path(tmp)
                for index, phrase in enumerate(phrases, start=1):
                    chunk_dir = tmp_dir / f"chunk-{index:03d}"
                    chunk_dir.mkdir()
                    pcm_path = chunk_dir / "speech.pcm"
                    wav_path = chunk_dir / "speech.wav"
                    underruns_before = playback.audio_underruns
                    video_underruns_before = playback.video_underruns

                    await _send_event(
                        channel,
                        "status",
                        phase="tts",
                        chunk=index,
                        chunks=len(phrases),
                        message=f"Phrase {index}/{len(phrases)}: generating speech…",
                    )
                    started = time.perf_counter()
                    pcm, tts_detail = await _synthesize(
                        phrase,
                        instruction or TTS_INSTRUCTION,
                        pcm_path,
                        wav_path,
                    )
                    tts_seconds = time.perf_counter() - started
                    total_tts += tts_seconds

                    await _send_event(
                        channel,
                        "status",
                        phase="animation",
                        chunk=index,
                        chunks=len(phrases),
                        message=f"Phrase {index}/{len(phrases)}: rendering facial motion…",
                    )
                    started = time.perf_counter()
                    frames, fps, render_detail = await asyncio.to_thread(
                        _render_animation, wav_path, chunk_dir
                    )
                    render_seconds = time.perf_counter() - started
                    total_render += render_seconds
                    media_duration = playback.append(frames, fps, pcm)
                    first_ready_ms = (
                        round((time.perf_counter() - request_started) * 1000)
                        if index == 1
                        else None
                    )
                    underrun_ms = (
                        playback.audio_underruns - underruns_before
                    ) * AUDIO_SAMPLES / AUDIO_RATE * 1000
                    video_underrun_ms = (
                        playback.video_underruns - video_underruns_before
                    ) / VIDEO_FPS * 1000

                    LOG.info(
                        "Phrase %d/%d timings: tts=%.3fs first_byte=%dms download=%dms "
                        "render=%.3fs pipeline=%dms decode=%dms frames=%d media=%.3fs "
                        "buffer=%.3fs underrun=%.0fms",
                        index,
                        len(phrases),
                        tts_seconds,
                        tts_detail["first_byte_ms"],
                        tts_detail["download_ms"],
                        render_seconds,
                        render_detail["pipeline_ms"],
                        render_detail["decode_ms"],
                        render_detail["frames"],
                        media_duration,
                        playback.buffered_seconds,
                        underrun_ms,
                    )
                    await _send_event(
                        channel,
                        "metrics",
                        chunk=index,
                        chunks=len(phrases),
                        phrase=phrase,
                        tts_ms=round(tts_seconds * 1000),
                        tts_headers_ms=tts_detail["headers_ms"],
                        tts_first_byte_ms=tts_detail["first_byte_ms"],
                        tts_download_ms=tts_detail["download_ms"],
                        tts_finalize_ms=tts_detail["finalize_ms"],
                        tts_bytes=tts_detail["bytes"],
                        render_ms=round(render_seconds * 1000),
                        render_pipeline_ms=render_detail["pipeline_ms"],
                        render_decode_ms=render_detail["decode_ms"],
                        render_pipeline_reported_ms=render_detail[
                            "pipeline_reported_ms"
                        ],
                        render_frames=render_detail["frames"],
                        render_source_fps=render_detail["source_fps"],
                        media_seconds=round(media_duration, 3),
                        buffered_seconds=round(playback.buffered_seconds, 3),
                        underrun_ms=round(underrun_ms),
                        video_underrun_ms=round(video_underrun_ms),
                        first_ready_ms=first_ready_ms,
                    )

                    if index == 1:
                        await _send_event(
                            channel,
                            "playing",
                            message=f"Playing phrase 1/{len(phrases)} while preparing the rest…",
                            duration=round(media_duration, 3),
                            first_ready_ms=first_ready_ms,
                        )

        playback.finish()
        generation_seconds = time.perf_counter() - request_started
        await _send_event(
            channel,
            "status",
            phase="draining",
            message="All phrases generated; finishing playback…",
        )
        while playback.busy:
            await asyncio.sleep(0.05)

        total_seconds = time.perf_counter() - request_started
        await _send_event(
            channel,
            "summary",
            message="Progressive playback complete",
            chunks=len(phrases),
            tts_ms=round(total_tts * 1000),
            render_ms=round(total_render * 1000),
            generation_ms=round(generation_seconds * 1000),
            total_ms=round(total_seconds * 1000),
            underrun_ms=round(
                playback.audio_underruns * AUDIO_SAMPLES / AUDIO_RATE * 1000
            ),
        )
        await _send_event(channel, "ready", message="Ready")
    except asyncio.CancelledError:
        playback.clear()
        raise
    except Exception as exc:
        playback.clear()
        LOG.exception("Avatar generation failed")
        await _send_event(channel, "error", message=str(exc))


async def _warmup_pipeline() -> None:
    """Exercise the same TTS, JoyVASA and renderer path before the first user."""
    global warmup_complete, warmup_seconds, warmup_metrics, warmup_error

    started = time.perf_counter()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG.info("Startup warm-up begins with %r", WARMUP_TEXT)
    try:
        with tempfile.TemporaryDirectory(prefix="warmup-", dir=RESULTS_ROOT) as tmp:
            warmup_dir = Path(tmp)
            pcm_path = warmup_dir / "warmup.pcm"
            wav_path = warmup_dir / "warmup.wav"
            pcm, tts_detail = await _synthesize(
                WARMUP_TEXT,
                TTS_INSTRUCTION,
                pcm_path,
                wav_path,
            )
            frames, fps, render_detail = await asyncio.to_thread(
                _render_animation,
                wav_path,
                warmup_dir,
            )
            warmup_seconds = time.perf_counter() - started
            warmup_metrics = {
                "tts": tts_detail,
                "render": render_detail,
                "media_seconds": round(
                    max(len(pcm) / 24_000, len(frames) / max(fps, 1.0)),
                    3,
                ),
            }
            warmup_complete = True
            LOG.info(
                "Startup warm-up complete: total=%.3fs tts=%dms pipeline=%dms "
                "decode=%dms frames=%d",
                warmup_seconds,
                tts_detail["total_ms"],
                render_detail["pipeline_ms"],
                render_detail["decode_ms"],
                render_detail["frames"],
            )
    except Exception as exc:
        warmup_seconds = time.perf_counter() - started
        warmup_error = str(exc)
        LOG.exception(
            "Startup warm-up failed after %.3fs; continuing without warm-up",
            warmup_seconds,
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global BASE_AVATAR, pipeline, startup_error
    try:
        BASE_AVATAR = _load_avatar()
        LOG.info("Loading FasterLivePortrait and source portrait")
        pipeline = await asyncio.to_thread(_initialize_pipeline)
        if STARTUP_WARMUP:
            await _warmup_pipeline()
        LOG.info("Avatar pipeline is ready")
    except Exception as exc:
        startup_error = str(exc)
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
    tts_ready = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{TTS_URL}/docs")
            tts_ready = response.status_code < 500
    except httpx.HTTPError:
        pass

    providers = ort.get_available_providers()
    body = {
        "server_build": SERVER_BUILD,
        "ok": startup_error is None and tts_ready,
        "avatar_ready": startup_error is None,
        "tts_ready": tts_ready,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "onnxruntime_version": ort.__version__,
        "onnx_providers": providers,
        "cuda_provider": "CUDAExecutionProvider" in providers,
        "paste_back": AVATAR_PASTE_BACK,
        "startup_warmup_enabled": STARTUP_WARMUP,
        "startup_warmup_complete": warmup_complete,
        "startup_warmup_seconds": (
            round(warmup_seconds, 3) if warmup_seconds is not None else None
        ),
        "startup_warmup_metrics": warmup_metrics,
        "startup_warmup_error": warmup_error,
        "progressive_phrase_mode": PROGRESSIVE_PHRASE_MODE,
        "phrase_first_target_chars": PHRASE_FIRST_TARGET_CHARS,
        "phrase_target_chars": PHRASE_TARGET_CHARS,
        "phrase_max_chars": PHRASE_MAX_CHARS,
        "error": startup_error,
    }
    return JSONResponse(body, status_code=200 if body["ok"] else 503)


@app.get("/client-config")
async def client_config() -> JSONResponse:
    raw = os.getenv("ICE_SERVERS_JSON", "[]")
    try:
        ice_servers = json.loads(raw)
        if not isinstance(ice_servers, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(500, "ICE_SERVERS_JSON must be a JSON array")
    return JSONResponse({"iceServers": ice_servers})


@app.post("/offer")
async def offer(request: Request) -> JSONResponse:
    params = await request.json()
    if not isinstance(params.get("sdp"), str) or params.get("type") != "offer":
        raise HTTPException(400, "Expected an SDP offer")

    pc = RTCPeerConnection()
    pcs.add(pc)
    playback = PlaybackBuffer()
    jobs: set[asyncio.Task[Any]] = set()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        LOG.info("Peer state: %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed"}:
            for job in tuple(jobs):
                job.cancel()
            await pc.close()
            pcs.discard(pc)

    @pc.on("datachannel")
    def on_datachannel(channel: Any) -> None:
        LOG.info("Data channel connected: %s", channel.label)

        @channel.on("open")
        def on_open() -> None:
            task = asyncio.create_task(_send_event(channel, "ready", message="Ready"))
            jobs.add(task)
            task.add_done_callback(jobs.discard)

        @channel.on("message")
        def on_message(message: Any) -> None:
            try:
                payload = json.loads(message) if isinstance(message, str) else {}
            except json.JSONDecodeError:
                payload = {"text": str(message)}

            text = str(payload.get("text", "")).strip()
            instruction = str(payload.get("instruction", TTS_INSTRUCTION)).strip()
            if not text:
                return
            if len(text) > MAX_TEXT_LENGTH:
                task = asyncio.create_task(
                    _send_event(
                        channel,
                        "error",
                        message=f"Text is limited to {MAX_TEXT_LENGTH} characters.",
                    )
                )
            else:
                task = asyncio.create_task(
                    _create_clip(text, instruction, playback, channel)
                )
            jobs.add(task)
            task.add_done_callback(jobs.discard)

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    )
    pc.addTrack(AvatarVideoTrack(playback))
    pc.addTrack(AvatarAudioTrack(playback))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return JSONResponse(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )
