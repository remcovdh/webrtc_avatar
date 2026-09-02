"""WebRTC test server for a Breeze TTS 2 + JoyVASA + FasterLivePortrait avatar.

The neural stages currently prepare a complete utterance before playback. Once
prepared, audio and video are paced together over a persistent WebRTC session.
"""

from __future__ import annotations

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
import onnxruntime as ort
import torch
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
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

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
VIDEO_FPS = 30
AUDIO_RATE = 48_000
AUDIO_SAMPLES = 960  # 20 ms at 48 kHz

pcs: set[RTCPeerConnection] = set()
pipeline: GradioLivePortraitPipeline | None = None
startup_error: str | None = None
inference_lock = asyncio.Lock()


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

    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.infer_params.flag_pasteback = True
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


class PlaybackBuffer:
    """Per-peer playback state consumed by the two WebRTC tracks."""

    def __init__(self) -> None:
        self.video: deque[np.ndarray] = deque()
        self.audio: deque[np.ndarray] = deque()

    @property
    def busy(self) -> bool:
        return bool(self.video or self.audio)

    def load(self, frames: list[np.ndarray], fps: float, pcm_24k: np.ndarray) -> float:
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
        self.video = deque(frames[index] for index in video_indices)

        chunks: deque[np.ndarray] = deque()
        required_samples = max(len(pcm_48k), math.ceil(duration * AUDIO_RATE))
        padded = np.pad(pcm_48k, (0, required_samples - len(pcm_48k)))
        for offset in range(0, len(padded), AUDIO_SAMPLES):
            chunk = padded[offset : offset + AUDIO_SAMPLES]
            if len(chunk) < AUDIO_SAMPLES:
                chunk = np.pad(chunk, (0, AUDIO_SAMPLES - len(chunk)))
            chunks.append(chunk.reshape(1, -1))
        self.audio = chunks
        return duration

    def next_video(self) -> np.ndarray:
        return self.video.popleft() if self.video else BASE_AVATAR

    def next_audio(self) -> np.ndarray:
        if self.audio:
            return self.audio.popleft()
        return np.zeros((1, AUDIO_SAMPLES), dtype=np.int16)

    def clear(self) -> None:
        self.video.clear()
        self.audio.clear()


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


async def _synthesize(text: str, instruction: str, pcm_path: Path, wav_path: Path) -> np.ndarray:
    """Call Breeze's official endpoint and retain its raw 24 kHz PCM."""
    form = {
        "text": (None, text),
        "instruction": (None, instruction),
        "cfg_scale": (None, str(TTS_CFG_SCALE)),
        "seed": (None, str(TTS_SEED)),
    }
    timeout = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", f"{TTS_URL}/v1/audio/speech", files=form
        ) as response:
            if response.is_error:
                detail = (await response.aread()).decode("utf-8", errors="replace")[:300]
                raise RuntimeError(
                    f"Breeze TTS failed ({response.status_code}): {detail}"
                )
            with pcm_path.open("wb") as output:
                async for chunk in response.aiter_bytes():
                    output.write(chunk)

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
    return pcm


def _render_animation(wav_path: Path, output_dir: Path) -> tuple[list[np.ndarray], float]:
    if pipeline is None:
        raise RuntimeError(startup_error or "FasterLivePortrait is not ready")

    video_path, _crop_path, _elapsed = pipeline.run_audio_driving(
        str(wav_path), str(AVATAR_PATH), save_dir=str(output_dir)
    )
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: list[np.ndarray] = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(_letterbox(frame))
    capture.release()
    return frames, fps


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

    if inference_lock.locked():
        await _send_event(channel, "status", phase="queued", message="Queued behind another request…")

    try:
        async with inference_lock:
            await _send_event(channel, "status", phase="tts", message="Generating speech…")
            RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="avatar-", dir=RESULTS_ROOT) as tmp:
                tmp_dir = Path(tmp)
                pcm = await _synthesize(
                    text,
                    instruction or TTS_INSTRUCTION,
                    tmp_dir / "speech.pcm",
                    tmp_dir / "speech.wav",
                )
                await _send_event(
                    channel,
                    "status",
                    phase="animation",
                    message="Generating neural facial motion…",
                )
                frames, fps = await asyncio.to_thread(
                    _render_animation, tmp_dir / "speech.wav", tmp_dir
                )
                duration = playback.load(frames, fps, pcm)

        await _send_event(
            channel,
            "playing",
            message="Playing synchronized avatar stream",
            duration=round(duration, 3),
        )
        await asyncio.sleep(duration + 0.25)
        await _send_event(channel, "ready", message="Ready")
    except asyncio.CancelledError:
        playback.clear()
        raise
    except Exception as exc:
        LOG.exception("Avatar generation failed")
        await _send_event(channel, "error", message=str(exc))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global BASE_AVATAR, pipeline, startup_error
    try:
        BASE_AVATAR = _load_avatar()
        LOG.info("Loading FasterLivePortrait and source portrait")
        pipeline = await asyncio.to_thread(_initialize_pipeline)
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
        "ok": startup_error is None and tts_ready,
        "avatar_ready": startup_error is None,
        "tts_ready": tts_ready,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "onnx_providers": providers,
        "cuda_provider": "CUDAExecutionProvider" in providers,
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
