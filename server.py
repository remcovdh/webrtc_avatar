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
from dataclasses import dataclass
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
from src.pipelines.joyvasa_audio_to_motion_pipeline import (
    JoyVASAAudio2MotionPipeline,
)

LOG = logging.getLogger("avatar")
SERVER_BUILD = "neural-avatar-v2j-neural-idle-frame"
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
TTS_DESIGN_CFG_SCALE = float(
    os.getenv("BREEZE_DESIGN_CFG_SCALE", os.getenv("BREEZE_CFG_SCALE", "4"))
)
TTS_PRESET_CLONE_CFG_SCALE = float(
    os.getenv("BREEZE_PRESET_CLONE_CFG_SCALE", "1")
)
TTS_PRESET_DIRECTION_CFG_SCALE = float(
    os.getenv("BREEZE_PRESET_DIRECTION_CFG_SCALE", "4")
)
TTS_PRESET_CLONE_INSTRUCTION = os.getenv(
    "BREEZE_PRESET_CLONE_INSTRUCTION",
    "Speak naturally in the reference voice.",
).strip() or "Speak naturally in the reference voice."
TTS_PRESET_AUDIO_PATH = Path(
    os.getenv(
        "BREEZE_PRESET_AUDIO_PATH",
        "/workspace/inputs/voice-preset.wav",
    )
)
TTS_PRESET_TRANSCRIPT = os.getenv("BREEZE_PRESET_TRANSCRIPT", "").strip()
TTS_PRESET_TRANSCRIPT_FILE = Path(
    os.getenv(
        "BREEZE_PRESET_TRANSCRIPT_FILE",
        "/workspace/inputs/voice-preset.txt",
    )
)
TTS_DEFAULT_VOICE_MODE = os.getenv(
    "BREEZE_DEFAULT_VOICE_MODE", "design"
).strip().lower()
TTS_SEED = int(os.getenv("BREEZE_SEED", "42"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "500"))
AVATAR_PASTE_BACK = _env_bool("AVATAR_PASTE_BACK", False)
DIRECT_MEMORY_RENDER = _env_bool("DIRECT_MEMORY_RENDER", True)
RENDER_STRIDE = max(1, int(os.getenv("RENDER_STRIDE", "2")))
ADAPTIVE_RENDER_STRIDE = _env_bool("ADAPTIVE_RENDER_STRIDE", True)
CATCHUP_RENDER_STRIDE = max(
    RENDER_STRIDE, int(os.getenv("CATCHUP_RENDER_STRIDE", "3"))
)
CATCHUP_BUFFER_SECONDS = max(
    0.0, float(os.getenv("CATCHUP_BUFFER_SECONDS", "0.75"))
)
STARTUP_WARMUP = _env_bool("STARTUP_WARMUP", True)
WARMUP_TEXT = os.getenv("WARMUP_TEXT", "Hello.").strip() or "Hello."
USE_NEURAL_IDLE_FRAME = _env_bool("USE_NEURAL_IDLE_FRAME", True)
WARMUP_IDLE_FRAME_INDEX = int(os.getenv("WARMUP_IDLE_FRAME_INDEX", "0"))
TTS_STARTUP_WAIT_SECONDS = max(
    0.0, float(os.getenv("TTS_STARTUP_WAIT_SECONDS", "300"))
)
TTS_STARTUP_POLL_SECONDS = max(
    0.25, float(os.getenv("TTS_STARTUP_POLL_SECONDS", "2"))
)
PROGRESSIVE_PHRASE_MODE = _env_bool("PROGRESSIVE_PHRASE_MODE", True)
TTS_PREFETCH = _env_bool("TTS_PREFETCH", False)
PERSISTENT_PHRASE_MOTION = _env_bool("PERSISTENT_PHRASE_MOTION", True)
AVATAR_RELATIVE_MOTION = _env_bool("AVATAR_RELATIVE_MOTION", True)
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
tts_startup_wait_seconds: float | None = None
idle_frame_source = "original-avatar"
idle_frame_index: int | None = None

VOICE_MODE_DESIGN = "design"
VOICE_MODE_PRESET_CLONE = "preset-clone"
VOICE_MODE_PRESET_DIRECTION = "preset-direction"
VOICE_MODES = {
    VOICE_MODE_DESIGN,
    VOICE_MODE_PRESET_CLONE,
    VOICE_MODE_PRESET_DIRECTION,
}


@dataclass(frozen=True)
class VoiceRequest:
    mode: str
    instruction: str
    cfg_scale: float
    reference_path: Path | None = None
    reference_text: str = ""


def _load_preset_transcript() -> str:
    """Return the configured exact transcript without accepting client paths."""
    if TTS_PRESET_TRANSCRIPT:
        return TTS_PRESET_TRANSCRIPT
    try:
        return TTS_PRESET_TRANSCRIPT_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _preset_configuration() -> tuple[bool, str, str]:
    transcript = _load_preset_transcript()
    if not TTS_PRESET_AUDIO_PATH.is_file():
        return False, transcript, f"Missing {TTS_PRESET_AUDIO_PATH}"
    if not transcript:
        return (
            False,
            transcript,
            "Missing exact preset transcript in "
            f"{TTS_PRESET_TRANSCRIPT_FILE} or BREEZE_PRESET_TRANSCRIPT",
        )
    if TTS_PRESET_AUDIO_PATH.stat().st_size == 0:
        return False, transcript, f"Preset audio is empty: {TTS_PRESET_AUDIO_PATH}"
    return True, transcript, "ready"


def _resolve_voice_request(mode: str, instruction: str) -> VoiceRequest:
    normalized = (mode or TTS_DEFAULT_VOICE_MODE).strip().lower()
    aliases = {
        "preset": VOICE_MODE_PRESET_CLONE,
        "clone": VOICE_MODE_PRESET_CLONE,
        "direction": VOICE_MODE_PRESET_DIRECTION,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VOICE_MODES:
        raise ValueError(
            f"Unknown voice mode {normalized!r}; choose design, "
            "preset-clone or preset-direction."
        )

    requested_instruction = instruction.strip() or TTS_INSTRUCTION
    if normalized == VOICE_MODE_DESIGN:
        return VoiceRequest(
            mode=normalized,
            instruction=requested_instruction,
            cfg_scale=TTS_DESIGN_CFG_SCALE,
        )

    configured, transcript, problem = _preset_configuration()
    if not configured:
        raise ValueError(
            f"Preset voice is not configured: {problem}. Add a clean WAV and "
            "its exact transcript, then recreate the avatar service."
        )

    if normalized == VOICE_MODE_PRESET_CLONE:
        return VoiceRequest(
            mode=normalized,
            instruction=TTS_PRESET_CLONE_INSTRUCTION,
            cfg_scale=TTS_PRESET_CLONE_CFG_SCALE,
            reference_path=TTS_PRESET_AUDIO_PATH,
            reference_text=transcript,
        )

    return VoiceRequest(
        mode=normalized,
        instruction=requested_instruction,
        cfg_scale=TTS_PRESET_DIRECTION_CFG_SCALE,
        reference_path=TTS_PRESET_AUDIO_PATH,
        reference_text=transcript,
    )

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
    # Relative motion maps driving deltas onto the source pose. Persistence
    # below then keeps one reference across all phrases in an utterance.
    cfg.infer_params.flag_relative_motion = AVATAR_RELATIVE_MOTION
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
    voice: VoiceRequest,
    pcm_path: Path,
    wav_path: Path,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Call Breeze and report request, first-byte, transfer and finalize time."""
    request_started = time.perf_counter()
    form = {
        "text": (None, text),
        "instruction": (None, voice.instruction),
        "cfg_scale": (None, str(voice.cfg_scale)),
        "seed": (None, str(TTS_SEED)),
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
        form["ref_text"] = (None, voice.reference_text)

    timeout = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=15.0)
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
        "voice_mode": voice.mode,
        "cfg_scale": voice.cfg_scale,
        "reference_used": voice.reference_path is not None,
        "reference_bytes": reference_bytes,
    }


def _render_animation_legacy(
    wav_path: Path,
    output_dir: Path,
) -> tuple[list[np.ndarray], float, dict[str, Any]]:
    """Render through FLP's stock pickle, MP4, FFmpeg and decode path."""
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
        "backend": "legacy-mp4",
        "motion_ms": 0,
        "frame_loop_ms": 0,
        "pipeline_ms": round(pipeline_seconds * 1000),
        "decode_ms": round(decode_seconds * 1000),
        "total_ms": round((time.perf_counter() - render_started) * 1000),
        "pipeline_reported_ms": round(float(reported_elapsed or 0) * 1000),
        "render_stride": 1,
        "motion_frames": len(frames),
        "frames": len(frames),
        "source_fps": round(fps, 3),
        "playback_fps": round(fps, 3),
        "effective_fps": 0.0,
    }


def _ensure_joyvasa_pipeline() -> None:
    """Create JoyVASA exactly as FLP's run_audio_driving does."""
    if pipeline is None:
        raise RuntimeError(startup_error or "FasterLivePortrait is not ready")
    if pipeline.joyvasa_pipe is not None:
        return

    pipeline.joyvasa_pipe = JoyVASAAudio2MotionPipeline(
        motion_model_path=pipeline.cfg.joyvasa_models.motion_model_path,
        audio_model_path=pipeline.cfg.joyvasa_models.audio_model_path,
        motion_template_path=pipeline.cfg.joyvasa_models.motion_template_path,
        cfg_mode=pipeline.cfg.infer_params.cfg_mode,
        cfg_scale=pipeline.cfg.infer_params.cfg_scale,
    )


def _render_animation_direct(
    wav_path: Path,
    render_stride: int | None = None,
    reset_motion_reference: bool = True,
) -> tuple[list[np.ndarray], float, dict[str, Any]]:
    """Run JoyVASA and FLP frames in memory, without pickle/video/FFmpeg I/O.

    This deliberately keeps phrase-level batching so render stride can be
    measured independently. A later step can append frames incrementally.
    """
    if pipeline is None:
        raise RuntimeError(startup_error or "FasterLivePortrait is not ready")
    if pipeline.is_source_video:
        raise RuntimeError("Direct-memory rendering currently requires a source image")

    selected_stride = max(1, render_stride or RENDER_STRIDE)
    render_started = time.perf_counter()
    _ensure_joyvasa_pipeline()
    assert pipeline.joyvasa_pipe is not None

    motion_started = time.perf_counter()
    motion_info = pipeline.joyvasa_pipe.gen_motion_sequence(str(wav_path))
    motion_seconds = time.perf_counter() - motion_started

    source_fps = float(motion_info.get("output_fps") or 25.0)
    playback_fps = source_fps / selected_stride
    motion_list = motion_info["motion"]
    eyes_list = motion_info.get("c_eyes_lst", motion_info.get("c_d_eyes_lst"))
    lips_list = motion_info.get("c_lip_lst", motion_info.get("c_d_lip_lst"))

    frame_loop_started = time.perf_counter()
    frames: list[np.ndarray] = []
    for frame_index in range(0, len(motion_list), selected_stride):
        motion = motion_list[frame_index]
        eyes = (
            eyes_list[frame_index]
            if eyes_list is not None and frame_index < len(eyes_list)
            else None
        )
        lips = (
            lips_list[frame_index]
            if lips_list is not None and frame_index < len(lips_list)
            else None
        )
        output = pipeline.run_with_pkl(
            [motion, eyes, lips],
            pipeline.src_imgs[0],
            pipeline.src_infos[0],
            # FasterLivePortrait stores the initial driving rotation and motion
            # reference on first_frame. Reset once per user utterance, not once
            # per phrase, so later chunks remain in the same motion space.
            first_frame=reset_motion_reference and frame_index == 0,
        )
        out_crop = output[0]
        if out_crop is None:
            LOG.warning("Direct renderer returned no face for frame %d", frame_index)
            continue
        frames.append(
            _letterbox(cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR))
        )

    frame_loop_seconds = time.perf_counter() - frame_loop_started
    total_seconds = time.perf_counter() - render_started
    return frames, playback_fps, {
        "backend": "direct-memory",
        "motion_ms": round(motion_seconds * 1000),
        "frame_loop_ms": round(frame_loop_seconds * 1000),
        "pipeline_ms": round(total_seconds * 1000),
        "decode_ms": 0,
        "total_ms": round(total_seconds * 1000),
        "pipeline_reported_ms": 0,
        "render_stride": selected_stride,
        "motion_frames": len(motion_list),
        "frames": len(frames),
        "source_fps": round(source_fps, 3),
        "playback_fps": round(playback_fps, 3),
        "effective_fps": round(
            len(frames) / frame_loop_seconds if frame_loop_seconds > 0 else 0.0,
            3,
        ),
        "motion_reference_reset": reset_motion_reference,
        "persistent_phrase_motion": PERSISTENT_PHRASE_MOTION,
        "relative_motion": AVATAR_RELATIVE_MOTION,
    }


def _render_animation(
    wav_path: Path,
    output_dir: Path,
    render_stride: int | None = None,
    reset_motion_reference: bool = True,
) -> tuple[list[np.ndarray], float, dict[str, Any]]:
    if DIRECT_MEMORY_RENDER and not AVATAR_PASTE_BACK:
        return _render_animation_direct(
            wav_path,
            render_stride,
            reset_motion_reference,
        )
    return _render_animation_legacy(wav_path, output_dir)


def _select_render_stride(
    phrase_index: int,
    buffered_seconds: float,
) -> tuple[int, str]:
    """Select temporal quality from the buffer state at render start."""
    if not DIRECT_MEMORY_RENDER or AVATAR_PASTE_BACK:
        return 1, "legacy-backend"
    if not ADAPTIVE_RENDER_STRIDE or CATCHUP_RENDER_STRIDE == RENDER_STRIDE:
        return RENDER_STRIDE, "fixed"
    if phrase_index == 1:
        return RENDER_STRIDE, "first-phrase-quality"
    if buffered_seconds < CATCHUP_BUFFER_SECONDS:
        return CATCHUP_RENDER_STRIDE, "low-buffer-catchup"
    return RENDER_STRIDE, "buffer-healthy"


async def _prepare_phrase_audio(
    phrase: str,
    voice: VoiceRequest,
    tmp_dir: Path,
    index: int,
    phrase_count: int,
    channel: Any,
    prefetched: bool,
) -> dict[str, Any]:
    """Synthesize one phrase in its own directory and retain timing metadata."""
    chunk_dir = tmp_dir / f"chunk-{index:03d}"
    chunk_dir.mkdir(exist_ok=True)
    pcm_path = chunk_dir / "speech.pcm"
    wav_path = chunk_dir / "speech.wav"
    await _send_event(
        channel,
        "status",
        phase="tts-prefetch" if prefetched else "tts",
        chunk=index,
        chunks=phrase_count,
        message=(
            f"Phrase {index}/{phrase_count}: pre-generating speech during render…"
            if prefetched
            else f"Phrase {index}/{phrase_count}: generating speech…"
        ),
    )
    started = time.perf_counter()
    pcm, detail = await _synthesize(
        phrase,
        voice,
        pcm_path,
        wav_path,
    )
    completed = time.perf_counter()
    return {
        "chunk_dir": chunk_dir,
        "wav_path": wav_path,
        "pcm": pcm,
        "detail": detail,
        "seconds": completed - started,
        "prefetched": prefetched,
    }


async def _create_clip(
    text: str,
    instruction: str,
    voice_mode: str,
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

    try:
        voice = _resolve_voice_request(voice_mode, instruction)
    except ValueError as exc:
        await _send_event(channel, "error", message=str(exc))
        return

    playback.begin()
    request_started = time.perf_counter()
    total_tts = 0.0
    total_render = 0.0
    tts_task: asyncio.Task[dict[str, Any]] | None = None

    await _send_event(
        channel,
        "plan",
        message=(
            f"Prepared {len(phrases)} progressive phrase"
            f"{'s' if len(phrases) != 1 else ''} with {voice.mode} voice"
        ),
        phrases=phrases,
        voice_mode=voice.mode,
        cfg_scale=voice.cfg_scale,
    )

    if inference_lock.locked():
        await _send_event(channel, "status", phase="queued", message="Queued behind another request…")

    try:
        async with inference_lock:
            RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="avatar-", dir=RESULTS_ROOT) as tmp:
                tmp_dir = Path(tmp)
                for index, phrase in enumerate(phrases, start=1):
                    underruns_before = playback.audio_underruns
                    video_underruns_before = playback.video_underruns

                    if tts_task is None:
                        tts_task = asyncio.create_task(
                            _prepare_phrase_audio(
                                phrase,
                                voice,
                                tmp_dir,
                                index,
                                len(phrases),
                                channel,
                                prefetched=False,
                            )
                        )
                    current_tts_task = tts_task
                    tts_task = None
                    tts_wait_started = time.perf_counter()
                    audio_result = await current_tts_task
                    tts_wait_seconds = time.perf_counter() - tts_wait_started
                    chunk_dir = audio_result["chunk_dir"]
                    wav_path = audio_result["wav_path"]
                    pcm = audio_result["pcm"]
                    tts_detail = audio_result["detail"]
                    tts_seconds = audio_result["seconds"]
                    tts_wait_ms = round(tts_wait_seconds * 1000)
                    tts_overlap_ms = max(0, round(tts_seconds * 1000) - tts_wait_ms)
                    total_tts += tts_seconds

                    await _send_event(
                        channel,
                        "status",
                        phase="animation",
                        chunk=index,
                        chunks=len(phrases),
                        message=f"Phrase {index}/{len(phrases)}: rendering facial motion…",
                    )
                    if TTS_PREFETCH and index < len(phrases):
                        tts_task = asyncio.create_task(
                            _prepare_phrase_audio(
                                phrases[index],
                                voice,
                                tmp_dir,
                                index + 1,
                                len(phrases),
                                channel,
                                prefetched=True,
                            )
                        )
                    buffer_before_render = playback.buffered_seconds
                    selected_stride, stride_reason = _select_render_stride(
                        index,
                        buffer_before_render,
                    )
                    started = time.perf_counter()
                    frames, fps, render_detail = await asyncio.to_thread(
                        _render_animation,
                        wav_path,
                        chunk_dir,
                        selected_stride,
                        index == 1 or not PERSISTENT_PHRASE_MOTION,
                    )
                    render_detail["adaptive_stride"] = ADAPTIVE_RENDER_STRIDE
                    render_detail["stride_reason"] = stride_reason
                    render_detail["buffer_before_render_seconds"] = round(
                        buffer_before_render,
                        3,
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
                        "Phrase %d/%d timings: voice=%s cfg=%.2f tts=%.3fs tts_wait=%dms "
                        "tts_overlap=%dms prefetched=%s first_byte=%dms download=%dms "
                        "render=%.3fs backend=%s stride=%d adaptive=%s reason=%s "
                        "buffer_before=%.3fs motion=%dms "
                        "motion_reference_reset=%s persistent_motion=%s "
                        "frame_loop=%dms effective_fps=%.2f pipeline=%dms "
                        "decode=%dms frames=%d/%d playback_fps=%.2f media=%.3fs "
                        "buffer=%.3fs underrun=%.0fms",
                        index,
                        len(phrases),
                        voice.mode,
                        voice.cfg_scale,
                        tts_seconds,
                        tts_wait_ms,
                        tts_overlap_ms,
                        audio_result["prefetched"],
                        tts_detail["first_byte_ms"],
                        tts_detail["download_ms"],
                        render_seconds,
                        render_detail["backend"],
                        render_detail["render_stride"],
                        render_detail["adaptive_stride"],
                        render_detail["stride_reason"],
                        render_detail["buffer_before_render_seconds"],
                        render_detail["motion_ms"],
                        render_detail.get("motion_reference_reset", True),
                        render_detail.get("persistent_phrase_motion", False),
                        render_detail["frame_loop_ms"],
                        render_detail["effective_fps"],
                        render_detail["pipeline_ms"],
                        render_detail["decode_ms"],
                        render_detail["frames"],
                        render_detail["motion_frames"],
                        render_detail["playback_fps"],
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
                        voice_mode=voice.mode,
                        voice_cfg_scale=voice.cfg_scale,
                        voice_reference_used=voice.reference_path is not None,
                        tts_ms=round(tts_seconds * 1000),
                        tts_wait_ms=tts_wait_ms,
                        tts_overlap_ms=tts_overlap_ms,
                        tts_prefetched=audio_result["prefetched"],
                        tts_headers_ms=tts_detail["headers_ms"],
                        tts_first_byte_ms=tts_detail["first_byte_ms"],
                        tts_download_ms=tts_detail["download_ms"],
                        tts_finalize_ms=tts_detail["finalize_ms"],
                        tts_bytes=tts_detail["bytes"],
                        render_ms=round(render_seconds * 1000),
                        render_backend=render_detail["backend"],
                        render_stride=render_detail["render_stride"],
                        render_adaptive_stride=render_detail["adaptive_stride"],
                        render_stride_reason=render_detail["stride_reason"],
                        render_buffer_before_seconds=render_detail[
                            "buffer_before_render_seconds"
                        ],
                        render_motion_ms=render_detail["motion_ms"],
                        render_motion_reference_reset=render_detail.get(
                            "motion_reference_reset", True
                        ),
                        render_persistent_phrase_motion=render_detail.get(
                            "persistent_phrase_motion", False
                        ),
                        render_frame_loop_ms=render_detail["frame_loop_ms"],
                        render_effective_fps=render_detail["effective_fps"],
                        render_pipeline_ms=render_detail["pipeline_ms"],
                        render_decode_ms=render_detail["decode_ms"],
                        render_pipeline_reported_ms=render_detail[
                            "pipeline_reported_ms"
                        ],
                        render_motion_frames=render_detail["motion_frames"],
                        render_frames=render_detail["frames"],
                        render_source_fps=render_detail["source_fps"],
                        render_playback_fps=render_detail["playback_fps"],
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
    finally:
        if tts_task is not None:
            if not tts_task.done():
                tts_task.cancel()
            await asyncio.gather(tts_task, return_exceptions=True)


async def _wait_for_tts() -> float:
    """Wait for Breeze because manual/partial Compose starts can bypass depends_on."""
    started = time.perf_counter()
    deadline = started + TTS_STARTUP_WAIT_SECONDS
    last_problem = "no response"
    announced = False

    while True:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{TTS_URL}/docs")
            if response.status_code < 500:
                waited = time.perf_counter() - started
                if announced:
                    LOG.info("Breeze became ready after %.3fs", waited)
                return waited
            last_problem = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last_problem = f"{type(exc).__name__}: {exc}"

        now = time.perf_counter()
        if now >= deadline:
            raise TimeoutError(
                f"Breeze was not ready at {TTS_URL} after "
                f"{TTS_STARTUP_WAIT_SECONDS:.1f}s ({last_problem})"
            )
        if not announced:
            LOG.info(
                "Waiting up to %.1fs for Breeze at %s before startup warm-up",
                TTS_STARTUP_WAIT_SECONDS,
                TTS_URL,
            )
            announced = True
        await asyncio.sleep(min(TTS_STARTUP_POLL_SECONDS, deadline - now))


async def _warmup_pipeline() -> None:
    """Exercise the same TTS, JoyVASA and renderer path before the first user."""
    global warmup_complete, warmup_seconds, warmup_metrics, warmup_error
    global tts_startup_wait_seconds
    global BASE_AVATAR, idle_frame_source, idle_frame_index

    started = time.perf_counter()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG.info("Startup warm-up begins with %r", WARMUP_TEXT)
    try:
        tts_startup_wait_seconds = await _wait_for_tts()
        with tempfile.TemporaryDirectory(prefix="warmup-", dir=RESULTS_ROOT) as tmp:
            warmup_dir = Path(tmp)
            pcm_path = warmup_dir / "warmup.pcm"
            wav_path = warmup_dir / "warmup.wav"
            try:
                warmup_voice = _resolve_voice_request(
                    TTS_DEFAULT_VOICE_MODE,
                    TTS_INSTRUCTION,
                )
            except ValueError as exc:
                LOG.warning(
                    "Default voice mode is unavailable during warm-up (%s); "
                    "warming the designed voice instead",
                    exc,
                )
                warmup_voice = _resolve_voice_request(
                    VOICE_MODE_DESIGN,
                    TTS_INSTRUCTION,
                )
            pcm, tts_detail = await _synthesize(
                WARMUP_TEXT,
                warmup_voice,
                pcm_path,
                wav_path,
            )
            frames, fps, render_detail = await asyncio.to_thread(
                _render_animation,
                wav_path,
                warmup_dir,
            )
            if USE_NEURAL_IDLE_FRAME:
                if not frames:
                    raise RuntimeError(
                        "Startup warm-up returned no frame for the neural idle image"
                    )
                selected_index = min(
                    max(WARMUP_IDLE_FRAME_INDEX, 0),
                    len(frames) - 1,
                )
                # WebRTC must start in the same aligned, neural render space as
                # speech. The source photograph stays hidden and is used only
                # to initialise FasterLivePortrait.
                BASE_AVATAR = frames[selected_index].copy()
                idle_frame_source = "startup-warmup-neural-frame"
                idle_frame_index = selected_index
                LOG.info(
                    "Neural idle frame prepared from warm-up frame %d/%d",
                    selected_index,
                    len(frames),
                )
            warmup_seconds = time.perf_counter() - started
            warmup_metrics = {
                "tts_startup_wait_ms": round(tts_startup_wait_seconds * 1000),
                "tts": tts_detail,
                "voice_mode": warmup_voice.mode,
                "voice_cfg_scale": warmup_voice.cfg_scale,
                "render": render_detail,
                "idle_frame_source": idle_frame_source,
                "idle_frame_index": idle_frame_index,
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
    preset_configured, _preset_transcript, preset_problem = _preset_configuration()
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
        "direct_memory_render": DIRECT_MEMORY_RENDER,
        "render_backend": (
            "direct-memory"
            if DIRECT_MEMORY_RENDER and not AVATAR_PASTE_BACK
            else "legacy-mp4"
        ),
        "configured_render_stride": RENDER_STRIDE,
        "adaptive_render_stride": ADAPTIVE_RENDER_STRIDE,
        "catchup_render_stride": CATCHUP_RENDER_STRIDE,
        "catchup_buffer_seconds": CATCHUP_BUFFER_SECONDS,
        "render_stride": (
            RENDER_STRIDE
            if DIRECT_MEMORY_RENDER and not AVATAR_PASTE_BACK
            else 1
        ),
        "startup_warmup_enabled": STARTUP_WARMUP,
        "startup_warmup_complete": warmup_complete,
        "startup_warmup_seconds": (
            round(warmup_seconds, 3) if warmup_seconds is not None else None
        ),
        "startup_warmup_metrics": warmup_metrics,
        "startup_warmup_error": warmup_error,
        "neural_idle_frame_enabled": USE_NEURAL_IDLE_FRAME,
        "idle_frame_source": idle_frame_source,
        "idle_frame_index": idle_frame_index,
        "tts_startup_wait_limit_seconds": TTS_STARTUP_WAIT_SECONDS,
        "tts_startup_poll_seconds": TTS_STARTUP_POLL_SECONDS,
        "tts_startup_wait_seconds": (
            round(tts_startup_wait_seconds, 3)
            if tts_startup_wait_seconds is not None
            else None
        ),
        "progressive_phrase_mode": PROGRESSIVE_PHRASE_MODE,
        "tts_prefetch": TTS_PREFETCH,
        "tts_prefetch_depth": 1 if TTS_PREFETCH else 0,
        "persistent_phrase_motion": PERSISTENT_PHRASE_MOTION,
        "relative_motion": AVATAR_RELATIVE_MOTION,
        "default_voice_mode": TTS_DEFAULT_VOICE_MODE,
        "voice_modes": sorted(VOICE_MODES),
        "preset_voice_configured": preset_configured,
        "preset_voice_status": preset_problem,
        "design_cfg_scale": TTS_DESIGN_CFG_SCALE,
        "preset_clone_cfg_scale": TTS_PRESET_CLONE_CFG_SCALE,
        "preset_direction_cfg_scale": TTS_PRESET_DIRECTION_CFG_SCALE,
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
    preset_configured, _preset_transcript, preset_problem = _preset_configuration()
    default_mode = TTS_DEFAULT_VOICE_MODE
    if default_mode not in VOICE_MODES:
        default_mode = VOICE_MODE_DESIGN
    if default_mode != VOICE_MODE_DESIGN and not preset_configured:
        default_mode = VOICE_MODE_DESIGN

    voice_modes = [
        {
            "id": VOICE_MODE_DESIGN,
            "label": "Designed voice",
            "description": "Create a voice from the direction for every request.",
            "available": True,
            "cfgScale": TTS_DESIGN_CFG_SCALE,
        },
        {
            "id": VOICE_MODE_PRESET_CLONE,
            "label": "Preset voice (clone)",
            "description": "Keep the preset identity with a neutral delivery.",
            "available": preset_configured,
            "cfgScale": TTS_PRESET_CLONE_CFG_SCALE,
        },
        {
            "id": VOICE_MODE_PRESET_DIRECTION,
            "label": "Preset voice + direction",
            "description": "Keep the preset identity and apply the direction.",
            "available": preset_configured,
            "cfgScale": TTS_PRESET_DIRECTION_CFG_SCALE,
        },
    ]
    return JSONResponse(
        {
            "iceServers": ice_servers,
            "voiceModes": voice_modes,
            "defaultVoiceMode": default_mode,
            "defaultVoiceInstruction": TTS_INSTRUCTION,
            "presetVoiceConfigured": preset_configured,
            "presetVoiceStatus": preset_problem,
            "presetVoiceFilename": (
                TTS_PRESET_AUDIO_PATH.name if preset_configured else None
            ),
        }
    )


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

        ready_announced = False

        def announce_ready() -> None:
            """Send ready even when aiortc reports the channel already open."""
            nonlocal ready_announced
            if ready_announced or getattr(channel, "readyState", None) != "open":
                return
            ready_announced = True
            task = asyncio.create_task(_send_event(channel, "ready", message="Ready"))
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

            text = str(payload.get("text", "")).strip()
            instruction = str(payload.get("instruction", TTS_INSTRUCTION)).strip()
            voice_mode = str(
                payload.get("voice_mode", TTS_DEFAULT_VOICE_MODE)
            ).strip()
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
                    _create_clip(
                        text,
                        instruction,
                        voice_mode,
                        playback,
                        channel,
                    )
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
