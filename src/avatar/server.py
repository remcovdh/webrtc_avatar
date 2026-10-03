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
from avatar.playback import AUDIO_RATE, AUDIO_SAMPLES, VIDEO_FPS, IdleFrame, PlaybackBuffer
from avatar.tracks import AvatarAudioTrack, AvatarVideoTrack
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
# Chatterbox behind chatterbox_api.py is the only TTS provider.
TTS_PROVIDER = "chatterbox"
TTS_URL = SETTINGS.tts_url.rstrip("/")
TTS_HEALTH_PATH = "/health"
# Reference recording for the cloned voice (preset-clone mode).
TTS_PRESET_AUDIO_PATH = Path(SETTINGS.preset_audio_path)
TTS_DEFAULT_VOICE_MODE = SETTINGS.default_voice_mode
TTS_SEED = 42
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
# How long the warm-up waits for the TTS service, and how often it checks.
TTS_STARTUP_WAIT_SECONDS = 300.0
TTS_STARTUP_POLL_SECONDS = 2.0
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
pipeline: GradioLivePortraitPipeline | None = None
startup_error: str | None = None
inference_lock = asyncio.Lock()
warmup_complete = False
warmup_seconds: float | None = None
warmup_metrics: dict[str, Any] = {}
warmup_error: str | None = None
tts_startup_wait_seconds: float | None = None
warping_backend = "cuda"
render_fps_estimate = EXPECTED_RENDER_FPS

# "design" is Chatterbox's built-in voice; "preset-clone" clones the reference
# recording.
VOICE_MODE_DESIGN = "design"
VOICE_MODE_PRESET_CLONE = "preset-clone"
VOICE_MODES = {VOICE_MODE_DESIGN, VOICE_MODE_PRESET_CLONE}


@dataclass(frozen=True)
class VoiceRequest:
    mode: str
    reference_path: Path | None = None


def _preset_configuration() -> tuple[bool, str]:
    if not TTS_PRESET_AUDIO_PATH.is_file():
        return False, f"Missing {TTS_PRESET_AUDIO_PATH}"
    if TTS_PRESET_AUDIO_PATH.stat().st_size == 0:
        return False, f"Preset audio is empty: {TTS_PRESET_AUDIO_PATH}"
    return True, "ready"


def _resolve_voice_request(mode: str) -> VoiceRequest:
    normalized = (mode or TTS_DEFAULT_VOICE_MODE).strip().lower()
    aliases = {
        "preset": VOICE_MODE_PRESET_CLONE,
        "clone": VOICE_MODE_PRESET_CLONE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VOICE_MODES:
        raise ValueError(
            f"Voice mode {normalized!r} is unavailable; "
            f"choose {', '.join(sorted(VOICE_MODES))}."
        )
    if normalized == VOICE_MODE_DESIGN:
        return VoiceRequest(mode=normalized)

    configured, problem = _preset_configuration()
    if not configured:
        raise ValueError(
            f"Preset voice is not configured: {problem}. Add a clean WAV, "
            "then recreate the avatar service."
        )
    return VoiceRequest(mode=normalized, reference_path=TTS_PRESET_AUDIO_PATH)


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


# The image shown while idle; every playback buffer reads it.
IDLE_FRAME = IdleFrame()


_FLP_CROP_IMAGE = flp_pipeline.crop_image


def _crop_source_image(img: np.ndarray, pts: np.ndarray, **kwargs: Any) -> dict:
    """FLP's source crop, upright by default and without black borders.

    The face crop (2.3x the face size) reaches past the edges of a tightly
    framed portrait; FLP filled that with black bands. Redo the same warp with
    mirrored edges so the background and hair continue instead.
    """
    kwargs.setdefault("flag_do_rot", AVATAR_CROP_ROTATION)
    crop = _FLP_CROP_IMAGE(img, pts, **kwargs)
    dsize = kwargs.get("dsize", 224)
    crop["img_crop"] = cv2.warpAffine(
        img,
        crop["M_o2c"][:2],
        (dsize, dsize),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return crop


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
    # The face crop is the output. Pasting it back into the photograph needs
    # a GPU matrix inverse per frame, which failed on a GPU shared with TTS.
    cfg.infer_params.flag_pasteback = False
    # Relative motion maps driving deltas onto the source pose. Persistence
    # below then keeps one reference across all phrases in an utterance.
    cfg.infer_params.flag_relative_motion = AVATAR_RELATIVE_MOTION
    cfg.infer_params.flag_stitching = True
    cfg.infer_params.animation_region = AVATAR_ANIMATION_REGION
    cfg.infer_params.driving_multiplier = AVATAR_DRIVING_MULTIPLIER
    cfg.infer_params.flag_normalize_lip = AVATAR_NORMALIZE_LIP
    cfg.infer_params.flag_eye_retargeting = AVATAR_EYE_RETARGETING
    cfg.infer_params.flag_lip_retargeting = AVATAR_LIP_RETARGETING
    cfg.infer_params.cfg_scale = JOYVASA_CFG_SCALE

    LOG.info(
        "Visual motion config: region=%s multiplier=%.2f normalize_lip=%s "
        "eye_retargeting=%s lip_retargeting=%s",
        AVATAR_ANIMATION_REGION,
        AVATAR_DRIVING_MULTIPLIER,
        AVATAR_NORMALIZE_LIP,
        AVATAR_EYE_RETARGETING,
        AVATAR_LIP_RETARGETING,
    )

    flp_pipeline.crop_image = _crop_source_image
    loaded = GradioLivePortraitPipeline(cfg=cfg, is_animal=False)
    _enable_tensorrt_warping(loaded)
    if not loaded.prepare_source(str(AVATAR_PATH), realtime=False):
        raise ValueError(
            "No face was detected in avatar.jpg. Use a clear, front-facing portrait."
        )
    return loaded


def _enable_tensorrt_warping(loaded: Any) -> None:
    """Swap warping_spade's ORT session for a TensorRT FP16 one if possible.

    The model's inputs and outputs are unchanged, so FLP's predictor keeps
    working; any failure leaves the CUDA session in place.
    """
    global warping_backend
    warping_backend = "cuda"
    if not AVATAR_TENSORRT:
        return
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        LOG.warning("AVATAR_TENSORRT is set but ONNX Runtime has no TensorRT provider")
        return
    model = loaded.model_dict["warping_spade"]
    model_path = model.kwargs["model_path"]
    Path(TENSORRT_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    options = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": TENSORRT_CACHE_DIR,
    }
    started = time.perf_counter()
    try:
        session = ort.InferenceSession(
            model_path,
            providers=[("TensorrtExecutionProvider", options), "CUDAExecutionProvider"],
        )
        # Build (or load) the engines now rather than on the first request.
        session.run(
            None,
            {
                item.name: np.zeros(item.shape, dtype=np.float32)
                for item in session.get_inputs()
            },
        )
    except Exception:
        LOG.exception("TensorRT warping session failed; keeping the CUDA provider")
        return
    model.predictor.onnx_model = session
    warping_backend = "tensorrt-fp16"
    LOG.info(
        "warping_spade uses TensorRT FP16 (ready in %.1fs, cache %s)",
        time.perf_counter() - started,
        TENSORRT_CACHE_DIR,
    )












async def _send_event(channel: Any, event_type: str, **payload: Any) -> None:
    if getattr(channel, "readyState", None) == "open":
        channel.send(json.dumps({"type": event_type, **payload}))


async def _synthesize(
    text: str,
    voice: VoiceRequest,
    pcm_path: Path,
    wav_path: Path,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Call the TTS service and normalize its raw 24 kHz PCM."""
    request_started = time.perf_counter()
    form = {
        "text": (None, text),
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
                    f"{TTS_PROVIDER} TTS failed ({response.status_code}): {detail}"
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
        raise RuntimeError(f"{TTS_PROVIDER} returned an empty audio stream")
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
        "provider": TTS_PROVIDER,
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


def _adjust_driving_motion(
    motion: dict[str, Any],
    reference: dict[str, Any] | None,
    source_exp: np.ndarray | None = None,
    eye_scale: float = AVATAR_EYE_MOTION_SCALE,
    lip_scale: float = AVATAR_LIP_MOTION_SCALE,
    lip_mode: str = AVATAR_LIP_MOTION_MODE,
    head_scale: float = AVATAR_HEAD_MOTION_SCALE,
) -> dict[str, Any]:
    """Adjust JoyVASA's eyes, mouth and head before FLP applies them.

    With relative motion FLP animates `source + (driving - reference)`.
    Scaling driving values around the reference scales the change that
    reaches the portrait; setting mouth keypoints to
    `driving - source + reference` makes FLP produce JoyVASA's absolute mouth.
    """
    absolute_lips = lip_mode == "absolute" and source_exp is not None
    if (
        reference is None
        or not AVATAR_RELATIVE_MOTION
        or (
            eye_scale == 1.0
            and head_scale == 1.0
            and not absolute_lips
            and lip_scale == 1.0
        )
    ):
        return motion
    reference_exp = np.asarray(reference["exp"])
    exp = np.array(motion["exp"], copy=True)
    eyes, lips = EYE_EXPRESSION_INDICES, LIP_EXPRESSION_INDICES
    exp[:, eyes, :] = reference_exp[:, eyes, :] + eye_scale * (
        exp[:, eyes, :] - reference_exp[:, eyes, :]
    )
    if absolute_lips:
        exp[:, lips, :] = (
            exp[:, lips, :] - np.asarray(source_exp)[:, lips, :]
            + reference_exp[:, lips, :]
        )
    else:
        exp[:, lips, :] = reference_exp[:, lips, :] + lip_scale * (
            exp[:, lips, :] - reference_exp[:, lips, :]
        )
    adjusted = {**motion, "exp": exp}
    if head_scale != 1.0:

        def damp(key: str) -> np.ndarray:
            return reference[key] + head_scale * (motion[key] - reference[key])

        angles = {key: damp(key) for key in ("pitch", "yaw", "roll")}
        adjusted.update(angles, t=damp("t"))
        adjusted["R"] = (
            get_rotation_matrix(angles["pitch"], angles["yaw"], angles["roll"])
            .reshape(np.shape(motion["R"]))
            .astype(np.float32)
        )
    return adjusted


def _dump_motion(wav_path: Path, motion_info: dict[str, Any]) -> None:
    dump_dir = Path(DEBUG_DUMP_DIR)
    dump_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%dT%H%M%S')}-{time.perf_counter_ns() % 10**6:06d}"
    shutil.copyfile(wav_path, dump_dir / f"{stem}.wav")
    motion = motion_info["motion"]
    np.savez_compressed(
        dump_dir / f"{stem}.npz",
        **{key: np.stack([frame[key] for frame in motion]) for key in motion[0]},
        fps=float(motion_info.get("output_fps") or 25.0),
    )


def _render_animation(
    wav_path: Path,
    render_stride: int | None = None,
    reset_motion_reference: bool = True,
    window_callback: Any | None = None,
) -> tuple[list[np.ndarray], float, dict[str, Any]]:
    """Run JoyVASA and FLP frames in memory, without pickle/video/FFmpeg I/O.

    This deliberately keeps phrase-level batching so render stride can be
    measured independently. A later step can append frames incrementally.
    """
    if pipeline is None:
        raise RuntimeError(startup_error or "FasterLivePortrait is not ready")
    if pipeline.is_source_video:
        raise RuntimeError("Direct-memory rendering currently requires a source image")

    selected_stride = render_stride or RENDER_STRIDE
    render_started = time.perf_counter()
    _ensure_joyvasa_pipeline()
    assert pipeline.joyvasa_pipe is not None

    motion_started = time.perf_counter()
    motion_info = pipeline.joyvasa_pipe.gen_motion_sequence(str(wav_path))
    motion_seconds = time.perf_counter() - motion_started
    if DEBUG_DUMP_DIR:
        _dump_motion(wav_path, motion_info)

    source_fps = float(motion_info.get("output_fps") or 25.0)
    playback_fps = source_fps / selected_stride
    motion_list = motion_info["motion"]
    eyes_list = motion_info.get("c_eyes_lst", motion_info.get("c_d_eyes_lst"))
    lips_list = motion_info.get("c_lip_lst", motion_info.get("c_d_lip_lst"))

    frame_loop_started = time.perf_counter()
    frames: list[np.ndarray] = []
    window: list[np.ndarray] = []
    window_index = 0
    window_started = time.perf_counter()
    window_times_ms: list[int] = []
    rendered_count = 0
    expected_rendered_frames = math.ceil(len(motion_list) / selected_stride)
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
        first_frame = reset_motion_reference and frame_index == 0
        # The first frame becomes FLP's reference itself, so it is unscaled.
        motion = _adjust_driving_motion(
            motion,
            None
            if first_frame or pipeline.R_d_0 is None
            else pipeline.x_d_0_info,
            # src_infos[face][0] is FLP's x_s_info for the source portrait.
            pipeline.src_infos[0][0][0]["exp"],
        )
        output = pipeline.run_with_pkl(
            [motion, eyes, lips],
            pipeline.src_imgs[0],
            pipeline.src_infos[0],
            # FasterLivePortrait stores the initial driving rotation and motion
            # reference on first_frame. Reset once per user utterance, not once
            # per phrase, so later chunks remain in the same motion space.
            first_frame=first_frame,
        )
        out_crop = output[0]
        if out_crop is None:
            LOG.warning("Direct renderer returned no face for frame %d", frame_index)
            continue
        frame = _letterbox(cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR))
        rendered_count += 1
        if window_callback is None:
            frames.append(frame)
        else:
            window.append(frame)
            if len(window) >= RENDER_WINDOW_FRAMES:
                window_index += 1
                window_ms = round((time.perf_counter() - window_started) * 1000)
                window_times_ms.append(window_ms)
                window_callback(
                    window,
                    playback_fps,
                    {
                        "index": window_index,
                        "render_ms": window_ms,
                        "expected_rendered_frames": expected_rendered_frames,
                        "motion_frames": len(motion_list),
                    },
                )
                window = []
                window_started = time.perf_counter()

    if window_callback is not None and window:
        window_index += 1
        window_ms = round((time.perf_counter() - window_started) * 1000)
        window_times_ms.append(window_ms)
        window_callback(
            window,
            playback_fps,
            {
                "index": window_index,
                "render_ms": window_ms,
                "expected_rendered_frames": expected_rendered_frames,
                "motion_frames": len(motion_list),
            },
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
        "frames": rendered_count,
        "source_fps": round(source_fps, 3),
        "playback_fps": round(playback_fps, 3),
        "effective_fps": round(
            rendered_count / frame_loop_seconds if frame_loop_seconds > 0 else 0.0,
            3,
        ),
        "motion_reference_reset": reset_motion_reference,
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
        "lip_retargeting": AVATAR_LIP_RETARGETING,
        "incremental_windows": window_callback is not None,
        "window_size": RENDER_WINDOW_FRAMES if window_callback is not None else 0,
        "window_count": window_index,
        "window_max_ms": max(window_times_ms, default=0),
        "window_mean_ms": round(
            sum(window_times_ms) / len(window_times_ms)
            if window_times_ms
            else 0
        ),
    }


def _select_render_stride(
    phrase_index: int,
    buffered_seconds: float,
) -> tuple[int, str]:
    """Select temporal quality from the buffer state at render start."""
    if not ADAPTIVE_RENDER_STRIDE or CATCHUP_RENDER_STRIDE == RENDER_STRIDE:
        return RENDER_STRIDE, "fixed"
    if phrase_index == 1:
        return RENDER_STRIDE, "first-phrase-quality"
    if buffered_seconds < CATCHUP_BUFFER_SECONDS:
        return CATCHUP_RENDER_STRIDE, "low-buffer-catchup"
    return RENDER_STRIDE, "buffer-healthy"


async def _render_phrase_in_windows(
    wav_path: Path,
    selected_stride: int,
    reset_motion_reference: bool,
    pcm: np.ndarray,
    playback: PlaybackBuffer,
    channel: Any,
    phrase_index: int,
    phrase_count: int,
    request_started: float,
    prefetch_callback: Any | None = None,
    prefetch_min_buffer_seconds: float = 0.0,
) -> tuple[float, dict[str, Any], int]:
    """Render on a worker thread and publish completed windows on the event loop."""
    loop = asyncio.get_running_loop()
    windows: asyncio.Queue[tuple[list[np.ndarray], float, dict[str, Any]]] = (
        asyncio.Queue()
    )
    render_started = time.perf_counter()

    def publish_window(
        frames: list[np.ndarray],
        fps: float,
        detail: dict[str, Any],
    ) -> None:
        # Block the worker only until ownership of this small window has moved
        # to the event loop. GPU rendering resumes while WebRTC consumes it.
        asyncio.run_coroutine_threadsafe(
            windows.put((frames, fps, detail)),
            loop,
        ).result()

    render_task = asyncio.create_task(
        asyncio.to_thread(
            _render_animation,
            wav_path,
            selected_stride,
            reset_motion_reference,
            publish_window,
        )
    )
    media_duration: float | None = None
    first_window_ms: int | None = None
    published_windows = 0
    deferred_prefetch_started = False
    deferred_prefetch_buffer_seconds: float | None = None

    try:
        while not render_task.done() or not windows.empty():
            try:
                frames, fps, window_detail = await asyncio.wait_for(
                    windows.get(),
                    timeout=0.05,
                )
            except TimeoutError:
                continue

            if media_duration is None:
                media_duration = playback.begin_phrase_stream(
                    pcm,
                    fps,
                    window_detail["expected_rendered_frames"],
                    render_rate=(
                        render_fps_estimate / fps if SPEECH_START_GATE else None
                    ),
                    window_seconds=RENDER_WINDOW_FRAMES / fps,
                )
                first_window_ms = round(
                    (time.perf_counter() - render_started) * 1000
                )
                playback.append_video_window(frames)
                published_windows += 1
                await _send_event(
                    channel,
                    "playing",
                    message=(
                        f"Playing phrase {phrase_index}/{phrase_count} after "
                        f"the first {len(frames)}-frame window…"
                    ),
                    duration=round(media_duration, 3),
                    first_window_ms=first_window_ms,
                    first_ready_ms=round(
                        (time.perf_counter() - request_started) * 1000
                    ),
                )
            else:
                playback.append_video_window(frames)
                published_windows += 1

            # Adaptive prefetch is deliberately evaluated only after at least
            # one video window has been published. This protects first-frame
            # latency from shared-GPU TTS contention and avoids starting work
            # when the playable A/V buffer is already too shallow.
            if (
                prefetch_callback is not None
                and not deferred_prefetch_started
                and playback.buffered_seconds >= prefetch_min_buffer_seconds
            ):
                deferred_prefetch_buffer_seconds = playback.buffered_seconds
                deferred_prefetch_started = bool(prefetch_callback())

        _unused_frames, _fps, render_detail = await render_task
    except Exception:
        if not render_task.done():
            render_task.cancel()
        await asyncio.gather(render_task, return_exceptions=True)
        raise

    if media_duration is None or first_window_ms is None or published_windows == 0:
        raise RuntimeError("Incremental FLP rendering returned no frame windows")
    playback.finish_phrase_stream()
    render_detail["first_window_ms"] = first_window_ms
    render_detail["published_windows"] = published_windows
    render_detail["deferred_prefetch_started"] = deferred_prefetch_started
    render_detail["deferred_prefetch_buffer_seconds"] = (
        round(deferred_prefetch_buffer_seconds, 3)
        if deferred_prefetch_buffer_seconds is not None
        else None
    )
    return media_duration, render_detail, first_window_ms


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
        "wav_path": wav_path,
        "pcm": pcm,
        "detail": detail,
        "seconds": completed - started,
        "prefetched": prefetched,
    }


async def _create_clip(
    text: str,
    voice_mode: str,
    playback: PlaybackBuffer,
    channel: Any,
) -> None:
    global render_fps_estimate
    if startup_error:
        await _send_event(channel, "error", message=startup_error)
        return
    if playback.busy:
        await _send_event(channel, "error", message="Wait for the current speech to finish.")
        return

    phrases = split_phrases(text, SETTINGS)
    if not phrases:
        await _send_event(channel, "error", message="Enter text to speak.")
        return

    try:
        voice = _resolve_voice_request(voice_mode)
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
        tts_provider=TTS_PROVIDER,
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
                    video_dropped_before = playback.video_dropped
                    speech_holds_before = playback.speech_holds

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
                    next_prefetch_started = False
                    next_prefetch_buffer_seconds: float | None = None

                    def start_next_prefetch() -> bool:
                        nonlocal tts_task
                        nonlocal next_prefetch_started
                        nonlocal next_prefetch_buffer_seconds
                        if (
                            not TTS_PREFETCH
                            or index >= len(phrases)
                            or tts_task is not None
                        ):
                            return False
                        next_prefetch_buffer_seconds = playback.buffered_seconds
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
                        next_prefetch_started = True
                        return True

                    if TTS_PREFETCH and TTS_PREFETCH_POLICY == "eager":
                        start_next_prefetch()
                    buffer_before_render = playback.buffered_seconds
                    selected_stride, stride_reason = _select_render_stride(
                        index,
                        buffer_before_render,
                    )
                    started = time.perf_counter()
                    incremental = INCREMENTAL_FRAME_WINDOWS
                    if incremental:
                        media_duration, render_detail, first_window_ms = (
                            await _render_phrase_in_windows(
                                wav_path,
                                selected_stride,
                                index == 1 or not PERSISTENT_PHRASE_MOTION,
                                pcm,
                                playback,
                                channel,
                                index,
                                len(phrases),
                                request_started,
                                (
                                    start_next_prefetch
                                    if TTS_PREFETCH
                                    and TTS_PREFETCH_POLICY == "adaptive"
                                    and index < len(phrases)
                                    else None
                                ),
                                TTS_PREFETCH_MIN_BUFFER_SECONDS,
                            )
                        )
                        first_ready_ms = (
                            round((started - request_started) * 1000)
                            + first_window_ms
                            if index == 1
                            else None
                        )
                    else:
                        frames, fps, render_detail = await asyncio.to_thread(
                            _render_animation,
                            wav_path,
                            selected_stride,
                            index == 1 or not PERSISTENT_PHRASE_MOTION,
                        )
                        media_duration = playback.append(frames, fps, pcm)
                        if (
                            TTS_PREFETCH
                            and TTS_PREFETCH_POLICY == "adaptive"
                            and playback.buffered_seconds
                            >= TTS_PREFETCH_MIN_BUFFER_SECONDS
                        ):
                            start_next_prefetch()
                        first_ready_ms = (
                            round((time.perf_counter() - request_started) * 1000)
                            if index == 1
                            else None
                        )
                    render_detail["adaptive_stride"] = ADAPTIVE_RENDER_STRIDE
                    render_detail["stride_reason"] = stride_reason
                    render_detail["buffer_before_render_seconds"] = round(
                        buffer_before_render,
                        3,
                    )
                    render_detail["next_prefetch_started"] = next_prefetch_started
                    render_detail["next_prefetch_buffer_seconds"] = (
                        round(next_prefetch_buffer_seconds, 3)
                        if next_prefetch_buffer_seconds is not None
                        else None
                    )
                    render_seconds = time.perf_counter() - started
                    total_render += render_seconds
                    if render_detail.get("effective_fps", 0) > 0:
                        render_fps_estimate = (
                            0.5 * render_fps_estimate
                            + 0.5 * render_detail["effective_fps"]
                        )
                    underrun_ms = (
                        playback.audio_underruns - underruns_before
                    ) * AUDIO_SAMPLES / AUDIO_RATE * 1000
                    video_underrun_ms = (
                        playback.video_underruns - video_underruns_before
                    ) / VIDEO_FPS * 1000
                    video_dropped_ms = (
                        playback.video_dropped - video_dropped_before
                    ) / VIDEO_FPS * 1000
                    speech_hold_ms = (
                        playback.speech_holds - speech_holds_before
                    ) * AUDIO_SAMPLES / AUDIO_RATE * 1000

                    LOG.info(
                        "Phrase %d/%d timings: voice=%s tts=%.3fs tts_wait=%dms "
                        "tts_overlap=%dms prefetched=%s first_byte=%dms download=%dms "
                        "render=%.3fs backend=%s stride=%d adaptive=%s reason=%s "
                        "buffer_before=%.3fs prefetch_policy=%s next_prefetch=%s "
                        "prefetch_buffer=%.3fs motion=%dms "
                        "motion_reference_reset=%s persistent_motion=%s "
                        "frame_loop=%dms effective_fps=%.2f windows=%d first_window=%dms "
                        "pipeline=%dms "
                        "decode=%dms frames=%d/%d playback_fps=%.2f media=%.3fs "
                        "buffer=%.3fs underrun=%.0fms",
                        index,
                        len(phrases),
                        voice.mode,
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
                        TTS_PREFETCH_POLICY if TTS_PREFETCH else "off",
                        render_detail["next_prefetch_started"],
                        render_detail["next_prefetch_buffer_seconds"] or 0.0,
                        render_detail["motion_ms"],
                        render_detail.get("motion_reference_reset", True),
                        render_detail.get("persistent_phrase_motion", False),
                        render_detail["frame_loop_ms"],
                        render_detail["effective_fps"],
                        render_detail.get("window_count", 0),
                        render_detail.get("first_window_ms", 0),
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
                        tts_provider=TTS_PROVIDER,
                        voice_reference_used=voice.reference_path is not None,
                        tts_ms=round(tts_seconds * 1000),
                        tts_wait_ms=tts_wait_ms,
                        tts_overlap_ms=tts_overlap_ms,
                        tts_prefetched=audio_result["prefetched"],
                        tts_prefetch_policy=(
                            TTS_PREFETCH_POLICY if TTS_PREFETCH else "off"
                        ),
                        next_tts_prefetch_started=render_detail[
                            "next_prefetch_started"
                        ],
                        next_tts_prefetch_buffer_seconds=render_detail[
                            "next_prefetch_buffer_seconds"
                        ],
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
                        render_animation_region=render_detail.get(
                            "animation_region", AVATAR_ANIMATION_REGION
                        ),
                        render_driving_multiplier=render_detail.get(
                            "driving_multiplier", AVATAR_DRIVING_MULTIPLIER
                        ),
                        render_normalize_lip=render_detail.get(
                            "normalize_lip", AVATAR_NORMALIZE_LIP
                        ),
                        render_eye_retargeting=render_detail.get(
                            "eye_retargeting", AVATAR_EYE_RETARGETING
                        ),
                        render_lip_retargeting=render_detail.get(
                            "lip_retargeting", AVATAR_LIP_RETARGETING
                        ),
                        render_frame_loop_ms=render_detail["frame_loop_ms"],
                        render_effective_fps=render_detail["effective_fps"],
                        render_incremental_windows=render_detail.get(
                            "incremental_windows", False
                        ),
                        render_window_size=render_detail.get("window_size", 0),
                        render_window_count=render_detail.get("window_count", 0),
                        render_first_window_ms=render_detail.get(
                            "first_window_ms", 0
                        ),
                        render_window_mean_ms=render_detail.get(
                            "window_mean_ms", 0
                        ),
                        render_window_max_ms=render_detail.get(
                            "window_max_ms", 0
                        ),
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
                        video_dropped_ms=round(video_dropped_ms),
                        speech_hold_ms=round(speech_hold_ms),
                        speech_lead_ms=round(playback.speech_lead_seconds * 1000),
                        render_fps_estimate=round(render_fps_estimate, 3),
                        first_ready_ms=first_ready_ms,
                    )

                    if index == 1 and not incremental:
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
            video_underrun_ms=round(
                playback.video_underruns / VIDEO_FPS * 1000
            ),
            video_dropped_ms=round(playback.video_dropped / VIDEO_FPS * 1000),
            speech_hold_ms=round(
                playback.speech_holds * AUDIO_SAMPLES / AUDIO_RATE * 1000
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
    """Wait for the TTS service before startup warm-up."""
    started = time.perf_counter()
    deadline = started + TTS_STARTUP_WAIT_SECONDS
    last_problem = "no response"
    announced = False

    while True:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{TTS_URL}{TTS_HEALTH_PATH}")
            if response.status_code < 500:
                waited = time.perf_counter() - started
                if announced:
                    LOG.info("%s became ready after %.3fs", TTS_PROVIDER, waited)
                return waited
            last_problem = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last_problem = f"{type(exc).__name__}: {exc}"

        now = time.perf_counter()
        if now >= deadline:
            raise TimeoutError(
                f"{TTS_PROVIDER} was not ready at {TTS_URL} after "
                f"{TTS_STARTUP_WAIT_SECONDS:.1f}s ({last_problem})"
            )
        if not announced:
            LOG.info(
                "Waiting up to %.1fs for %s at %s before startup warm-up",
                TTS_STARTUP_WAIT_SECONDS,
                TTS_PROVIDER,
                TTS_URL,
            )
            announced = True
        await asyncio.sleep(min(TTS_STARTUP_POLL_SECONDS, deadline - now))


async def _warmup_pipeline() -> None:
    """Exercise the same TTS, JoyVASA and renderer path before the first user."""
    global warmup_complete, warmup_seconds, warmup_metrics, warmup_error
    global tts_startup_wait_seconds
    global render_fps_estimate

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
                warmup_voice = _resolve_voice_request(TTS_DEFAULT_VOICE_MODE)
            except ValueError as exc:
                LOG.warning(
                    "Default voice mode is unavailable during warm-up (%s); "
                    "warming the built-in voice instead",
                    exc,
                )
                warmup_voice = _resolve_voice_request(VOICE_MODE_DESIGN)
            pcm, tts_detail = await _synthesize(
                WARMUP_TEXT,
                warmup_voice,
                pcm_path,
                wav_path,
            )
            frames, fps, render_detail = await asyncio.to_thread(
                _render_animation,
                wav_path,
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
                IDLE_FRAME.frame = frames[selected_index].copy()
                IDLE_FRAME.source = "startup-warmup-neural-frame"
                IDLE_FRAME.index = selected_index
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
                "render": render_detail,
                "idle_frame_source": IDLE_FRAME.source,
                "idle_frame_index": IDLE_FRAME.index,
                "media_seconds": round(
                    max(len(pcm) / 24_000, len(frames) / max(fps, 1.0)),
                    3,
                ),
            }
            # A cold CUDA warm-up under-reports speed, so it may only raise the
            # estimate: with TensorRT it removes the first phrase's needless
            # wait for the assumed EXPECTED_RENDER_FPS.
            render_fps_estimate = max(
                render_fps_estimate, render_detail.get("effective_fps", 0.0)
            )
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
    global pipeline, startup_error
    try:
        IDLE_FRAME.frame = _load_avatar()
        LOG.info("Loading FasterLivePortrait and source portrait")
        pipeline = await asyncio.to_thread(_initialize_pipeline)
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
            response = await client.get(f"{TTS_URL}{TTS_HEALTH_PATH}")
            tts_ready = response.status_code < 500
    except httpx.HTTPError:
        pass

    providers = ort.get_available_providers()
    preset_configured, preset_problem = _preset_configuration()
    body = {
        "server_build": SERVER_BUILD,
        **CONFIG.summary(),
        "ok": startup_error is None and tts_ready,
        "avatar_ready": startup_error is None,
        "tts_ready": tts_ready,
        "tts_provider": TTS_PROVIDER,
        "tts_url": TTS_URL,
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
        "startup_warmup_complete": warmup_complete,
        "startup_warmup_seconds": (
            round(warmup_seconds, 3) if warmup_seconds is not None else None
        ),
        "startup_warmup_metrics": warmup_metrics,
        "startup_warmup_error": warmup_error,
        "neural_idle_frame_enabled": USE_NEURAL_IDLE_FRAME,
        "idle_frame_source": IDLE_FRAME.source,
        "idle_frame_index": IDLE_FRAME.index,
        "tts_startup_wait_seconds": (
            round(tts_startup_wait_seconds, 3)
            if tts_startup_wait_seconds is not None
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
        "warping_backend": warping_backend,
        "speech_start_gate": SPEECH_START_GATE,
        "speech_start_safety": SPEECH_START_SAFETY,
        "render_fps_estimate": round(render_fps_estimate, 3),
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
    preset_configured, preset_problem = _preset_configuration()
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
                TTS_PRESET_AUDIO_PATH.name if preset_configured else None
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
            await _create_clip(
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
                await _send_event(channel, "conversation", **state)

    if conductor is not None:
        run_job(speak_queued())
        await conductor.start(PeerOutput())

    async def forward_transcript(event: TranscriptEvent) -> None:
        channel = peer_channel.get("channel")
        if channel is None:
            return
        await _send_event(
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
                    _send_event(
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
