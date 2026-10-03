"""Face rendering: JoyVASA turns speech into face motion, FasterLivePortrait
(FLP) turns that motion into frames of the portrait.

`Renderer` owns the loaded models and renders one phrase at a time. The
helpers above it are plain functions without model state.
"""

from __future__ import annotations

import argparse
import functools
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Import torch before ONNX Runtime so ORT can reuse the CUDA 12 / cuDNN 9
# libraries shipped with the PyTorch image.
import torch
import onnxruntime as ort
from omegaconf import OmegaConf
import src.pipelines.faster_live_portrait_pipeline as flp_pipeline
from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline
from src.pipelines.joyvasa_audio_to_motion_pipeline import (
    JoyVASAAudio2MotionPipeline,
)
from src.utils.utils import get_rotation_matrix

from avatar.config import AvatarSettings

LOG = logging.getLogger("avatar.renderer")

# Guidance scale of JoyVASA's audio-to-motion diffusion.
JOYVASA_CFG_SCALE = 2.8
# FasterLivePortrait's own "eyes" and "lip" animation-region keypoints.
EYE_EXPRESSION_INDICES = [11, 13, 15, 16, 18]
LIP_EXPRESSION_INDICES = [6, 12, 14, 17, 19, 20]

# JoyVASA's official motion checkpoint stores its configuration as an
# argparse.Namespace. PyTorch 2.6+ blocks that class by default when loading
# weights. Allowlist only this known, inert configuration container while
# retaining weights_only=True for every other checkpoint global.
torch.serialization.add_safe_globals([argparse.Namespace])

_FLP_CROP_IMAGE = flp_pipeline.crop_image


def letterbox(image: np.ndarray, size: int = 512) -> np.ndarray:
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


def load_avatar(path: Path) -> np.ndarray:
    """The portrait, letterboxed: the idle image until the warm-up replaces it."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Mount ./inputs and add inputs/avatar.jpg."
        )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode {path}")
    return letterbox(image)


def crop_source_image(
    img: np.ndarray, pts: np.ndarray, *, crop_rotation: bool = False, **kwargs: Any
) -> dict:
    """FLP's source crop, upright by default and without black borders.

    The face crop (2.3x the face size) reaches past the edges of a tightly
    framed portrait; FLP filled that with black bands. Redo the same warp with
    mirrored edges so the background and hair continue instead.
    """
    kwargs.setdefault("flag_do_rot", crop_rotation)
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


def flp_infer_params(settings: AvatarSettings) -> dict[str, Any]:
    """The FasterLivePortrait inference parameters that follow from the settings."""
    return {
        # The face crop is the output. Pasting it back into the photograph
        # needs a GPU matrix inverse per frame, which failed on a GPU shared
        # with TTS.
        "flag_pasteback": False,
        # Relative motion maps driving deltas onto the source pose. Keeping
        # one reference across the phrases of an utterance is the caller's
        # job (see `Renderer.render`, reset_motion_reference).
        "flag_relative_motion": settings.relative_motion,
        "flag_stitching": True,
        "animation_region": settings.animation_region,
        "driving_multiplier": settings.driving_multiplier,
        "flag_normalize_lip": settings.normalize_lip,
        "flag_eye_retargeting": settings.eye_retargeting,
        "flag_lip_retargeting": settings.lip_retargeting,
        "cfg_scale": JOYVASA_CFG_SCALE,
    }


def adjust_driving_motion(
    motion: dict[str, Any],
    reference: dict[str, Any] | None,
    source_exp: np.ndarray | None = None,
    *,
    eye_scale: float,
    lip_scale: float,
    lip_mode: str,
    head_scale: float,
    relative_motion: bool = True,
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
        or not relative_motion
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


def _dump_motion(dump_dir: Path, wav_path: Path, motion_info: dict[str, Any]) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%dT%H%M%S')}-{time.perf_counter_ns() % 10**6:06d}"
    shutil.copyfile(wav_path, dump_dir / f"{stem}.wav")
    motion = motion_info["motion"]
    np.savez_compressed(
        dump_dir / f"{stem}.npz",
        **{key: np.stack([frame[key] for frame in motion]) for key in motion[0]},
        fps=float(motion_info.get("output_fps") or 25.0),
    )


class Renderer:
    """The loaded models plus what was learned about their speed."""

    def __init__(self, settings: AvatarSettings) -> None:
        self.settings = settings
        self.pipeline: GradioLivePortraitPipeline | None = None
        # "cuda" or "tensorrt-fp16", for warping_spade (the per-frame bottleneck).
        self.warping_backend = "cuda"
        # Rendered frames per second: an assumption until measured.
        self.fps_estimate = settings.expected_render_fps

    def _ready_pipeline(self) -> GradioLivePortraitPipeline:
        if self.pipeline is None:
            raise RuntimeError("FasterLivePortrait is not ready")
        return self.pipeline

    def measured(self, effective_fps: float) -> None:
        """Blend a phrase's measured render speed into the estimate."""
        if effective_fps > 0:
            self.fps_estimate = 0.5 * self.fps_estimate + 0.5 * effective_fps

    def measured_at_warmup(self, effective_fps: float) -> None:
        """A cold warm-up under-reports speed, so it may only raise the
        estimate: with TensorRT it removes the first phrase's needless wait
        for the assumed `expected_render_fps`."""
        self.fps_estimate = max(self.fps_estimate, effective_fps)

    def load(self) -> None:
        """Load FasterLivePortrait and precompute the source portrait."""
        settings = self.settings
        config_path = Path(settings.flp_config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"FasterLivePortrait config missing: {config_path}")

        LOG.info(
            "Runtime: torch=%s torch_cuda=%s onnxruntime=%s providers=%s",
            torch.__version__,
            torch.version.cuda,
            ort.__version__,
            ort.get_available_providers(),
        )

        cfg = OmegaConf.load(config_path)
        for name, value in flp_infer_params(settings).items():
            cfg.infer_params[name] = value
        LOG.info(
            "Visual motion config: region=%s multiplier=%.2f normalize_lip=%s "
            "eye_retargeting=%s lip_retargeting=%s",
            settings.animation_region,
            settings.driving_multiplier,
            settings.normalize_lip,
            settings.eye_retargeting,
            settings.lip_retargeting,
        )

        flp_pipeline.crop_image = functools.partial(
            crop_source_image, crop_rotation=settings.crop_rotation
        )
        loaded = GradioLivePortraitPipeline(cfg=cfg, is_animal=False)
        self._enable_tensorrt_warping(loaded)
        if not loaded.prepare_source(settings.image_path, realtime=False):
            raise ValueError(
                "No face was detected in avatar.jpg. Use a clear, front-facing portrait."
            )
        self.pipeline = loaded

    def _enable_tensorrt_warping(self, loaded: Any) -> None:
        """Swap warping_spade's ORT session for a TensorRT FP16 one if possible.

        The model's inputs and outputs are unchanged, so FLP's predictor keeps
        working; any failure leaves the CUDA session in place.
        """
        self.warping_backend = "cuda"
        if not self.settings.tensorrt:
            return
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            LOG.warning("AVATAR_TENSORRT is set but ONNX Runtime has no TensorRT provider")
            return
        model = loaded.model_dict["warping_spade"]
        model_path = model.kwargs["model_path"]
        cache_dir = self.settings.tensorrt_cache
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        options = {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
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
        self.warping_backend = "tensorrt-fp16"
        LOG.info(
            "warping_spade uses TensorRT FP16 (ready in %.1fs, cache %s)",
            time.perf_counter() - started,
            cache_dir,
        )

    def _ensure_joyvasa_pipeline(self) -> None:
        """Create JoyVASA exactly as FLP's run_audio_driving does."""
        pipeline = self._ready_pipeline()
        if pipeline.joyvasa_pipe is not None:
            return

        pipeline.joyvasa_pipe = JoyVASAAudio2MotionPipeline(
            motion_model_path=pipeline.cfg.joyvasa_models.motion_model_path,
            audio_model_path=pipeline.cfg.joyvasa_models.audio_model_path,
            motion_template_path=pipeline.cfg.joyvasa_models.motion_template_path,
            cfg_mode=pipeline.cfg.infer_params.cfg_mode,
            cfg_scale=pipeline.cfg.infer_params.cfg_scale,
        )

    def render(
        self,
        wav_path: Path,
        render_stride: int | None = None,
        reset_motion_reference: bool = True,
        window_callback: Any | None = None,
    ) -> tuple[list[np.ndarray], float, dict[str, Any]]:
        """Render one phrase: JoyVASA motion from the audio, then FLP frames.

        Without `window_callback` all frames are returned at the end. With it,
        frames are handed over per window of `render_window_frames` while
        rendering continues, and the returned frame list is empty. The
        callback's return value asks for the next window to catch up: when it
        is true (and adaptive stride is on), only every Nth frame of that
        window is rendered and shown N times, N being the catch-up stride
        relative to this phrase's stride. The frame rate of the stream stays
        the same, so playback timing is unaffected.
        `reset_motion_reference` makes this phrase's first frame FLP's motion
        reference; later phrases of the same utterance keep the first one's.
        Returns (frames, playback fps, timing and settings detail).
        """
        settings = self.settings
        pipeline = self._ready_pipeline()
        if pipeline.is_source_video:
            raise RuntimeError("Direct-memory rendering currently requires a source image")

        selected_stride = render_stride or settings.render_stride
        render_started = time.perf_counter()
        self._ensure_joyvasa_pipeline()
        assert pipeline.joyvasa_pipe is not None

        motion_started = time.perf_counter()
        motion_info = pipeline.joyvasa_pipe.gen_motion_sequence(str(wav_path))
        motion_seconds = time.perf_counter() - motion_started
        if settings.debug_dump_dir.strip():
            _dump_motion(Path(settings.debug_dump_dir.strip()), wav_path, motion_info)

        source_fps = float(motion_info.get("output_fps") or 25.0)
        playback_fps = source_fps / selected_stride
        motion_list = motion_info["motion"]
        eyes_list = motion_info.get("c_eyes_lst", motion_info.get("c_d_eyes_lst"))
        lips_list = motion_info.get("c_lip_lst", motion_info.get("c_d_lip_lst"))

        frame_loop_started = time.perf_counter()
        frames: list[np.ndarray] = []
        window: list[np.ndarray] = []
        window_started = time.perf_counter()
        window_times_ms: list[int] = []

        # How many frames one rendered frame stands for while catching up.
        hold_factor = (
            max(1, settings.catchup_render_stride // selected_stride)
            if settings.adaptive_render_stride
            else 1
        )
        catching_up = False
        last_frame: np.ndarray | None = None
        held_count = 0

        def publish_window() -> None:
            """Hand the collected frames to the caller and start a new window."""
            nonlocal window, window_started, catching_up
            window_ms = round((time.perf_counter() - window_started) * 1000)
            window_times_ms.append(window_ms)
            wants_catch_up = window_callback(
                window,
                playback_fps,
                {
                    "index": len(window_times_ms),
                    "render_ms": window_ms,
                    "expected_rendered_frames": expected_rendered_frames,
                    "motion_frames": len(motion_list),
                },
            )
            catching_up = bool(wants_catch_up) and hold_factor > 1
            window = []
            window_started = time.perf_counter()

        def emit(frame: np.ndarray) -> None:
            if window_callback is None:
                frames.append(frame)
            else:
                window.append(frame)
                if len(window) >= settings.render_window_frames:
                    publish_window()

        rendered_count = 0
        expected_rendered_frames = math.ceil(len(motion_list) / selected_stride)
        for frame_index in range(0, len(motion_list), selected_stride):
            if catching_up and last_frame is not None and len(window) % hold_factor:
                # Behind the voice: show the previous frame again instead of
                # rendering this one. A window always starts with a real frame.
                held_count += 1
                emit(last_frame)
                continue
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
            motion = adjust_driving_motion(
                motion,
                None
                if first_frame or pipeline.R_d_0 is None
                else pipeline.x_d_0_info,
                # src_infos[face][0] is FLP's x_s_info for the source portrait.
                pipeline.src_infos[0][0][0]["exp"],
                eye_scale=settings.eye_motion_scale,
                lip_scale=settings.lip_motion_scale,
                lip_mode=settings.lip_motion_mode,
                head_scale=settings.head_motion_scale,
                relative_motion=settings.relative_motion,
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
            last_frame = letterbox(cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR))
            rendered_count += 1
            emit(last_frame)

        if window_callback is not None and window:
            publish_window()

        frame_loop_seconds = time.perf_counter() - frame_loop_started
        total_seconds = time.perf_counter() - render_started
        return frames, playback_fps, {
            "motion_ms": round(motion_seconds * 1000),
            "frame_loop_ms": round(frame_loop_seconds * 1000),
            "pipeline_ms": round(total_seconds * 1000),
            "total_ms": round(total_seconds * 1000),
            "render_stride": selected_stride,
            "motion_frames": len(motion_list),
            "frames": rendered_count + held_count,
            # Frames shown twice (or more) instead of rendered, while catching up.
            "held_frames": held_count,
            "source_fps": round(source_fps, 3),
            "playback_fps": round(playback_fps, 3),
            "effective_fps": round(
                rendered_count / frame_loop_seconds if frame_loop_seconds > 0 else 0.0,
                3,
            ),
            "motion_reference_reset": reset_motion_reference,
            "persistent_phrase_motion": settings.persistent_phrase_motion,
            "relative_motion": settings.relative_motion,
            "animation_region": settings.animation_region,
            "driving_multiplier": settings.driving_multiplier,
            "normalize_lip": settings.normalize_lip,
            "eye_retargeting": settings.eye_retargeting,
            "eye_motion_scale": settings.eye_motion_scale,
            "lip_motion_scale": settings.lip_motion_scale,
            "lip_motion_mode": settings.lip_motion_mode,
            "head_motion_scale": settings.head_motion_scale,
            "lip_retargeting": settings.lip_retargeting,
            "incremental_windows": window_callback is not None,
            "window_size": (
                settings.render_window_frames if window_callback is not None else 0
            ),
            "window_count": len(window_times_ms),
            "window_max_ms": max(window_times_ms, default=0),
            "window_mean_ms": round(
                sum(window_times_ms) / len(window_times_ms)
                if window_times_ms
                else 0
            ),
        }
