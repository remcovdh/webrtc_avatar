"""What is reported about one spoken phrase: one dictionary, used for both the
`metrics` event to the page and benchmarks and the timing line in the log."""

from __future__ import annotations

import logging
from typing import Any

# render_detail keys passed on as `render_<key>`.
_RENDER_FIELDS = (
    "render_stride",
    "motion_ms",
    "motion_reference_reset",
    "persistent_phrase_motion",
    "animation_region",
    "driving_multiplier",
    "normalize_lip",
    "eye_retargeting",
    "lip_retargeting",
    "frame_loop_ms",
    "effective_fps",
    "incremental_windows",
    "window_size",
    "window_count",
    "window_mean_ms",
    "window_max_ms",
    "pipeline_ms",
    "motion_frames",
    "frames",
    "held_frames",
    "source_fps",
    "playback_fps",
)


def phrase_metrics(
    *,
    index: int,
    count: int,
    phrase: str,
    voice_mode: str,
    tts_seconds: float,
    tts_wait_ms: int,
    tts_prefetched: bool,
    tts_detail: dict[str, Any],
    prefetch_policy: str,
    next_prefetch_started: bool,
    next_prefetch_buffer_seconds: float | None,
    render_seconds: float,
    render_detail: dict[str, Any],
    adaptive_stride: bool,
    stride_reason: str,
    buffer_before_seconds: float,
    media_seconds: float,
    buffered_seconds: float,
    played_ms: dict[str, float],
    speech_lead_seconds: float,
    fps_estimate: float,
    first_ready_ms: int | None,
) -> dict[str, Any]:
    """The fields of the `metrics` event for one phrase.

    `played_ms` is what went wrong in playback while this phrase was prepared
    (see `PlaybackBuffer.counters_ms`).
    """
    tts_ms = round(tts_seconds * 1000)
    metrics: dict[str, Any] = {
        "chunk": index,
        "chunks": count,
        "phrase": phrase,
        "voice_mode": voice_mode,
        "tts_provider": tts_detail["provider"],
        "voice_reference_used": tts_detail["reference_used"],
        "tts_ms": tts_ms,
        "tts_wait_ms": tts_wait_ms,
        # How much of the synthesis ran while the previous phrase rendered.
        "tts_overlap_ms": max(0, tts_ms - tts_wait_ms),
        "tts_prefetched": tts_prefetched,
        "tts_prefetch_policy": prefetch_policy,
        "next_tts_prefetch_started": next_prefetch_started,
        "next_tts_prefetch_buffer_seconds": (
            round(next_prefetch_buffer_seconds, 3)
            if next_prefetch_buffer_seconds is not None
            else None
        ),
        "tts_headers_ms": tts_detail["headers_ms"],
        "tts_first_byte_ms": tts_detail["first_byte_ms"],
        "tts_download_ms": tts_detail["download_ms"],
        "tts_finalize_ms": tts_detail["finalize_ms"],
        "tts_bytes": tts_detail["bytes"],
        "render_ms": round(render_seconds * 1000),
        "render_adaptive_stride": adaptive_stride,
        "render_stride_reason": stride_reason,
        "render_buffer_before_seconds": round(buffer_before_seconds, 3),
        "render_first_window_ms": render_detail.get("first_window_ms", 0),
        "media_seconds": round(media_seconds, 3),
        "buffered_seconds": round(buffered_seconds, 3),
        "underrun_ms": round(played_ms["underrun_ms"]),
        "video_underrun_ms": round(played_ms["video_underrun_ms"]),
        "video_dropped_ms": round(played_ms["video_dropped_ms"]),
        "speech_hold_ms": round(played_ms["speech_hold_ms"]),
        "speech_lead_ms": round(speech_lead_seconds * 1000),
        "render_fps_estimate": round(fps_estimate, 3),
        "first_ready_ms": first_ready_ms,
    }
    for name in _RENDER_FIELDS:
        key = name if name.startswith("render_") else f"render_{name}"
        metrics[key] = render_detail[name]
    return metrics


def log_phrase(log: logging.Logger, metrics: dict[str, Any]) -> None:
    """One line per phrase with the timings that explain a slow or uneven start."""
    log.info(
        "Phrase %d/%d timings: voice=%s tts=%.3fs tts_wait=%dms "
        "tts_overlap=%dms prefetched=%s first_byte=%dms download=%dms "
        "render=%.3fs stride=%d adaptive=%s reason=%s "
        "buffer_before=%.3fs prefetch_policy=%s next_prefetch=%s "
        "prefetch_buffer=%.3fs motion=%dms "
        "motion_reference_reset=%s persistent_motion=%s "
        "frame_loop=%dms effective_fps=%.2f windows=%d first_window=%dms "
        "pipeline=%dms frames=%d/%d held=%d playback_fps=%.2f media=%.3fs "
        "buffer=%.3fs underrun=%.0fms",
        metrics["chunk"],
        metrics["chunks"],
        metrics["voice_mode"],
        metrics["tts_ms"] / 1000,
        metrics["tts_wait_ms"],
        metrics["tts_overlap_ms"],
        metrics["tts_prefetched"],
        metrics["tts_first_byte_ms"],
        metrics["tts_download_ms"],
        metrics["render_ms"] / 1000,
        metrics["render_stride"],
        metrics["render_adaptive_stride"],
        metrics["render_stride_reason"],
        metrics["render_buffer_before_seconds"],
        metrics["tts_prefetch_policy"],
        metrics["next_tts_prefetch_started"],
        metrics["next_tts_prefetch_buffer_seconds"] or 0.0,
        metrics["render_motion_ms"],
        metrics["render_motion_reference_reset"],
        metrics["render_persistent_phrase_motion"],
        metrics["render_frame_loop_ms"],
        metrics["render_effective_fps"],
        metrics["render_window_count"],
        metrics["render_first_window_ms"],
        metrics["render_pipeline_ms"],
        metrics["render_frames"],
        metrics["render_motion_frames"],
        metrics["render_held_frames"],
        metrics["render_playback_fps"],
        metrics["media_seconds"],
        metrics["buffered_seconds"],
        metrics["underrun_ms"],
    )
