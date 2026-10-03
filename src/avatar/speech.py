"""Speaking one text: phrases -> speech -> face frames -> playback.

`Speaker.speak` splits the text into phrases and, for each phrase, synthesizes
the speech, picks a render stride, renders the face and queues both on the
browser's `PlaybackBuffer`. Playback of phrase 1 starts as soon as its first
window of frames exists; later phrases are prepared while earlier ones play.

The renderer and the TTS client are passed in, so this module does not import
the models and can be tested with stand-ins.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from avatar.config import AvatarSettings
from avatar.metrics import log_phrase, phrase_metrics
from avatar.phrases import split_phrases
from avatar.playback import VIDEO_FPS, IdleFrame, PlaybackBuffer
from avatar.tts_client import PROVIDER, VOICE_MODE_DESIGN, VoiceRequest

LOG = logging.getLogger("avatar.speech")

# Spoken once at startup so the first user request finds every model loaded;
# its first frame also becomes the idle image.
WARMUP_TEXT = "Hello."


async def send_event(channel: Any, event_type: str, **payload: Any) -> None:
    """Send one JSON event over the browser's data channel, if it is open."""
    if getattr(channel, "readyState", None) == "open":
        channel.send(json.dumps({"type": event_type, **payload}))


def select_render_stride(
    settings: AvatarSettings,
    phrase_index: int,
    buffered_seconds: float,
) -> tuple[int, str]:
    """Select temporal quality from the buffer state at render start."""
    if (
        not settings.adaptive_render_stride
        or settings.catchup_render_stride == settings.render_stride
    ):
        return settings.render_stride, "fixed"
    if phrase_index == 1:
        return settings.render_stride, "first-phrase-quality"
    if buffered_seconds < settings.catchup_buffer_seconds:
        return settings.catchup_render_stride, "low-buffer-catchup"
    return settings.render_stride, "buffer-healthy"


@dataclass
class WarmupResult:
    """How the start-up warm-up went; reported on /health."""

    complete: bool = False
    seconds: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    tts_wait_seconds: float | None = None


@dataclass
class PhraseAudio:
    """One phrase's synthesized speech."""

    wav_path: Path
    pcm: np.ndarray
    detail: dict[str, Any]
    seconds: float
    prefetched: bool


@dataclass
class _Utterance:
    """Everything shared by the phrases of one `speak` call."""

    phrases: list[str]
    voice: VoiceRequest
    playback: PlaybackBuffer
    channel: Any
    tmp_dir: Path
    request_started: float
    # Speech of the next phrase, when it is already being synthesized.
    next_audio: asyncio.Task[PhraseAudio] | None = None
    total_tts: float = 0.0
    total_render: float = 0.0


class Speaker:
    def __init__(
        self,
        settings: AvatarSettings,
        renderer: Any,
        tts: Any,
        idle: IdleFrame,
    ) -> None:
        self.settings = settings
        self.renderer = renderer
        self.tts = tts
        self.idle = idle
        # One utterance at a time uses the GPU, across all browser sessions.
        self.lock = asyncio.Lock()
        self.warmup = WarmupResult()
        # Set when the models failed to load; every request then gets this.
        self.unavailable: str | None = None
        # How long a phrase takes from the start of rendering to its first
        # window of frames: an assumption until measured. The previous phrase
        # must leave at least this much media buffered, or there is a gap.
        self.phrase_start_seconds = 0.5

    # ------------------------------------------------------------------
    # Speaking
    # ------------------------------------------------------------------

    async def speak(
        self,
        text: str,
        voice_mode: str,
        playback: PlaybackBuffer,
        channel: Any,
    ) -> None:
        """Speak `text` on this browser's playback; reports through events."""
        if self.unavailable:
            await send_event(channel, "error", message=self.unavailable)
            return
        if playback.busy:
            await send_event(channel, "error", message="Wait for the current speech to finish.")
            return

        phrases = split_phrases(text, self.settings)
        if not phrases:
            await send_event(channel, "error", message="Enter text to speak.")
            return

        try:
            voice = self.tts.resolve_voice(voice_mode)
        except ValueError as exc:
            await send_event(channel, "error", message=str(exc))
            return

        playback.begin()
        request_started = time.perf_counter()
        utterance: _Utterance | None = None

        await send_event(
            channel,
            "plan",
            message=(
                f"Prepared {len(phrases)} progressive phrase"
                f"{'s' if len(phrases) != 1 else ''} with {voice.mode} voice"
            ),
            phrases=phrases,
            voice_mode=voice.mode,
            tts_provider=PROVIDER,
        )

        if self.lock.locked():
            await send_event(channel, "status", phase="queued", message="Queued behind another request…")

        try:
            async with self.lock:
                results_root = Path(self.settings.results_root)
                results_root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="avatar-", dir=results_root) as tmp:
                    utterance = _Utterance(
                        phrases, voice, playback, channel, Path(tmp), request_started
                    )
                    for index in range(1, len(phrases) + 1):
                        await self._speak_phrase(utterance, index)

            playback.finish()
            generation_seconds = time.perf_counter() - request_started
            await send_event(
                channel,
                "status",
                phase="draining",
                message="All phrases generated; finishing playback…",
            )
            while playback.busy:
                await asyncio.sleep(0.05)

            total_seconds = time.perf_counter() - request_started
            await send_event(
                channel,
                "summary",
                message="Progressive playback complete",
                chunks=len(phrases),
                tts_ms=round(utterance.total_tts * 1000),
                render_ms=round(utterance.total_render * 1000),
                generation_ms=round(generation_seconds * 1000),
                total_ms=round(total_seconds * 1000),
                **{name: round(value) for name, value in playback.counters_ms().items()},
            )
            await send_event(channel, "ready", message="Ready")
        except asyncio.CancelledError:
            playback.clear()
            raise
        except Exception as exc:
            playback.clear()
            LOG.exception("Avatar generation failed")
            await send_event(channel, "error", message=str(exc))
        finally:
            pending = utterance.next_audio if utterance is not None else None
            if pending is not None:
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def _speak_phrase(self, utterance: _Utterance, index: int) -> None:
        """Synthesize, render and queue phrase `index` (1-based), then report."""
        settings = self.settings
        playback = utterance.playback
        channel = utterance.channel
        count = len(utterance.phrases)
        played_before = playback.counters_ms()

        if utterance.next_audio is None:
            utterance.next_audio = asyncio.create_task(
                self._synthesize_phrase(utterance, index, prefetched=False)
            )
        audio_task, utterance.next_audio = utterance.next_audio, None
        tts_wait_started = time.perf_counter()
        audio = await audio_task
        tts_wait_ms = round((time.perf_counter() - tts_wait_started) * 1000)
        utterance.total_tts += audio.seconds

        await send_event(
            channel,
            "status",
            phase="animation",
            chunk=index,
            chunks=count,
            message=f"Phrase {index}/{count}: rendering facial motion…",
        )
        next_prefetch_started = False
        next_prefetch_buffer_seconds: float | None = None

        def start_next_prefetch() -> bool:
            nonlocal next_prefetch_started, next_prefetch_buffer_seconds
            if (
                not settings.tts_prefetch
                or index >= count
                or utterance.next_audio is not None
            ):
                return False
            next_prefetch_buffer_seconds = playback.buffered_seconds
            utterance.next_audio = asyncio.create_task(
                self._synthesize_phrase(utterance, index + 1, prefetched=True)
            )
            next_prefetch_started = True
            return True

        adaptive_prefetch = (
            settings.tts_prefetch and settings.tts_prefetch_policy == "adaptive"
        )
        if settings.tts_prefetch and settings.tts_prefetch_policy == "eager":
            start_next_prefetch()
        buffer_before_render = playback.buffered_seconds
        selected_stride, stride_reason = select_render_stride(
            settings, index, buffer_before_render
        )
        # FasterLivePortrait stores its motion reference on a phrase's first
        # frame. Reset once per utterance, not once per phrase, so later
        # phrases stay in the same motion space.
        reset_motion_reference = index == 1 or not settings.persistent_phrase_motion
        started = time.perf_counter()
        incremental = settings.incremental_frame_windows
        if incremental:
            media_duration, render_detail, first_window_ms = (
                await self._render_phrase_in_windows(
                    utterance,
                    index,
                    audio,
                    selected_stride,
                    reset_motion_reference,
                    (
                        start_next_prefetch
                        if adaptive_prefetch and index < count
                        else None
                    ),
                )
            )
            first_ready_ms = (
                round((started - utterance.request_started) * 1000) + first_window_ms
                if index == 1
                else None
            )
        else:
            frames, fps, render_detail = await asyncio.to_thread(
                self.renderer.render,
                audio.wav_path,
                selected_stride,
                reset_motion_reference,
            )
            media_duration = playback.append(frames, fps, audio.pcm)
            if (
                adaptive_prefetch
                and playback.buffered_seconds
                >= settings.tts_prefetch_min_buffer_seconds
            ):
                start_next_prefetch()
            first_ready_ms = (
                round((time.perf_counter() - utterance.request_started) * 1000)
                if index == 1
                else None
            )
        render_seconds = time.perf_counter() - started
        utterance.total_render += render_seconds
        self.renderer.measured(render_detail.get("effective_fps", 0))

        played_after = playback.counters_ms()
        metrics = phrase_metrics(
            index=index,
            count=count,
            phrase=utterance.phrases[index - 1],
            voice_mode=utterance.voice.mode,
            tts_seconds=audio.seconds,
            tts_wait_ms=tts_wait_ms,
            tts_prefetched=audio.prefetched,
            tts_detail=audio.detail,
            prefetch_policy=(
                settings.tts_prefetch_policy if settings.tts_prefetch else "off"
            ),
            next_prefetch_started=next_prefetch_started,
            next_prefetch_buffer_seconds=next_prefetch_buffer_seconds,
            render_seconds=render_seconds,
            render_detail=render_detail,
            adaptive_stride=settings.adaptive_render_stride,
            stride_reason=stride_reason,
            buffer_before_seconds=buffer_before_render,
            media_seconds=media_duration,
            buffered_seconds=playback.buffered_seconds,
            played_ms={
                name: played_after[name] - played_before[name] for name in played_after
            },
            speech_lead_seconds=playback.speech_lead_seconds,
            fps_estimate=self.renderer.fps_estimate,
            first_ready_ms=first_ready_ms,
        )
        log_phrase(LOG, metrics)
        await send_event(channel, "metrics", **metrics)

        if index == 1 and not incremental:
            await send_event(
                channel,
                "playing",
                message=f"Playing phrase 1/{count} while preparing the rest…",
                duration=round(media_duration, 3),
                first_ready_ms=first_ready_ms,
            )

    def needs_catch_up(
        self, playback: PlaybackBuffer, fps: float, another_phrase_follows: bool
    ) -> bool:
        """Whether the next window of frames should be rendered in catch-up mode.

        Rendering can drop below real time while the next phrase's speech is
        synthesized on the same GPU. Two things then go wrong unless rendering
        speeds up: the video falls behind the voice within the phrase (frames
        are skipped), and the phrase ends with too little media buffered to
        cover the start of the next one (a gap between the phrases).
        """
        if not self.settings.adaptive_render_stride:
            return False
        window_seconds = self.settings.render_window_frames / fps
        if len(playback.video) / VIDEO_FPS < window_seconds:
            return True  # less than one window ahead of the voice
        return (
            another_phrase_follows
            and playback.buffered_seconds < self.phrase_start_seconds
        )

    async def _synthesize_phrase(
        self, utterance: _Utterance, index: int, prefetched: bool
    ) -> PhraseAudio:
        """Synthesize one phrase in its own directory and retain timing metadata."""
        count = len(utterance.phrases)
        chunk_dir = utterance.tmp_dir / f"chunk-{index:03d}"
        chunk_dir.mkdir(exist_ok=True)
        wav_path = chunk_dir / "speech.wav"
        await send_event(
            utterance.channel,
            "status",
            phase="tts-prefetch" if prefetched else "tts",
            chunk=index,
            chunks=count,
            message=(
                f"Phrase {index}/{count}: pre-generating speech during render…"
                if prefetched
                else f"Phrase {index}/{count}: generating speech…"
            ),
        )
        started = time.perf_counter()
        pcm, detail = await self.tts.synthesize(
            utterance.phrases[index - 1],
            utterance.voice,
            chunk_dir / "speech.pcm",
            wav_path,
        )
        return PhraseAudio(
            wav_path, pcm, detail, time.perf_counter() - started, prefetched
        )

    async def _render_phrase_in_windows(
        self,
        utterance: _Utterance,
        index: int,
        audio: PhraseAudio,
        selected_stride: int,
        reset_motion_reference: bool,
        prefetch_callback: Callable[[], bool] | None,
    ) -> tuple[float, dict[str, Any], int]:
        """Render on a worker thread and publish completed windows on the event loop."""
        settings = self.settings
        playback = utterance.playback
        count = len(utterance.phrases)
        loop = asyncio.get_running_loop()
        windows: asyncio.Queue[
            tuple[list[np.ndarray], float, dict[str, Any], concurrent.futures.Future]
        ] = asyncio.Queue()
        render_started = time.perf_counter()
        another_phrase_follows = index < count
        # Set when this side stops consuming windows (an error or a cancel), so
        # the render thread never waits for an answer that will not come.
        abandoned = threading.Event()

        def publish_window(
            frames: list[np.ndarray],
            fps: float,
            detail: dict[str, Any],
        ) -> bool:
            """Called on the render thread; returns whether to catch up next."""
            if abandoned.is_set():
                return False
            catch_up: concurrent.futures.Future[bool] = concurrent.futures.Future()
            # Block the worker only until this small window is queued on the
            # browser's playback, which is also when the lead over the voice is
            # known. GPU rendering resumes while WebRTC consumes the window.
            asyncio.run_coroutine_threadsafe(
                windows.put((frames, fps, detail, catch_up)),
                loop,
            ).result()
            try:
                return catch_up.result(timeout=2.0)
            except Exception:
                return False

        render_task = asyncio.create_task(
            asyncio.to_thread(
                self.renderer.render,
                audio.wav_path,
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
        catch_up: concurrent.futures.Future | None = None

        try:
            while not render_task.done() or not windows.empty():
                try:
                    frames, fps, window_detail, catch_up = await asyncio.wait_for(
                        windows.get(),
                        timeout=0.05,
                    )
                except TimeoutError:
                    continue

                if media_duration is None:
                    media_duration = playback.begin_phrase_stream(
                        audio.pcm,
                        fps,
                        window_detail["expected_rendered_frames"],
                        render_rate=(
                            self.renderer.fps_estimate / fps
                            if settings.speech_start_gate
                            else None
                        ),
                        window_seconds=settings.render_window_frames / fps,
                    )
                    first_window_ms = round(
                        (time.perf_counter() - render_started) * 1000
                    )
                    playback.append_video_window(frames)
                    catch_up.set_result(
                        self.needs_catch_up(playback, fps, another_phrase_follows)
                    )
                    published_windows += 1
                    await send_event(
                        utterance.channel,
                        "playing",
                        message=(
                            f"Playing phrase {index}/{count} after "
                            f"the first {len(frames)}-frame window…"
                        ),
                        duration=round(media_duration, 3),
                        first_window_ms=first_window_ms,
                        first_ready_ms=round(
                            (time.perf_counter() - utterance.request_started) * 1000
                        ),
                    )
                else:
                    playback.append_video_window(frames)
                    catch_up.set_result(
                        self.needs_catch_up(playback, fps, another_phrase_follows)
                    )
                    published_windows += 1

                # Adaptive prefetch is deliberately evaluated only after at least
                # one video window has been published. This protects first-frame
                # latency from shared-GPU TTS contention and avoids starting work
                # when the playable A/V buffer is already too shallow.
                if (
                    prefetch_callback is not None
                    and not deferred_prefetch_started
                    and playback.buffered_seconds
                    >= settings.tts_prefetch_min_buffer_seconds
                ):
                    deferred_prefetch_buffer_seconds = playback.buffered_seconds
                    deferred_prefetch_started = bool(prefetch_callback())

            _unused_frames, _fps, render_detail = await render_task
        except BaseException:
            abandoned.set()
            unanswered = [catch_up] if catch_up is not None else []
            while not windows.empty():
                unanswered.append(windows.get_nowait()[3])
            for pending in unanswered:
                if not pending.done():
                    pending.set_result(False)
            if not render_task.done():
                render_task.cancel()
            await asyncio.gather(render_task, return_exceptions=True)
            raise

        if media_duration is None or first_window_ms is None or published_windows == 0:
            raise RuntimeError("Incremental FLP rendering returned no frame windows")
        playback.finish_phrase_stream()
        self.phrase_start_seconds = (
            0.5 * self.phrase_start_seconds + 0.5 * first_window_ms / 1000
        )
        render_detail["first_window_ms"] = first_window_ms
        render_detail["published_windows"] = published_windows
        render_detail["deferred_prefetch_started"] = deferred_prefetch_started
        render_detail["deferred_prefetch_buffer_seconds"] = (
            round(deferred_prefetch_buffer_seconds, 3)
            if deferred_prefetch_buffer_seconds is not None
            else None
        )
        return media_duration, render_detail, first_window_ms

    # ------------------------------------------------------------------
    # Start-up
    # ------------------------------------------------------------------

    async def warm_up(self) -> WarmupResult:
        """Exercise the same TTS, JoyVASA and renderer path before the first
        user, and take the idle image from the result. A failure is recorded
        and reported, not raised: the avatar still starts."""
        settings = self.settings
        result = self.warmup = WarmupResult()
        started = time.perf_counter()
        results_root = Path(settings.results_root)
        results_root.mkdir(parents=True, exist_ok=True)
        LOG.info("Startup warm-up begins with %r", WARMUP_TEXT)
        try:
            result.tts_wait_seconds = await self.tts.wait_until_ready()
            with tempfile.TemporaryDirectory(prefix="warmup-", dir=results_root) as tmp:
                warmup_dir = Path(tmp)
                wav_path = warmup_dir / "warmup.wav"
                try:
                    voice = self.tts.resolve_voice(settings.default_voice_mode)
                except ValueError as exc:
                    LOG.warning(
                        "Default voice mode is unavailable during warm-up (%s); "
                        "warming the built-in voice instead",
                        exc,
                    )
                    voice = self.tts.resolve_voice(VOICE_MODE_DESIGN)
                pcm, tts_detail = await self.tts.synthesize(
                    WARMUP_TEXT,
                    voice,
                    warmup_dir / "warmup.pcm",
                    wav_path,
                )
                frames, fps, render_detail = await asyncio.to_thread(
                    self.renderer.render,
                    wav_path,
                )
                if settings.use_neural_idle_frame:
                    if not frames:
                        raise RuntimeError(
                            "Startup warm-up returned no frame for the neural idle image"
                        )
                    selected_index = min(
                        max(settings.warmup_idle_frame_index, 0),
                        len(frames) - 1,
                    )
                    # WebRTC must start in the same aligned, neural render space as
                    # speech. The source photograph stays hidden and is used only
                    # to initialise FasterLivePortrait.
                    self.idle.frame = frames[selected_index].copy()
                    self.idle.source = "startup-warmup-neural-frame"
                    self.idle.index = selected_index
                    LOG.info(
                        "Neural idle frame prepared from warm-up frame %d/%d",
                        selected_index,
                        len(frames),
                    )
                result.seconds = time.perf_counter() - started
                result.metrics = {
                    "tts_startup_wait_ms": round(result.tts_wait_seconds * 1000),
                    "tts": tts_detail,
                    "voice_mode": voice.mode,
                    "render": render_detail,
                    "idle_frame_source": self.idle.source,
                    "idle_frame_index": self.idle.index,
                    "media_seconds": round(
                        max(len(pcm) / 24_000, len(frames) / max(fps, 1.0)),
                        3,
                    ),
                }
                self.renderer.measured_at_warmup(render_detail.get("effective_fps", 0.0))
                result.complete = True
                LOG.info(
                    "Startup warm-up complete: total=%.3fs tts=%dms pipeline=%dms "
                    "frames=%d",
                    result.seconds,
                    tts_detail["total_ms"],
                    render_detail["pipeline_ms"],
                    render_detail["frames"],
                )
        except Exception as exc:
            result.seconds = time.perf_counter() - started
            result.error = str(exc)
            LOG.exception(
                "Startup warm-up failed after %.3fs; continuing without warm-up",
                result.seconds,
            )
        return result
