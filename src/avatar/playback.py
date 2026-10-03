"""Per-browser playback state: queued audio and video on one clock.

Audio is the master clock (see `PlaybackBuffer`). The WebRTC tracks in
`avatar.tracks` pull their next frame from here.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

VIDEO_FPS = 30
AUDIO_RATE = 48_000
AUDIO_SAMPLES = 960  # 20 ms at 48 kHz


class IdleFrame:
    """The image shown while the avatar is not speaking.

    It starts as the letterboxed portrait and is replaced once, by a frame the
    renderer produced during warm-up, so idle and speech share one image space.
    Every playback buffer reads it through this holder and so follows the
    change.
    """

    def __init__(self, frame: np.ndarray | None = None) -> None:
        self.frame = (
            frame if frame is not None else np.zeros((512, 512, 3), dtype=np.uint8)
        )
        self.source = "original-avatar"
        self.index: int | None = None


class PlaybackBuffer:
    """Per-peer playback state consumed by the two WebRTC tracks.

    Audio is the master clock. Every queued video frame carries its time on the
    audio timeline, and the video track shows the frame that matches the audio
    already played. When rendering falls behind real time the late frames are
    skipped rather than shown late, so the mouth never drifts after the voice.
    """

    def __init__(
        self,
        idle: IdleFrame | None = None,
        lip_sync_offset_ms: float = 0.0,
        speech_start_safety: float = 1.15,
    ) -> None:
        self.idle = idle or IdleFrame()
        # Positive shows the mouth later than the voice.
        self.lip_sync_offset_ms = lip_sync_offset_ms
        self.speech_start_safety = speech_start_safety
        self.video: deque[tuple[float, np.ndarray]] = deque()
        self.audio: deque[np.ndarray] = deque()
        self.clear()

    @property
    def busy(self) -> bool:
        return self.producing or bool(self.video or self.audio)

    @property
    def buffered_seconds(self) -> float:
        video_seconds = len(self.video) / VIDEO_FPS
        audio_seconds = len(self.audio) * AUDIO_SAMPLES / AUDIO_RATE
        return min(video_seconds, audio_seconds)

    @property
    def playhead_seconds(self) -> float:
        return self._audio_played_samples / AUDIO_RATE

    def counters_ms(self) -> dict[str, float]:
        """What went wrong so far, in milliseconds of media: audio that was due
        but not there, video that was due but not there, video skipped because
        its moment had passed, and speech deliberately held for the video."""
        chunk_ms = AUDIO_SAMPLES / AUDIO_RATE * 1000
        frame_ms = 1000 / VIDEO_FPS
        return {
            "underrun_ms": self.audio_underruns * chunk_ms,
            "video_underrun_ms": self.video_underruns * frame_ms,
            "video_dropped_ms": self.video_dropped * frame_ms,
            "speech_hold_ms": self.speech_holds * chunk_ms,
        }

    def begin(self) -> None:
        self.clear()
        self.producing = True

    def _queue_audio(self, pcm_48k: np.ndarray, duration: float) -> float:
        """Queue padded audio and return its start time on the audio timeline."""
        start = self._audio_queued_samples / AUDIO_RATE
        required_samples = max(len(pcm_48k), math.ceil(duration * AUDIO_RATE))
        padded = np.pad(pcm_48k, (0, required_samples - len(pcm_48k)))
        for offset in range(0, len(padded), AUDIO_SAMPLES):
            chunk = padded[offset : offset + AUDIO_SAMPLES]
            if len(chunk) < AUDIO_SAMPLES:
                chunk = np.pad(chunk, (0, AUDIO_SAMPLES - len(chunk)))
            self.audio.append(chunk.reshape(1, -1))
            self._audio_queued_samples += AUDIO_SAMPLES
        return start

    def append(self, frames: list[np.ndarray], fps: float, pcm_24k: np.ndarray) -> float:
        if not frames:
            raise ValueError("The animation renderer returned no video frames")
        if fps <= 0:
            fps = 25.0

        pcm_48k = np.repeat(pcm_24k.astype(np.int16, copy=False), 2)
        audio_duration = len(pcm_48k) / AUDIO_RATE
        video_duration = len(frames) / fps
        duration = max(audio_duration, video_duration)

        start = self._queue_audio(pcm_48k, duration)
        target_count = max(1, math.ceil(duration * VIDEO_FPS))
        video_indices = np.minimum(
            (np.arange(target_count) * fps / VIDEO_FPS).astype(int),
            len(frames) - 1,
        )
        self.video.extend(
            (start + target_index / VIDEO_FPS, frames[index])
            for target_index, index in enumerate(video_indices)
        )
        self.started = True
        return duration

    def begin_phrase_stream(
        self,
        pcm_24k: np.ndarray,
        fps: float,
        expected_rendered_frames: int,
        render_rate: float | None = None,
        window_seconds: float = 0.0,
        safety: float | None = None,
    ) -> float:
        """Queue phrase audio and initialise drift-free windowed video timing.

        `render_rate` is media seconds rendered per wall-clock second. Below
        1.0 the phrase's speech is held until enough video exists that the
        rest arrives in time. Frames only become available a whole window
        (`window_seconds`, W) at a time; requiring each window to be complete
        before its first frame is due gives a lead of
        `W + (duration - W) * (1 - rate)` seconds of video.
        """
        if fps <= 0:
            fps = 25.0
        if safety is None:
            safety = self.speech_start_safety
        if expected_rendered_frames <= 0:
            raise ValueError("The incremental renderer expected no video frames")

        pcm_48k = np.repeat(pcm_24k.astype(np.int16, copy=False), 2)
        audio_duration = len(pcm_48k) / AUDIO_RATE
        video_duration = expected_rendered_frames / fps
        duration = max(audio_duration, video_duration)

        self._stream_start = self._queue_audio(pcm_48k, duration)
        self._speech_gate_samples = round(self._stream_start * AUDIO_RATE)
        window = min(window_seconds, duration)
        self.speech_lead_seconds = (
            min(
                duration,
                (window + (duration - window) * (1.0 - render_rate)) * safety,
            )
            if render_rate is not None and 0.0 < render_rate < 1.0
            else 0.0
        )
        self._speech_gate_open = self.speech_lead_seconds <= 0.0
        self._stream_fps = fps
        self._stream_source_frames = 0
        self._stream_video_emitted = 0
        self._stream_video_target = max(1, math.ceil(duration * VIDEO_FPS))
        self._stream_media_duration = duration
        self._stream_last_frame = None
        return duration

    def _stream_time(self, target_index: int) -> float:
        return self._stream_start + target_index / VIDEO_FPS

    def append_video_window(self, frames: list[np.ndarray]) -> None:
        """Append rendered frames while preserving resampling phase per phrase."""
        if not frames:
            return
        if self._stream_fps <= 0 or self._stream_video_target <= 0:
            raise RuntimeError("Phrase stream was not initialised")

        source_start = self._stream_source_frames
        source_end = source_start + len(frames)
        desired = min(
            self._stream_video_target,
            math.ceil(source_end * VIDEO_FPS / self._stream_fps),
        )
        for target_index in range(self._stream_video_emitted, desired):
            source_index = min(
                math.floor(target_index * self._stream_fps / VIDEO_FPS),
                source_end - 1,
            )
            if source_index < source_start:
                frame = self.last_video
            else:
                frame = frames[source_index - source_start]
            self.video.append((self._stream_time(target_index), frame))

        self._stream_source_frames = source_end
        self._stream_video_emitted = desired
        self._stream_last_frame = frames[-1]
        if self._stream_video_emitted / VIDEO_FPS >= self.speech_lead_seconds:
            self._speech_gate_open = True
        # Audio and the first video window become visible atomically from the
        # event loop's perspective, so speech never starts on the idle frame.
        self.started = True

    def finish_phrase_stream(self) -> float:
        """Pad a short video tail with its last frame and close stream state."""
        if self._stream_video_emitted < self._stream_video_target:
            tail_frame = self._stream_last_frame
            if tail_frame is None:
                raise RuntimeError("Incremental renderer produced no video frames")
            self.video.extend(
                (self._stream_time(target_index), tail_frame)
                for target_index in range(
                    self._stream_video_emitted, self._stream_video_target
                )
            )
        duration = self._stream_media_duration
        self._speech_gate_open = True
        self._stream_fps = 0.0
        self._stream_start = 0.0
        self._stream_source_frames = 0
        self._stream_video_emitted = 0
        self._stream_video_target = 0
        self._stream_media_duration = 0.0
        self._stream_last_frame = None
        return duration

    def finish(self) -> None:
        self.producing = False

    def next_video(self) -> np.ndarray:
        # The lip-sync offset shows the mouth later than the voice. While the
        # voice is silent (after a phrase, or during a speech hold) the audio
        # clock stops, so let the video catch up by up to the offset; otherwise
        # the last offset's worth of frames would never become due.
        offset = self.lip_sync_offset_ms / 1000
        silence = self._silence_samples / AUDIO_RATE
        playhead = self.playhead_seconds - offset + min(silence, max(offset, 0.0))
        # Skip frames whose moment in the audio has already passed.
        while len(self.video) > 1 and self.video[1][0] <= playhead:
            self.video.popleft()
            self.video_dropped += 1
        if self.video:
            frame_time, frame = self.video[0]
            if frame_time <= playhead + 1 / VIDEO_FPS:
                self.video.popleft()
                self.last_video = frame
            return self.last_video
        if self.started and (self.producing or self.audio):
            if self.producing:
                self.video_underruns += 1
            return self.last_video
        return self.idle.frame

    def next_audio(self) -> np.ndarray:
        # Incremental rendering queues phrase audio before FLP has completed
        # its first video window. Do not consume that audio until the first
        # window atomically sets `started`; otherwise speech leads the mouth by
        # approximately one first-window render interval.
        if self.started and self.audio:
            if (
                not self._speech_gate_open
                and self._audio_played_samples >= self._speech_gate_samples
            ):
                # Planned hold before a phrase; the video waits on the same
                # clock, so both resume together once enough is rendered.
                self.speech_holds += 1
                self._silence_samples += AUDIO_SAMPLES
                return np.zeros((1, AUDIO_SAMPLES), dtype=np.int16)
            self._audio_played_samples += AUDIO_SAMPLES
            self._silence_samples = 0
            return self.audio.popleft()
        if self.started:
            self._silence_samples += AUDIO_SAMPLES
            if self.producing:
                self.audio_underruns += 1
        return np.zeros((1, AUDIO_SAMPLES), dtype=np.int16)

    def clear(self) -> None:
        self.video.clear()
        self.audio.clear()
        self.producing = False
        self.started = False
        self.last_video = self.idle.frame
        self.audio_underruns = 0
        self.video_underruns = 0
        self.video_dropped = 0
        self.speech_holds = 0
        self.speech_lead_seconds = 0.0
        self._audio_queued_samples = 0
        self._audio_played_samples = 0
        self._silence_samples = 0
        self._speech_gate_samples = 0
        self._speech_gate_open = True
        self._stream_fps = 0.0
        self._stream_start = 0.0
        self._stream_source_frames = 0
        self._stream_video_emitted = 0
        self._stream_video_target = 0
        self._stream_media_duration = 0.0
        self._stream_last_frame = None
