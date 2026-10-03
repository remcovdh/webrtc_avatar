"""The WebRTC audio and video tracks that play a `PlaybackBuffer`."""

from __future__ import annotations

import asyncio
import time
from fractions import Fraction

import numpy as np
from aiortc import AudioStreamTrack, VideoStreamTrack
from av import AudioFrame, VideoFrame

from avatar.playback import AUDIO_RATE, AUDIO_SAMPLES, PlaybackBuffer


def keepalive_amplitude(dbfs: float | None) -> float:
    """Amplitude (16-bit samples) of the keep-alive noise; None switches it off.

    During pauses the audio track sends very quiet noise instead of digital
    silence. Laptop audio chips power the speaker amplifier down after a few
    seconds of silence and clip the first few hundred ms when sound resumes
    ("Great, then we agree." was heard as "then we agree."). -60 dBFS is far
    below anything audible in a room.
    """
    return 0.0 if dbfs is None else 32767 * 10 ** (dbfs / 20)


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


def keepalive_noise(
    samples: np.ndarray, rng: np.random.Generator, amplitude: float
) -> np.ndarray:
    """Replace an all-zero audio chunk with inaudible noise (see
    `keepalive_amplitude`); real audio passes through untouched."""
    if amplitude <= 0 or samples.any():
        return samples
    noise = rng.normal(0.0, amplitude, samples.shape)
    return np.clip(np.round(noise), -32768, 32767).astype(np.int16)


class AvatarAudioTrack(AudioStreamTrack):
    def __init__(self, playback: PlaybackBuffer, keepalive_dbfs: float | None) -> None:
        super().__init__()
        self.playback = playback
        self.keepalive = keepalive_amplitude(keepalive_dbfs)
        self._start: float | None = None
        self._timestamp = 0
        self._rng = np.random.default_rng()

    async def recv(self) -> AudioFrame:
        if self._start is None:
            self._start = time.time()
        else:
            self._timestamp += AUDIO_SAMPLES
            target = self._start + self._timestamp / AUDIO_RATE
            await asyncio.sleep(max(0, target - time.time()))

        frame = AudioFrame.from_ndarray(
            keepalive_noise(self.playback.next_audio(), self._rng, self.keepalive),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = AUDIO_RATE
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, AUDIO_RATE)
        return frame
