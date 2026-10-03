"""One browser connection: WebRTC peer, playback, microphone and messages.

A `Session` is created for every SDP offer. It plays the avatar's audio and
video to the browser, forwards the browser's microphone to the listener, and
passes messages between the page's data channel, the `Speaker` and the
conductor. It is also the conductor's `AvatarOutput`: the conductor calls
`say` and `show` on it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict
from typing import Any, Callable, Coroutine

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioResampler

from avatar.conductor_client import ConductorClient
from avatar.config import AvatarSettings
from avatar.listener_client import SocketListener
from avatar.playback import PlaybackBuffer
from avatar.speech import Speaker, send_event
from avatar.tracks import AvatarAudioTrack, AvatarVideoTrack
from shared.listener_protocol import SAMPLE_RATE as LISTENER_SAMPLE_RATE
from shared.listener_protocol import TranscriptEvent

LOG = logging.getLogger("avatar.session")

MAX_TEXT_LENGTH = 500


class Session:
    def __init__(
        self,
        settings: AvatarSettings,
        speaker: Speaker,
        avatar_config: dict[str, Any],
        on_closed: Callable[["Session"], None],
    ) -> None:
        self.settings = settings
        self.speaker = speaker
        self.on_closed = on_closed
        self.pc = RTCPeerConnection()
        self.playback = PlaybackBuffer(
            speaker.idle, settings.lip_sync_offset_ms, settings.speech_start_safety
        )
        # The page's data channel, once it has connected.
        self.channel: Any = None
        self.jobs: set[asyncio.Task[Any]] = set()
        # Lines the conductor wants spoken, in order (a filler, then the answer).
        self.say_queue: asyncio.Queue[str] = asyncio.Queue()
        # Empty sockets switch the part off: no listening, or typed text only.
        listener_socket = settings.listener_socket.strip()
        conductor_socket = settings.conductor_socket.strip()
        self.listener = SocketListener(listener_socket) if listener_socket else None
        self.conductor = (
            ConductorClient(conductor_socket, avatar_config=avatar_config)
            if conductor_socket
            else None
        )
        self.pc.on("track", self._on_track)
        self.pc.on("connectionstatechange", self._on_connection_state)
        self.pc.on("datachannel", self._on_datachannel)

    async def answer(self, sdp: str, offer_type: str) -> RTCSessionDescription:
        """Accept the browser's offer and return this side's description."""
        if self.conductor is not None:
            self._run(self._speak_queued())
            await self.conductor.start(self)

        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type=offer_type)
        )
        # aiortc gives every sender its own random msid stream, and browsers only
        # lip-sync (via RTCP sender reports) tracks that share one stream.
        stream_id = str(uuid.uuid4())
        for track in (
            AvatarVideoTrack(self.playback),
            AvatarAudioTrack(self.playback, self.settings.audio_keepalive_dbfs),
        ):
            self.pc.addTrack(track)._stream_id = stream_id
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        return self.pc.localDescription

    async def close(self) -> None:
        await self.pc.close()

    def _run(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        """Run a coroutine as a job that is cancelled when the session ends."""
        task = asyncio.create_task(coroutine)
        self.jobs.add(task)
        task.add_done_callback(self.jobs.discard)

    # ------------------------------------------------------------------
    # Speaking
    # ------------------------------------------------------------------

    async def speak(self, text: str, voice_mode: str) -> None:
        """Every utterance of the avatar, so the conductor can ignore the
        avatar's own voice while it talks (half-duplex)."""
        if self.conductor is not None:
            await self.conductor.on_avatar_speaking(True)
        try:
            await self.speaker.speak(text, voice_mode, self.playback, self.channel)
        finally:
            if self.conductor is not None:
                await self.conductor.on_avatar_speaking(False)

    async def _speak_queued(self) -> None:
        """Speak the conductor's lines in order, e.g. a filler then the answer."""
        while True:
            text = await self.say_queue.get()
            while self.playback.busy:  # typed text may still be playing
                await asyncio.sleep(0.05)
            await self.speak(text, self.settings.default_voice_mode)

    # `AvatarOutput` for the conductor (see shared.conversation_protocol).

    async def say(self, text: str) -> None:
        if text:
            await self.say_queue.put(text)

    async def show(self, state: dict[str, Any]) -> None:
        if self.channel is not None:
            await send_event(self.channel, "conversation", **state)

    # ------------------------------------------------------------------
    # Listening
    # ------------------------------------------------------------------

    async def _forward_transcript(self, event: TranscriptEvent) -> None:
        if self.channel is None:
            return
        await send_event(
            self.channel,
            "transcript",
            kind=event.type,
            utterance=event.utterance,
            text=event.text,
            start=round(event.start, 2),
            end=round(event.end, 2),
            speaker=event.speaker,
            # Lets the page (and the M2 conductor) ignore the avatar's own
            # voice picked up by the microphone.
            avatar_speaking=self.playback.busy,
        )
        if self.conductor is not None:
            await self.conductor.on_transcript(asdict(event))

    async def _listen(self, track: Any) -> None:
        """Browser microphone -> 16 kHz mono PCM -> listener."""
        assert self.listener is not None
        await self.listener.start(self._forward_transcript)
        resampler = AudioResampler(
            format="s16", layout="mono", rate=LISTENER_SAMPLE_RATE
        )
        try:
            while True:
                frame = await track.recv()
                for resampled in resampler.resample(frame):
                    await self.listener.feed(resampled.to_ndarray().tobytes())
        except MediaStreamError:
            pass
        finally:
            await self.listener.close()

    # ------------------------------------------------------------------
    # WebRTC events
    # ------------------------------------------------------------------

    def _on_track(self, track: Any) -> None:
        if track.kind != "audio" or self.listener is None:
            return
        LOG.info("Microphone track received; streaming it to the listener")
        self._run(self._listen(track))

    async def _on_connection_state(self) -> None:
        LOG.info("Peer state: %s", self.pc.connectionState)
        if self.pc.connectionState in {"failed", "closed"}:
            for job in tuple(self.jobs):
                job.cancel()
            if self.conductor is not None:
                await self.conductor.close()
            await self.pc.close()
            self.on_closed(self)

    def _on_datachannel(self, channel: Any) -> None:
        LOG.info("Data channel connected: %s", channel.label)
        self.channel = channel
        ready_announced = False

        def announce_ready() -> None:
            """Send ready even when aiortc reports the channel already open."""
            nonlocal ready_announced
            if ready_announced or getattr(channel, "readyState", None) != "open":
                return
            ready_announced = True
            self._run(send_event(channel, "ready", message="Ready"))

        channel.on("open", announce_ready)
        # A remotely-created RTCDataChannel can already be open when the
        # datachannel callback runs, in which case its open event is not seen by
        # this newly attached handler. Check the state on the next loop turn.
        asyncio.get_running_loop().call_soon(announce_ready)
        channel.on("message", self._on_message)

    def _on_message(self, message: Any) -> None:
        """A message from the page: a control message or a text to speak."""
        try:
            payload = json.loads(message) if isinstance(message, str) else {}
        except json.JSONDecodeError:
            payload = {"text": str(message)}

        kind = payload.get("type")
        if kind == "push_to_talk":
            if self.conductor is not None:
                self._run(self.conductor.on_push_to_talk(bool(payload.get("pressed"))))
        elif kind == "correction":
            if self.conductor is not None:
                self._run(self.conductor.on_correction({
                    "turn": payload.get("turn"),
                    "text": str(payload.get("text", "")),
                    "fields": payload.get("fields") or {},
                }))
        elif kind == "listen_language":
            language = str(payload.get("language", ""))
            if self.conductor is not None:
                self._run(self.conductor.on_language(language))
            if self.listener is not None:
                self._run(self.listener.set_language(language))
        else:
            self._speak_typed(payload)

    def _speak_typed(self, payload: dict[str, Any]) -> None:
        text = str(payload.get("text", "")).strip()
        voice_mode = str(
            payload.get("voice_mode", self.settings.default_voice_mode)
        ).strip()
        if not text:
            return
        if len(text) > MAX_TEXT_LENGTH:
            self._run(
                send_event(
                    self.channel,
                    "error",
                    message=f"Text is limited to {MAX_TEXT_LENGTH} characters.",
                )
            )
        else:
            self._run(self.speak(text, voice_mode))
