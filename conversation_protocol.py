"""Contract between the avatar process and the conductor process.

The conductor owns the conversation: it hears transcript events, decides when
the user's turn is over, and tells the avatar what to say. The avatar process
only does WebRTC, listening input and rendering. Two small interfaces keep the
parts swappable:

- `Conductor`: what the avatar calls (transcripts, avatar state, push-to-talk).
- `AvatarOutput`: what the conductor calls (say something, show turn state).

Across processes, `conductor_client.ConductorClient` implements `Conductor`
over a Unix socket and the conductor answers with `AvatarOutput` commands. The
messages are JSON in the `listener_protocol` framing (kind `J`).
"""

from __future__ import annotations

from typing import Any, Protocol

MESSAGE = b"J"  # both directions: one JSON object


class Conductor(Protocol):
    async def start(self, output: "AvatarOutput") -> None: ...

    async def on_transcript(self, event: dict[str, Any]) -> None:
        """A listener event: speech_start / partial / final (see
        `listener_protocol.TranscriptEvent`)."""
        ...

    async def on_avatar_speaking(self, speaking: bool) -> None: ...

    async def on_push_to_talk(self, pressed: bool) -> None: ...

    async def close(self) -> None: ...


class AvatarOutput(Protocol):
    async def say(self, text: str) -> None:
        """Speak `text` (English) through TTS and the renderer."""
        ...

    async def show(self, state: dict[str, Any]) -> None:
        """Pass conversation state to the page (turn state, last turn, ...)."""
        ...
