"""Shared contract between the conductor side and the listening worker.

The `Listener` interface is what the rest of the system codes against. The
listening models run in their own process (their Transformers build clashes
with the avatar image), so `SocketListener` in `listener_client.py` implements
the interface by talking to `listener_worker.py` over a local Unix socket with
the tiny framing below: a 1-byte kind, a 4-byte big-endian length, a payload.
No HTTP, no schema machinery.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

SAMPLE_RATE = 16_000

# Frame kinds.
AUDIO = b"A"  # client -> worker: mono int16 PCM at SAMPLE_RATE
CONTROL = b"C"  # client -> worker: JSON, {"type": "reset"} or
#                  {"type": "language", "language": "en-US" | "nl-NL" | "auto"}
EVENT = b"E"  # worker -> client: JSON TranscriptEvent

_HEADER = struct.Struct(">cI")
MAX_PAYLOAD = 16 * 1024 * 1024


@dataclass
class TranscriptEvent:
    """What the listener reports. `partial` events update the utterance being
    spoken; one `final` event closes it with its speaker label."""

    type: str  # "speech_start" | "partial" | "final" | "status"
    utterance: int = 0
    text: str = ""
    start: float = 0.0  # seconds since the session started
    end: float = 0.0
    speaker: str | None = None  # "spk0", "spk1", ... (final events)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @classmethod
    def from_json(cls, payload: bytes) -> "TranscriptEvent":
        return cls(**json.loads(payload))


class Listener(Protocol):
    """Speech in, transcript events out. Implementations may run in-process
    or behind a proxy; callers cannot tell the difference."""

    async def start(self, on_event: Callable[[TranscriptEvent], Any]) -> None: ...

    async def feed(self, pcm16k: bytes) -> None: ...

    async def reset(self) -> None: ...

    async def set_language(self, language: str) -> None:
        """Expected speech language for the next utterances, e.g. "en-US",
        "nl-NL" or "auto" (automatic detection mixes languages more)."""
        ...

    async def close(self) -> None: ...


def encode_frame(kind: bytes, payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"Frame payload too large: {len(payload)} bytes")
    return _HEADER.pack(kind, len(payload)) + payload


async def read_frame(reader: Any) -> tuple[bytes, bytes]:
    """Read one frame from an asyncio StreamReader."""
    header = await reader.readexactly(_HEADER.size)
    kind, length = _HEADER.unpack(header)
    if length > MAX_PAYLOAD:
        raise ValueError(f"Frame payload too large: {length} bytes")
    return kind, await reader.readexactly(length)
