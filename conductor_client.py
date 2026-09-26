"""`Conductor` implementation that proxies to the conductor process.

Speaks JSON messages in the `listener_protocol` framing over a Unix socket and
turns the conductor's commands into `AvatarOutput` calls. If the conductor is
not running, messages are dropped and the client keeps retrying, so the avatar
still works (without conversation) on its own.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from conversation_protocol import MESSAGE, AvatarOutput
from listener_protocol import encode_frame, read_frame

LOG = logging.getLogger("conductor")


class ConductorClient:
    def __init__(self, socket_path: str, retry_seconds: float = 3.0) -> None:
        self.socket_path = socket_path
        self.retry_seconds = retry_seconds
        self._writer: asyncio.StreamWriter | None = None
        self._output: AvatarOutput | None = None
        self._task: asyncio.Task | None = None
        self._closed = False
        self._speaking = False
        self._language: str | None = None

    @property
    def connected(self) -> bool:
        return self._writer is not None

    async def start(self, output: AvatarOutput) -> None:
        self._output = output
        self._task = asyncio.create_task(self._connect_loop())

    async def _connect_loop(self) -> None:
        while not self._closed:
            try:
                reader, writer = await asyncio.open_unix_connection(self.socket_path)
            except OSError:
                await asyncio.sleep(self.retry_seconds)
                continue
            self._writer = writer
            LOG.info("Connected to the conductor at %s", self.socket_path)
            # A (re)started conductor must know whether the avatar is talking.
            await self._send({"type": "avatar_speaking", "speaking": self._speaking})
            if self._language is not None:
                await self._send({"type": "language", "language": self._language})
            try:
                while True:
                    kind, payload = await read_frame(reader)
                    if kind == MESSAGE:
                        await self._dispatch(json.loads(payload))
            except (asyncio.IncompleteReadError, ConnectionError):
                LOG.warning("Conductor connection lost; retrying")
            finally:
                self._writer = None
                writer.close()

    async def _dispatch(self, command: dict[str, Any]) -> None:
        if self._output is None:
            return
        if command.get("type") == "say":
            await self._output.say(str(command.get("text", "")))
        elif command.get("type") == "show":
            await self._output.show(command.get("state", {}))

    async def _send(self, message: dict[str, Any]) -> None:
        writer = self._writer
        if writer is None:
            return
        try:
            writer.write(encode_frame(MESSAGE, json.dumps(message).encode()))
            await writer.drain()
        except ConnectionError:
            self._writer = None

    async def on_transcript(self, event: dict[str, Any]) -> None:
        await self._send({"type": "transcript", "event": event})

    async def on_avatar_speaking(self, speaking: bool) -> None:
        self._speaking = speaking
        await self._send({"type": "avatar_speaking", "speaking": speaking})

    async def on_push_to_talk(self, pressed: bool) -> None:
        await self._send({"type": "push_to_talk", "pressed": pressed})

    async def on_language(self, language: str) -> None:
        self._language = language
        await self._send({"type": "language", "language": language})

    async def on_correction(self, correction: dict[str, Any]) -> None:
        await self._send({"type": "correction", **correction})

    async def close(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
