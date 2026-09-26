"""`Listener` implementation that proxies to `listener_worker.py`.

The worker runs in its own image; this thin client speaks the Unix-socket
framing from `listener_protocol`. If the worker is not running, audio is
dropped and the client keeps retrying, so the avatar works without listening.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from listener_protocol import (
    AUDIO,
    CONTROL,
    EVENT,
    TranscriptEvent,
    encode_frame,
    read_frame,
)

LOG = logging.getLogger("listener")


class SocketListener:
    def __init__(self, socket_path: str, retry_seconds: float = 3.0) -> None:
        self.socket_path = socket_path
        self.retry_seconds = retry_seconds
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task | None = None
        self._connect_task: asyncio.Task | None = None
        self._on_event: Callable[[TranscriptEvent], Any] | None = None
        self._closed = False
        self._language: str | None = None

    @property
    def connected(self) -> bool:
        return self._writer is not None

    async def start(self, on_event: Callable[[TranscriptEvent], Any]) -> None:
        self._on_event = on_event
        self._connect_task = asyncio.create_task(self._connect_loop())

    async def _connect_loop(self) -> None:
        while not self._closed:
            try:
                reader, writer = await asyncio.open_unix_connection(self.socket_path)
            except OSError:
                await asyncio.sleep(self.retry_seconds)
                continue
            self._writer = writer
            LOG.info("Connected to the listener at %s", self.socket_path)
            if self._language is not None:
                # A (re)started worker must hear the page's choice again.
                await self._send_language()
            try:
                while True:
                    kind, payload = await read_frame(reader)
                    if kind == EVENT and self._on_event is not None:
                        result = self._on_event(TranscriptEvent.from_json(payload))
                        if asyncio.iscoroutine(result):
                            await result
            except (asyncio.IncompleteReadError, ConnectionError):
                LOG.warning("Listener connection lost; retrying")
            finally:
                self._writer = None
                writer.close()

    async def _send(self, kind: bytes, payload: bytes) -> None:
        writer = self._writer
        if writer is None:
            return
        try:
            writer.write(encode_frame(kind, payload))
            await writer.drain()
        except ConnectionError:
            self._writer = None

    async def feed(self, pcm16k: bytes) -> None:
        await self._send(AUDIO, pcm16k)

    async def set_language(self, language: str) -> None:
        self._language = language
        await self._send_language()

    async def _send_language(self) -> None:
        await self._send(
            CONTROL, json.dumps({"type": "language", "language": self._language}).encode()
        )

    async def reset(self) -> None:
        await self._send(CONTROL, json.dumps({"type": "reset"}).encode())

    async def close(self) -> None:
        self._closed = True
        if self._connect_task is not None:
            self._connect_task.cancel()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
