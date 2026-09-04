#!/usr/bin/env python3
"""Regression tests for the headless benchmark WebRTC readiness handshake."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_avatar


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"sdp": "answer-sdp", "type": "answer"}


class FakeHttpClient:
    async def __aenter__(self) -> "FakeHttpClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(self, *_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse()


class FakePeer:
    iceGatheringState = "complete"

    def __init__(self) -> None:
        self.localDescription = SimpleNamespace(sdp="offer-sdp", type="offer")
        self.remote_description: object | None = None

    def addTransceiver(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def createOffer(self) -> object:
        return SimpleNamespace(sdp="offer-sdp", type="offer")

    async def setLocalDescription(self, offer: object) -> None:
        self.localDescription = offer

    async def setRemoteDescription(self, answer: object) -> None:
        self.remote_description = answer


def make_client() -> benchmark_avatar.AvatarBenchmarkClient:
    client = object.__new__(benchmark_avatar.AvatarBenchmarkClient)
    client.base_url = "http://127.0.0.1:8000"
    client.timeout_seconds = 1.0
    client.ready_event_grace_seconds = 0.001
    client.peer = FakePeer()
    client.events = asyncio.Queue()
    client.opened = asyncio.Event()
    client.opened.set()
    client.track_tasks = set()
    return client


class BenchmarkHandshakeTest(unittest.IsolatedAsyncioTestCase):
    async def test_open_channel_without_ready_event_continues(self) -> None:
        client = make_client()
        with patch.object(
            benchmark_avatar.httpx,
            "AsyncClient",
            return_value=FakeHttpClient(),
        ):
            await client.connect([])
        self.assertIsNotNone(client.peer.remote_description)

    async def test_ready_event_is_consumed(self) -> None:
        client = make_client()
        await client.events.put({"type": "ready"})
        with patch.object(
            benchmark_avatar.httpx,
            "AsyncClient",
            return_value=FakeHttpClient(),
        ):
            await client.connect([])
        self.assertTrue(client.events.empty())


if __name__ == "__main__":
    unittest.main()
