"""Build-time tests for the avatar side of listening: the socket proxy."""

import asyncio
import os
import tempfile
import unittest

from listener_client import SocketListener
from listener_protocol import (
    AUDIO,
    CONTROL,
    EVENT,
    TranscriptEvent,
    encode_frame,
    read_frame,
)


class SocketListenerTests(unittest.TestCase):
    def test_audio_reaches_the_worker_and_events_come_back(self) -> None:
        async def scenario() -> tuple[list[bytes], list[TranscriptEvent]]:
            path = os.path.join(tempfile.mkdtemp(), "listener.sock")
            received: list[bytes] = []

            async def fake_worker(reader, writer) -> None:
                kind, payload = await read_frame(reader)
                received.append(kind + payload)
                event = TranscriptEvent("final", 1, "hello", 0.0, 1.0, "spk0")
                writer.write(encode_frame(EVENT, event.to_json()))
                await writer.drain()
                await asyncio.sleep(0.2)
                writer.close()

            server = await asyncio.start_unix_server(fake_worker, path=path)
            events: list[TranscriptEvent] = []
            listener = SocketListener(path, retry_seconds=0.05)
            await listener.start(events.append)
            for _ in range(40):
                if listener.connected:
                    break
                await asyncio.sleep(0.05)
            await listener.feed(b"\x01\x00\x02\x00")
            for _ in range(40):
                if events:
                    break
                await asyncio.sleep(0.05)
            await listener.close()
            server.close()
            return received, events

        received, events = asyncio.run(scenario())
        self.assertEqual(received, [AUDIO + b"\x01\x00\x02\x00"])
        self.assertEqual([(e.type, e.text, e.speaker) for e in events], [("final", "hello", "spk0")])

    def test_language_chosen_before_connecting_reaches_the_worker(self) -> None:
        async def scenario() -> list[bytes]:
            path = os.path.join(tempfile.mkdtemp(), "listener.sock")
            received: list[bytes] = []

            async def fake_worker(reader, writer) -> None:
                kind, payload = await read_frame(reader)
                received.append(kind + payload)

            server = await asyncio.start_unix_server(fake_worker, path=path)
            listener = SocketListener(path, retry_seconds=0.05)
            await listener.set_language("nl-NL")  # page chose before the mic
            await listener.start(lambda event: None)
            for _ in range(40):
                if received:
                    break
                await asyncio.sleep(0.05)
            await listener.close()
            server.close()
            return received

        self.assertEqual(
            asyncio.run(scenario()),
            [CONTROL + b'{"type": "language", "language": "nl-NL"}'],
        )

    def test_missing_worker_drops_audio_without_errors(self) -> None:
        async def scenario() -> bool:
            listener = SocketListener("/nonexistent/listener.sock", retry_seconds=0.05)
            await listener.start(lambda event: None)
            await listener.feed(b"\x00\x00")
            connected = listener.connected
            await listener.close()
            return connected

        self.assertFalse(asyncio.run(scenario()))


if __name__ == "__main__":
    unittest.main()
