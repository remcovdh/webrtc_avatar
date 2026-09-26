"""Build-time tests for the avatar side of the conductor: the socket proxy."""

import asyncio
import json
import os
import tempfile
import unittest

from conductor_client import ConductorClient
from conversation_protocol import MESSAGE
from listener_protocol import encode_frame, read_frame


class Output:
    def __init__(self) -> None:
        self.said: list[str] = []
        self.shown: list[dict] = []

    async def say(self, text: str) -> None:
        self.said.append(text)

    async def show(self, state: dict) -> None:
        self.shown.append(state)


class ConductorClientTests(unittest.TestCase):
    def test_state_goes_out_and_commands_come_back(self) -> None:
        async def scenario() -> tuple[list[dict], Output]:
            path = os.path.join(tempfile.mkdtemp(), "conductor.sock")
            received: list[dict] = []

            async def fake_conductor(reader, writer) -> None:
                for _ in range(2):
                    _, payload = await read_frame(reader)
                    received.append(json.loads(payload))
                for command in ({"type": "show", "state": {"turn_state": "listening"}},
                                {"type": "say", "text": "You said: hi"}):
                    writer.write(encode_frame(MESSAGE, json.dumps(command).encode()))
                await writer.drain()
                await asyncio.sleep(0.2)

            server = await asyncio.start_unix_server(fake_conductor, path=path)
            output = Output()
            client = ConductorClient(path, retry_seconds=0.05)
            await client.start(output)
            for _ in range(40):
                if client.connected:
                    break
                await asyncio.sleep(0.05)
            await client.on_push_to_talk(True)
            for _ in range(40):
                if output.said:
                    break
                await asyncio.sleep(0.05)
            await client.close()
            server.close()
            return received, output

        received, output = asyncio.run(scenario())
        # On connect the conductor first learns whether the avatar is talking.
        self.assertEqual(received, [
            {"type": "avatar_speaking", "speaking": False},
            {"type": "push_to_talk", "pressed": True},
        ])
        self.assertEqual(output.said, ["You said: hi"])
        self.assertEqual(output.shown, [{"turn_state": "listening"}])


if __name__ == "__main__":
    unittest.main()
