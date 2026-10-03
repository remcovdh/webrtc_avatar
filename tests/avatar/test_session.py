"""Tests for one browser session: message routing and the speaking queue.

The peer connection is real but never connected; the speaker and conductor are
stand-ins.
"""

import asyncio
import json
import unittest

from avatar.config import load
from avatar.playback import IdleFrame
from avatar.session import MAX_TEXT_LENGTH, Session


class FakeSpeaker:
    def __init__(self, log: list) -> None:
        self.idle = IdleFrame()
        self.log = log

    async def speak(self, text, voice_mode, playback, channel) -> None:
        self.log.append(("speak", text, voice_mode))
        await asyncio.sleep(0)


class FakeConductor:
    def __init__(self, log: list) -> None:
        self.log = log

    def __getattr__(self, name: str):
        async def record(*arguments):
            self.log.append((name, *arguments))
        return record


class FakeChannel:
    readyState = "open"
    label = "avatar-control"

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.handlers: dict = {}

    def send(self, message: str) -> None:
        self.events.append(json.loads(message))

    def on(self, name: str, handler) -> None:
        self.handlers[name] = handler


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.log: list = []
        settings = load({"AVATAR_LISTENER_SOCKET": "", "AVATAR_CONDUCTOR_SOCKET": ""}).settings
        self.closed: list = []
        self.session = Session(settings, FakeSpeaker(self.log), {}, self.closed.append)
        self.session.conductor = FakeConductor(self.log)
        self.channel = FakeChannel()
        self.session._on_datachannel(self.channel)
        self.addAsyncCleanup(self.session.close)

    async def settle(self) -> None:
        for _ in range(5):
            await asyncio.sleep(0)

    async def send(self, **payload) -> None:
        self.channel.handlers["message"](json.dumps(payload))
        await self.settle()

    async def test_ready_is_announced_once_on_an_already_open_channel(self) -> None:
        await self.settle()
        self.channel.handlers["open"]()
        await self.settle()
        self.assertEqual([e["type"] for e in self.channel.events], ["ready"])

    async def test_typed_text_is_spoken_and_the_conductor_knows(self) -> None:
        await self.send(text=" Hello there. ", voice_mode="design")
        self.assertEqual(self.log, [
            ("on_avatar_speaking", True),
            ("speak", "Hello there.", "design"),
            ("on_avatar_speaking", False),
        ])

    async def test_text_without_a_voice_mode_uses_the_default(self) -> None:
        await self.send(text="Hello there.")
        self.assertIn(("speak", "Hello there.", "preset-clone"), self.log)

    async def test_empty_and_too_long_text_are_not_spoken(self) -> None:
        await self.send(text="   ")
        await self.send(text="x" * (MAX_TEXT_LENGTH + 1))
        self.assertEqual(self.log, [])
        self.assertIn("limited to 500", self.channel.events[-1]["message"])

    async def test_control_messages_go_to_the_conductor(self) -> None:
        await self.send(type="push_to_talk", pressed=True)
        await self.send(type="listen_language", language="nl-NL")
        await self.send(type="correction", turn=3, text="Hoi", fields={"intent": "greeting"})
        self.assertEqual(self.log, [
            ("on_push_to_talk", True),
            ("on_language", "nl-NL"),
            ("on_correction", {"turn": 3, "text": "Hoi", "fields": {"intent": "greeting"}}),
        ])

    async def test_conductor_lines_are_spoken_in_order(self) -> None:
        queue = asyncio.create_task(self.session._speak_queued())
        self.addCleanup(queue.cancel)
        await self.session.say("One moment.")
        await self.session.say("")  # nothing to say
        await self.session.say("The answer is here.")
        await asyncio.sleep(0.05)
        spoken = [entry[1] for entry in self.log if entry[0] == "speak"]
        self.assertEqual(spoken, ["One moment.", "The answer is here."])

    async def test_conversation_state_reaches_the_page(self) -> None:
        await self.session.show({"state": "listening"})
        self.assertEqual(self.channel.events[-1], {"type": "conversation", "state": "listening"})

    async def test_without_a_conductor_typed_text_still_works(self) -> None:
        self.session.conductor = None
        await self.send(text="Hello there.")
        await self.send(type="push_to_talk", pressed=True)
        self.assertEqual(self.log, [("speak", "Hello there.", "preset-clone")])


if __name__ == "__main__":
    unittest.main()
