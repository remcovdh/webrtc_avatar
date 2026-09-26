"""Build-time tests for the conductor's turn-taking and data handling."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import conductor
from conductor import EchoResponder, TurnTaker, turn_wait_ms


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TurnWaitTests(unittest.TestCase):
    def test_wait_depends_on_how_the_sentence_ends(self) -> None:
        self.assertEqual(turn_wait_ms("Is it raining?"), 400)
        self.assertEqual(turn_wait_ms("Het regent."), 400)
        self.assertEqual(turn_wait_ms("I was thinking"), 700)
        self.assertEqual(turn_wait_ms("I like it but"), 1200)
        self.assertEqual(turn_wait_ms("ik weet het niet, maar,"), 1200)


class TurnTakerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.turns = TurnTaker(now=self.clock)

    def say(self, utterance: int, text: str, speaker: str | None = "spk0") -> None:
        self.turns.on_speech_start(utterance)
        self.turns.on_final(utterance, text, speaker)

    def test_question_ends_the_turn_after_the_short_threshold(self) -> None:
        self.say(1, "What time is it?")
        # 400 ms threshold minus the listener's 300 ms of silence already waited.
        self.clock.t = 0.09
        self.assertIsNone(self.turns.poll())
        self.clock.t = 0.11
        turn = self.turns.poll()
        self.assertEqual((turn.text, turn.speaker, turn.ended_by), ("What time is it?", "spk0", "silence"))
        self.assertIsNone(self.turns.poll())

    def test_continuing_after_a_connective_merges_into_one_turn(self) -> None:
        self.say(1, "I would like coffee and")
        self.clock.t = 0.8  # below the 1.2 s connective threshold
        self.assertIsNone(self.turns.poll())
        self.turns.on_speech_start(2)  # the user goes on
        self.clock.t = 2.0
        self.assertIsNone(self.turns.poll())
        self.turns.on_final(2, "a croissant.", "spk0")
        self.clock.t = 2.2
        self.assertEqual(self.turns.poll().text, "I would like coffee and a croissant.")

    def test_speech_while_the_avatar_talks_is_ignored(self) -> None:
        self.turns.on_avatar_speaking(True)
        self.say(1, "You said hello")
        self.turns.on_avatar_speaking(False)
        self.clock.t = 5
        self.assertIsNone(self.turns.poll())
        self.say(2, "Hello again.")
        self.clock.t = 6
        turn = self.turns.poll()
        self.assertEqual(turn.text, "Hello again.")
        self.assertEqual(turn.ignored[0]["reason"], "avatar_speaking")

    def test_only_the_primary_speaker_is_answered(self) -> None:
        self.say(1, "Hi there.", "spk0")
        self.clock.t = 1
        self.assertEqual(self.turns.poll().text, "Hi there.")
        self.say(2, "Background talk.", "spk1")
        self.clock.t = 2
        self.assertIsNone(self.turns.poll())
        self.say(3, "Me again.", "spk0")
        self.clock.t = 3
        turn = self.turns.poll()
        self.assertEqual(turn.text, "Me again.")
        self.assertEqual(turn.ignored[0]["reason"], "not_primary")

    def test_push_to_talk_ignores_silence_and_speaker_and_ends_on_release(self) -> None:
        self.say(1, "Hi.", "spk0")
        self.clock.t = 1
        self.turns.poll()
        self.turns.on_push_to_talk(True)
        self.say(2, "My colleague asks.", "spk1")
        self.clock.t = 5
        self.assertIsNone(self.turns.poll())  # no silence threshold while held
        self.turns.on_speech_start(3)
        self.turns.on_push_to_talk(False)
        self.clock.t = 5.5
        self.assertIsNone(self.turns.poll())  # waiting for utterance 3's final
        self.turns.on_final(3, "What is new?", "spk1")
        turn = self.turns.poll()
        self.assertEqual((turn.text, turn.ended_by), ("My colleague asks. What is new?", "push_to_talk"))

    def test_push_to_talk_gives_up_on_a_missing_final(self) -> None:
        self.turns.on_push_to_talk(True)
        self.say(1, "Hello.")
        self.turns.on_speech_start(2)
        self.turns.on_push_to_talk(False)
        self.clock.t = 1.6
        self.assertEqual(self.turns.poll().text, "Hello.")


class DataTests(unittest.TestCase):
    def test_echo_reply_and_forget_selection(self) -> None:
        import forget

        turn = conductor.Turn("Hallo", "spk0", [1], "silence")
        self.assertEqual(EchoResponder().reply(turn, "nl-NL"), ("You said: Hallo", {}))

        folder = Path(tempfile.mkdtemp())
        (folder / "audio").mkdir()
        (folder / "audio" / "a.wav").write_bytes(b"x")
        old = folder / "20260901-aaa.jsonl"
        new = folder / "20261005-bbb.jsonl"
        old.write_text(json.dumps({"audio": ["/workspace/results/conversations/audio/a.wav"]}) + "\n")
        new.write_text("{}\n")
        self.assertEqual(forget.select(folder, None, date(2026, 10, 1), False), [old])
        self.assertEqual(forget.select(folder, "bbb", None, False), [new])
        self.assertEqual(forget.audio_of(old), [folder / "audio" / "a.wav"])


if __name__ == "__main__":
    unittest.main()
