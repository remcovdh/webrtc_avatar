"""Build-time tests for the listening worker's model-free logic."""

import asyncio
import unittest

import numpy as np

from listener_protocol import AUDIO, TranscriptEvent, encode_frame, read_frame
from listener_worker import ChunkAssembler, DiarSpan, speaker_for_span


class ChunkAssemblerTests(unittest.TestCase):
    def test_exact_first_and_later_chunk_sizes(self) -> None:
        assembler = ChunkAssembler(first=5, per=3)
        chunks = assembler.push(np.arange(12, dtype=np.float32))
        self.assertEqual([len(c) for c, _ in chunks], [5, 3, 3])
        self.assertEqual([first for _, first in chunks], [True, False, False])
        # One sample left; flushing pads it to a full later chunk.
        tail, first = assembler.flush()
        self.assertFalse(first)
        np.testing.assert_array_equal(tail, [11, 0, 0])
        self.assertIsNone(assembler.flush())

    def test_short_stream_flushes_as_first_chunk(self) -> None:
        assembler = ChunkAssembler(first=5, per=3)
        self.assertEqual(assembler.push(np.ones(2, dtype=np.float32)), [])
        tail, first = assembler.flush()
        self.assertTrue(first)
        self.assertEqual(len(tail), 5)


class SpeakerForSpanTests(unittest.TestCase):
    def setUp(self) -> None:
        a = np.zeros((10, 8)); a[:, 0] = 1.0   # speaker 0 during 0-1 s
        b = np.zeros((10, 8)); b[:, 1] = 0.9   # speaker 1 during 1-2 s
        self.spans = [DiarSpan(0.0, 1.0, a), DiarSpan(1.0, 2.0, b)]

    def test_picks_the_most_active_speaker_in_the_window(self) -> None:
        self.assertEqual(speaker_for_span(self.spans, 0.0, 0.9), "spk0")
        self.assertEqual(speaker_for_span(self.spans, 1.1, 2.0), "spk1")
        self.assertEqual(speaker_for_span(self.spans, 0.2, 1.9), "spk1")

    def test_silence_or_no_coverage_gives_no_speaker(self) -> None:
        silent = [DiarSpan(0.0, 1.0, np.full((10, 8), 0.05))]
        self.assertIsNone(speaker_for_span(silent, 0.0, 1.0))
        self.assertIsNone(speaker_for_span(self.spans, 5.0, 6.0))


class ProtocolTests(unittest.TestCase):
    def test_frame_and_event_round_trip(self) -> None:
        event = TranscriptEvent("final", 3, "Hallo daar", 1.0, 2.5, "spk1")

        async def round_trip() -> tuple[bytes, bytes]:
            reader = asyncio.StreamReader()
            reader.feed_data(encode_frame(AUDIO, event.to_json()))
            reader.feed_eof()
            return await read_frame(reader)

        kind, payload = asyncio.run(round_trip())
        self.assertEqual(kind, AUDIO)
        self.assertEqual(TranscriptEvent.from_json(payload), event)


if __name__ == "__main__":
    unittest.main()
