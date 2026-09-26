"""Listening worker: VAD -> streaming ASR -> diarization, behind a Unix socket.

Runs in the `listener` image (a Transformers build with the Nemotron streaming
models). The avatar side talks to it through `listener_client.SocketListener`,
which implements the `listener_protocol.Listener` interface.

- Silero VAD (CPU) splits the incoming 16 kHz audio into utterances.
- Nemotron 3.5 streaming ASR transcribes each utterance while it is spoken,
  starting a short pre-roll before the detected speech start.
- Nemotron-3-Diarization runs continuously over all audio, so speaker identities
  stay stable across utterances; each final utterance gets the speaker that was
  most active during it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from listener_protocol import (
    AUDIO,
    CONTROL,
    EVENT,
    SAMPLE_RATE,
    TranscriptEvent,
    encode_frame,
    read_frame,
)

LOG = logging.getLogger("listener")

SOCKET_PATH = os.getenv("LISTENER_SOCKET", "/run/avatar/listener.sock")
ASR_MODEL = os.getenv("LISTENER_ASR_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
DIAR_MODEL = os.getenv("LISTENER_DIAR_MODEL", "nvidia/Nemotron-3-Diarization")
# Right attention context of the streaming ASR: 0/3/6/13 = 80/320/560/1120 ms.
ASR_LOOKAHEAD = int(os.getenv("LISTENER_ASR_LOOKAHEAD", "3"))
# Expected language until the page chooses one. Automatic detection mixed
# languages on short Dutch utterances, so English is the default.
ASR_LANGUAGE = os.getenv("LISTENER_ASR_LANGUAGE", "en-US")
DIAR_MODE = os.getenv("LISTENER_DIAR_MODE", "very_low_latency")
VAD_THRESHOLD = float(os.getenv("LISTENER_VAD_THRESHOLD", "0.5"))
# Silence that closes an utterance. The conductor's turn-taking (M2) decides on
# top of this when the user's turn is over.
MIN_SILENCE_MS = int(os.getenv("LISTENER_MIN_SILENCE_MS", "500"))
PREROLL_MS = int(os.getenv("LISTENER_PREROLL_MS", "300"))
# Silence appended after an utterance so the ASR's look-ahead can finish the
# last words and punctuation.
FLUSH_MS = int(os.getenv("LISTENER_FLUSH_MS", "700"))
VAD_WINDOW = 512  # Silero VAD's window at 16 kHz


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without models)
# ---------------------------------------------------------------------------


class ChunkAssembler:
    """Cut a stream of samples into the exact chunk sizes a streaming model
    needs: `first` samples for the first chunk, `per` for every later one."""

    def __init__(self, first: int, per: int) -> None:
        self.first, self.per = first, per
        self._buffer = np.zeros(0, dtype=np.float32)
        self.chunks_emitted = 0

    def push(self, samples: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        """Complete chunks, each with whether it is the stream's first."""
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])
        chunks = []
        while True:
            size = self.first if self.chunks_emitted == 0 else self.per
            if len(self._buffer) < size:
                return chunks
            chunks.append((self._buffer[:size], self.chunks_emitted == 0))
            self._buffer = self._buffer[size:]
            self.chunks_emitted += 1

    def flush(self) -> tuple[np.ndarray, bool] | None:
        """The remaining samples zero-padded to one full chunk, or None."""
        if len(self._buffer) == 0:
            return None
        first = self.chunks_emitted == 0
        size = self.first if first else self.per
        chunk = np.pad(self._buffer, (0, size - len(self._buffer)))
        self._buffer = np.zeros(0, dtype=np.float32)
        self.chunks_emitted += 1
        return chunk, first


def fit_frames(features, required: int):
    """Trim or zero-pad mel features [batch, frames, bins] to `required` frames.

    The streaming processor's own sample counts give one frame too many for
    the first chunk in the pinned Transformers build, which the model rejects.
    """
    import torch

    frames = features.shape[1]
    if frames > required:
        return features[:, :required]
    if frames < required:
        return torch.nn.functional.pad(features, (0, 0, 0, required - frames))
    return features


@dataclass
class DiarSpan:
    start: float
    end: float
    probs: np.ndarray  # [frames, speakers]


def speaker_for_span(
    spans: list[DiarSpan], start: float, end: float, min_activity: float = 0.2
) -> str | None:
    """The speaker most active between `start` and `end` seconds, or None when
    nobody is clearly active. A span's frames are spread evenly over its time."""
    total, frames = None, 0
    for span in spans:
        if span.end <= start or span.start >= end or len(span.probs) == 0:
            continue
        times = np.linspace(span.start, span.end, len(span.probs), endpoint=False)
        inside = (times >= start) & (times < end)
        if not inside.any():
            continue
        activity = span.probs[inside].sum(axis=0)
        total = activity if total is None else total + activity
        frames += int(inside.sum())
    if total is None:
        return None
    mean = total / frames
    best = int(np.argmax(mean))
    return f"spk{best}" if mean[best] >= min_activity else None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Models:
    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad
        from transformers import (
            AutoModelForAudioFrameClassification,
            AutoModelForRNNT,
            AutoProcessor,
        )

        started = time.perf_counter()
        self.torch = torch
        self.vad = load_silero_vad()
        self.asr_processor = AutoProcessor.from_pretrained(ASR_MODEL)
        self.asr_processor.set_num_lookahead_tokens(ASR_LOOKAHEAD)
        self.asr = (
            AutoModelForRNNT.from_pretrained(ASR_MODEL, dtype=torch.bfloat16)
            .to("cuda")
            .eval()
        )
        sub = self.asr.config.encoder_config.subsampling_factor
        self.asr_first_frames = 1 + sub * ASR_LOOKAHEAD
        self.asr_per_frames = sub * (ASR_LOOKAHEAD + 1)
        self.diar_processor = AutoProcessor.from_pretrained(DIAR_MODEL)
        self.diar_processor.set_streaming_mode(DIAR_MODE)
        self.diar = (
            AutoModelForAudioFrameClassification.from_pretrained(DIAR_MODEL)
            .to("cuda")
            .eval()
        )
        LOG.info(
            "Listener models ready in %.1fs: ASR latency %d ms, diarization %s "
            "(%d ms), GPU memory %.2f GB",
            time.perf_counter() - started,
            self.asr_processor.streaming_latency_ms,
            DIAR_MODE,
            self.diar_processor.streaming_latency_ms,
            torch.cuda.memory_allocated() / 1e9,
        )


# ---------------------------------------------------------------------------
# Per-connection session
# ---------------------------------------------------------------------------


class Utterance:
    """One ASR stream: audio goes in through `push`, text comes out through
    `emit` callbacks from the ASR thread."""

    def __init__(
        self,
        models: Models,
        index: int,
        start: float,
        emit: Callable[[TranscriptEvent], None],
        language: str = ASR_LANGUAGE,
    ) -> None:
        self.models, self.index, self.start, self.emit = models, index, start, emit
        self.language = language
        self.end = start
        self._audio: queue.Queue[np.ndarray | None] = queue.Queue()
        self.text = ""
        self.done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, samples: np.ndarray) -> None:
        self._audio.put(samples)

    def finish(self, end: float) -> None:
        self.end = end
        self._audio.put(np.zeros(int(SAMPLE_RATE * FLUSH_MS / 1000), np.float32))
        self._audio.put(None)

    def _chunks(self) -> Iterator:
        m = self.models
        processor = m.asr_processor
        assembler = ChunkAssembler(
            processor.num_samples_first_audio_chunk,
            processor.num_samples_per_audio_chunk,
        )

        def features(chunk: np.ndarray, first: bool):
            feats = processor(
                chunk,
                sampling_rate=SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=first,
                language=self.language,
                return_tensors="pt",
            )["input_features"]
            required = m.asr_first_frames if first else m.asr_per_frames
            return fit_frames(feats, required).to("cuda", dtype=m.torch.bfloat16)

        while True:
            samples = self._audio.get()
            if samples is None:
                tail = assembler.flush()
                if tail is not None:
                    yield features(*tail)
                return
            for chunk, first in assembler.push(samples):
                yield features(chunk, first)

    def _run(self) -> None:
        from transformers import TextIteratorStreamer

        m = self.models
        streamer = TextIteratorStreamer(
            m.asr_processor.tokenizer, skip_special_tokens=True, timeout=60
        )
        prompt = m.asr_processor(
            np.zeros(m.asr_processor.num_samples_first_audio_chunk, np.float32),
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=True,
            language=self.language,
            return_tensors="pt",
        )["prompt_ids"].to("cuda")

        def generate() -> None:
            try:
                with m.torch.inference_mode():
                    m.asr.generate(
                        input_features=self._chunks(),
                        prompt_ids=prompt,
                        num_lookahead_tokens=ASR_LOOKAHEAD,
                        streamer=streamer,
                    )
            except Exception:
                LOG.exception("ASR failed for utterance %d", self.index)
                streamer.end()

        worker = threading.Thread(target=generate, daemon=True)
        worker.start()
        for piece in streamer:
            if not piece:
                continue
            self.text += piece
            self.emit(
                TranscriptEvent(
                    "partial", self.index, self.text.strip(), self.start, self.end
                )
            )
        worker.join()
        self.text = self.text.strip()
        self.done.set()


class Session:
    def __init__(
        self, models: Models, emit: Callable[[TranscriptEvent], None]
    ) -> None:
        from silero_vad import VADIterator

        self.models, self.emit = models, emit
        self.language = ASR_LANGUAGE
        self.vad = VADIterator(
            models.vad,
            threshold=VAD_THRESHOLD,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=MIN_SILENCE_MS,
        )
        self.samples_seen = 0
        self._vad_buffer = np.zeros(0, np.float32)
        self._recent: deque[np.ndarray] = deque()
        self._recent_samples = 0
        self.utterance: Utterance | None = None
        self.utterance_count = 0
        self.diar_spans: list[DiarSpan] = []
        self.diar_covered = 0.0
        self._diar_audio: queue.Queue[np.ndarray | None] = queue.Queue()
        self._diar_thread = threading.Thread(target=self._diarize, daemon=True)
        self._diar_thread.start()
        self._finalizers: list[threading.Thread] = []

    # -- audio in ---------------------------------------------------------
    def feed(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._diar_audio.put(samples)
        self._vad_buffer = np.concatenate([self._vad_buffer, samples])
        while len(self._vad_buffer) >= VAD_WINDOW:
            window = self._vad_buffer[:VAD_WINDOW]
            self._vad_buffer = self._vad_buffer[VAD_WINDOW:]
            self._process_window(window)

    def _process_window(self, window: np.ndarray) -> None:
        event = self.vad(self.models.torch.from_numpy(window))
        self.samples_seen += len(window)
        self._remember(window)
        if event and "start" in event and self.utterance is None:
            self._start_utterance(max(0, int(event["start"])))
        elif self.utterance is not None:
            self.utterance.push(window)
        if event and "end" in event and self.utterance is not None:
            self._end_utterance(int(event["end"]) / SAMPLE_RATE)

    def _remember(self, window: np.ndarray) -> None:
        self._recent.append(window)
        self._recent_samples += len(window)
        limit = int(SAMPLE_RATE * (PREROLL_MS + 200) / 1000)
        while self._recent_samples - len(self._recent[0]) > limit:
            self._recent_samples -= len(self._recent.popleft())

    def _start_utterance(self, start_sample: int) -> None:
        self.utterance_count += 1
        preroll_from = max(0, start_sample - int(SAMPLE_RATE * PREROLL_MS / 1000))
        recent = np.concatenate(self._recent) if self._recent else np.zeros(0)
        recent_start = self.samples_seen - len(recent)
        preroll = recent[max(0, preroll_from - recent_start):]
        start = (self.samples_seen - len(preroll)) / SAMPLE_RATE
        self.utterance = Utterance(
            self.models, self.utterance_count, start, self.emit, self.language
        )
        self.utterance.push(preroll)
        self.emit(TranscriptEvent("speech_start", self.utterance_count, start=start))

    def _end_utterance(self, end: float) -> None:
        utterance, self.utterance = self.utterance, None
        assert utterance is not None
        utterance.finish(end)
        finalizer = threading.Thread(
            target=self._finalize, args=(utterance,), daemon=True
        )
        finalizer.start()
        self._finalizers.append(finalizer)

    def _finalize(self, utterance: Utterance) -> None:
        utterance.done.wait(timeout=30)
        deadline = time.monotonic() + 2.0
        while self.diar_covered < utterance.end and time.monotonic() < deadline:
            time.sleep(0.05)
        speaker = speaker_for_span(self.diar_spans, utterance.start, utterance.end)
        self.emit(
            TranscriptEvent(
                "final",
                utterance.index,
                utterance.text,
                utterance.start,
                utterance.end,
                speaker,
            )
        )

    # -- diarization --------------------------------------------------------
    def _diarize(self) -> None:
        m = self.models
        processor = m.diar_processor
        assembler = ChunkAssembler(
            processor.num_samples_first_audio_chunk,
            processor.num_samples_per_audio_chunk,
        )
        cache, position = None, 0
        while True:
            samples = self._diar_audio.get()
            if samples is None:
                return
            for chunk, first in assembler.push(samples):
                inputs = processor(
                    chunk,
                    sampling_rate=SAMPLE_RATE,
                    is_streaming=True,
                    is_first_audio_chunk=first,
                    return_tensors="pt",
                ).to("cuda")
                try:
                    with m.torch.inference_mode():
                        output = m.diar(**inputs, speaker_cache=cache)
                except Exception:
                    LOG.exception("Diarization failed; resetting its cache")
                    cache = None
                    continue
                cache = output.speaker_cache
                probs = output.logits.float().squeeze(0).sigmoid().cpu().numpy()
                start = position / SAMPLE_RATE
                position += len(chunk)
                self.diar_spans.append(DiarSpan(start, position / SAMPLE_RATE, probs))
                self.diar_covered = position / SAMPLE_RATE
                # Keep ~10 minutes of history.
                if len(self.diar_spans) > 1000:
                    del self.diar_spans[:100]

    def set_language(self, language: str) -> None:
        if language not in self.models.asr_processor.prompt_dictionary:
            self.emit(TranscriptEvent("status", text=f"unknown language {language!r}"))
            return
        self.language = language
        self.emit(TranscriptEvent("status", text="language", detail={"language": language}))

    def close(self) -> None:
        self._diar_audio.put(None)
        if self.utterance is not None:
            self.utterance.finish(self.samples_seen / SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Socket server
# ---------------------------------------------------------------------------


async def serve(models: Models) -> None:
    path = Path(SOCKET_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue[TranscriptEvent] = asyncio.Queue()

        def emit(event: TranscriptEvent) -> None:
            loop.call_soon_threadsafe(outbox.put_nowait, event)

        async def send_events() -> None:
            while True:
                event = await outbox.get()
                writer.write(encode_frame(EVENT, event.to_json()))
                await writer.drain()

        session = Session(models, emit)
        sender = asyncio.create_task(send_events())
        emit(TranscriptEvent("status", text="ready"))
        LOG.info("Listener client connected")
        try:
            while True:
                kind, payload = await read_frame(reader)
                if kind == AUDIO:
                    session.feed(payload)
                elif kind == CONTROL:
                    command = json.loads(payload)
                    if command.get("type") == "reset":
                        language = session.language
                        session.close()
                        session = Session(models, emit)
                        session.set_language(language)
                    elif command.get("type") == "language":
                        session.set_language(str(command.get("language", "")))
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            session.close()
            sender.cancel()
            writer.close()
            LOG.info("Listener client disconnected")

    server = await asyncio.start_unix_server(handle, path=str(path))
    os.chmod(path, 0o666)
    LOG.info("Listening on %s", path)
    async with server:
        await server.serve_forever()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(serve(Models()))


if __name__ == "__main__":
    main()
