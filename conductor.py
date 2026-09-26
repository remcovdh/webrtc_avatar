"""Conductor: owns the conversation between the user and the avatar.

Runs as its own process. The avatar process connects over a Unix socket
(`conductor_client.ConductorClient`) per browser session, sends transcript
events, its speaking state and push-to-talk, and receives `say` / `show`
commands back.

It decides when the user's turn is over (half-duplex, silence thresholds that
depend on how the sentence ends, push-to-talk, primary speaker), logs every
turn, and asks a `Responder` for the reply: System 1 (M3, `system1.py`) or the
M2 echo. System 2 (M4) plugs in behind the same interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from conversation_protocol import MESSAGE
from listener_protocol import encode_frame, read_frame

LOG = logging.getLogger("conductor")

SOCKET_PATH = os.getenv("CONDUCTOR_SOCKET", "/run/avatar/conductor.sock")
LOG_DIR = Path(os.getenv("CONVERSATION_LOG_DIR", "/workspace/results/conversations"))
# Silence (ms) after which the user's turn is over, by how the text ends.
TURN_SILENCE_MS = int(os.getenv("TURN_SILENCE_MS", "700"))
TURN_END_PUNCTUATION_MS = int(os.getenv("TURN_END_PUNCTUATION_MS", "400"))
TURN_CONNECTIVE_MS = int(os.getenv("TURN_CONNECTIVE_MS", "1200"))
# The listener already waited this long before closing an utterance.
LISTENER_SILENCE_MS = int(os.getenv("LISTENER_MIN_SILENCE_MS", "300"))
# Push-to-talk: how long to wait for the last utterance's final text.
PTT_FINAL_WAIT_MS = int(os.getenv("PTT_FINAL_WAIT_MS", "1500"))
RESPONDER = os.getenv("CONDUCTOR_RESPONDER", "system1").strip().lower()
CONNECTIVES = {
    word.strip().lower()
    for word in os.getenv(
        "TURN_CONNECTIVES",
        "and,but,or,because,so,then,that,if,en,maar,of,omdat,want,dus,dan,dat,als",
    ).split(",")
}


# ---------------------------------------------------------------------------
# Turn-taking (pure, clock injected for tests)
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    text: str
    speaker: str | None
    utterances: list[int]
    ended_by: str  # "silence" | "push_to_talk"
    ignored: list[dict[str, Any]] = field(default_factory=list)
    audio: list[str] = field(default_factory=list)  # opt-in utterance WAVs


def turn_wait_ms(text: str) -> int:
    """Total silence that ends a turn, by how the text so far ends."""
    stripped = text.rstrip()
    if stripped.endswith(("?", ".", "!")):
        return TURN_END_PUNCTUATION_MS
    words = stripped.rstrip(",;:").split()
    if words and words[-1].lower() in CONNECTIVES:
        return TURN_CONNECTIVE_MS
    return TURN_SILENCE_MS


class TurnTaker:
    """Collects final utterances into one user turn and decides when it ends.

    Half-duplex: speech that starts while the avatar talks is its own voice
    (or an interruption, which is backlog) and is ignored. Only the primary
    speaker (the first labelled voice) is answered; push-to-talk overrides both
    the speaker filter and the silence thresholds.
    """

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self.now = now
        self.avatar_speaking = False
        self.primary_speaker: str | None = None
        self.ptt = False
        self._ptt_released_at: float | None = None
        self._open: set[int] = set()
        self._echo: set[int] = set()
        # Utterances begun while push-to-talk was held keep its privileges
        # even when their final text arrives after the release.
        self._ptt_utterances: set[int] = set()
        self._texts: list[str] = []
        self._utterances: list[int] = []
        self._speaker: str | None = None
        self._ignored: list[dict[str, Any]] = []
        self._audio: list[str] = []
        self._deadline: float | None = None

    @property
    def collecting(self) -> bool:
        return bool(self._texts or self._open)

    def on_avatar_speaking(self, speaking: bool) -> None:
        self.avatar_speaking = speaking

    def on_speech_start(self, utterance: int) -> None:
        if self.avatar_speaking:
            self._echo.add(utterance)
            return
        self._open.add(utterance)
        if self.ptt:
            self._ptt_utterances.add(utterance)
        self._deadline = None  # the user is still talking

    def on_final(
        self, utterance: int, text: str, speaker: str | None, audio: str | None = None
    ) -> None:
        self._open.discard(utterance)
        text = text.strip()
        if audio:
            self._audio.append(audio)
        if utterance in self._echo:
            self._echo.discard(utterance)
            self._ignored.append({"utterance": utterance, "text": text, "reason": "avatar_speaking"})
            return
        if not text:
            return
        pushed = self.ptt or utterance in self._ptt_utterances
        self._ptt_utterances.discard(utterance)
        if not pushed and speaker is not None:
            if self.primary_speaker is None:
                self.primary_speaker = speaker
            elif speaker != self.primary_speaker:
                self._ignored.append(
                    {"utterance": utterance, "text": text, "speaker": speaker, "reason": "not_primary"}
                )
                return
        self._texts.append(text)
        self._utterances.append(utterance)
        self._speaker = self._speaker or speaker
        if not pushed:
            wait = max(0, turn_wait_ms(" ".join(self._texts)) - LISTENER_SILENCE_MS)
            self._deadline = self.now() + wait / 1000

    def on_push_to_talk(self, pressed: bool) -> None:
        self.ptt = pressed
        if pressed:
            self._deadline = None
            self._ptt_released_at = None
        else:
            self._ptt_released_at = self.now()

    def poll(self) -> Turn | None:
        """The finished turn, once its end condition holds."""
        now = self.now()
        if self._ptt_released_at is not None:
            waited = (now - self._ptt_released_at) * 1000
            if self._open and waited < PTT_FINAL_WAIT_MS:
                return None
            self._ptt_released_at = None
            return self._take("push_to_talk")
        if self.ptt or self._open or self._deadline is None or now < self._deadline:
            return None
        return self._take("silence")

    def _take(self, ended_by: str) -> Turn | None:
        turn = (
            Turn(
                " ".join(self._texts),
                self._speaker,
                self._utterances,
                ended_by,
                self._ignored,
                self._audio,
            )
            if self._texts
            else None
        )
        self._texts, self._utterances, self._speaker, self._audio = [], [], None, []
        self._ignored, self._deadline, self._open = [], None, set()
        return turn


# ---------------------------------------------------------------------------
# Reply policy (System 1 / System 2 plug in here in M3 / M4)
# ---------------------------------------------------------------------------


class Responder(Protocol):
    def reply(self, turn: Turn, language: str) -> tuple[str | None, dict[str, Any]]:
        """The avatar's reply (or None) and details for the log and the page.
        Blocking is fine: the conductor calls it in a worker thread."""
        ...

    def correct(self, text: str, fields: dict[str, str]) -> None:
        """A correction from the page for a sentence the responder judged."""
        ...


class EchoResponder:
    """M2: prove the loop by repeating what was heard."""

    def reply(self, turn: Turn, language: str) -> tuple[str | None, dict[str, Any]]:
        return f"You said: {turn.text}", {}

    def correct(self, text: str, fields: dict[str, str]) -> None:
        pass


class System1Responder:
    """M3: react with a phrase chosen from System 1's typed decision."""

    def __init__(self) -> None:
        from system1 import LayaSystem1, Reactions

        self.system1 = LayaSystem1()
        self.system1.warm_up()
        self.reactions = Reactions(self.system1.config)

    def reply(self, turn: Turn, language: str) -> tuple[str | None, dict[str, Any]]:
        decision = self.system1.decide(turn.text, language)
        return self.reactions.pick(decision), {
            "decision": decision.to_dict(),
            "options": self.system1.options(),
        }

    def correct(self, text: str, fields: dict[str, str]) -> None:
        self.system1.correct(text, fields)

# ---------------------------------------------------------------------------
# Conversation log (local JSON lines, one per turn)
# ---------------------------------------------------------------------------


class ConversationLog:
    def __init__(self, directory: Path, session_id: str) -> None:
        self.path = directory / f"{time.strftime('%Y%m%d')}-{session_id}.jsonl"
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Session per avatar connection
# ---------------------------------------------------------------------------


class Session:
    def __init__(self, send: Callable[[dict[str, Any]], Any], responder: Responder) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.send, self.responder = send, responder
        self.turns = TurnTaker()
        self.log = ConversationLog(LOG_DIR, self.id)
        self.turn_count = 0
        self.language = "en-US"  # the page's listening language
        self._state = ""

    async def handle(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "transcript":
            event = message.get("event", {})
            if event.get("type") == "speech_start":
                self.turns.on_speech_start(int(event.get("utterance", 0)))
            elif event.get("type") == "final":
                self.turns.on_final(
                    int(event.get("utterance", 0)),
                    str(event.get("text", "")),
                    event.get("speaker"),
                    (event.get("detail") or {}).get("audio"),
                )
        elif kind == "avatar_speaking":
            self.turns.on_avatar_speaking(bool(message.get("speaking")))
        elif kind == "push_to_talk":
            self.turns.on_push_to_talk(bool(message.get("pressed")))
        elif kind == "language":
            self.language = str(message.get("language") or "en-US")
        elif kind == "correction":
            await self._correct(message)
        await self._update_state()

    async def _correct(self, message: dict[str, Any]) -> None:
        text = str(message.get("text", ""))
        fields = {k: str(v) for k, v in (message.get("fields") or {}).items()}
        await asyncio.to_thread(self.responder.correct, text, fields)
        self.log.write({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session": self.id,
            "correction": {"turn": message.get("turn"), "text": text, "fields": fields},
        })
        await self.send({"type": "show", "state": {"corrected": {"turn": message.get("turn"), "fields": fields}}})

    async def tick(self) -> None:
        turn = self.turns.poll()
        if turn is not None:
            await self._respond(turn)
        await self._update_state()

    async def _respond(self, turn: Turn) -> None:
        self.turn_count += 1
        started = time.perf_counter()
        try:
            reply, detail = await asyncio.to_thread(self.responder.reply, turn, self.language)
        except Exception:
            LOG.exception("Responder failed; staying silent this turn")
            reply, detail = None, {"error": "responder failed"}
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session": self.id,
            "turn": self.turn_count,
            "language": self.language,
            **asdict(turn),
            **detail,
            "reply": reply,
            "reply_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        self.log.write(record)
        LOG.info("Turn %d (%s): %r -> %r", self.turn_count, turn.ended_by, turn.text, reply)
        await self.send({"type": "show", "state": {"turn": record}})
        if reply:
            await self.send({"type": "say", "text": reply})

    async def _update_state(self) -> None:
        t = self.turns
        state = (
            "avatar_speaking" if t.avatar_speaking
            else "push_to_talk" if t.ptt
            else "user_speaking" if t.collecting
            else "listening"
        )
        if state != self._state:
            self._state = state
            await self.send(
                {"type": "show", "state": {"turn_state": state, "primary_speaker": t.primary_speaker}}
            )


async def serve(responder_factory: Callable[[], Responder]) -> None:
    path = Path(SOCKET_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        lock = asyncio.Lock()

        async def send(message: dict[str, Any]) -> None:
            async with lock:
                writer.write(encode_frame(MESSAGE, json.dumps(message).encode()))
                await writer.drain()

        session = Session(send, responder_factory())
        LOG.info("Avatar session %s connected; log %s", session.id, session.log.path)
        await send({"type": "show", "state": {"session": session.id, "logging": True}})

        async def ticker() -> None:
            while True:
                await asyncio.sleep(0.05)
                await session.tick()

        ticks = asyncio.create_task(ticker())
        try:
            while True:
                kind, payload = await read_frame(reader)
                if kind == MESSAGE:
                    await session.handle(json.loads(payload))
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            ticks.cancel()
            writer.close()
            LOG.info("Avatar session %s disconnected", session.id)

    server = await asyncio.start_unix_server(handle, path=str(path))
    os.chmod(path, 0o666)
    LOG.info("Conductor listening on %s", path)
    async with server:
        await server.serve_forever()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # One responder for all sessions: its models load once.
    responder = System1Responder() if RESPONDER == "system1" else EchoResponder()
    asyncio.run(serve(lambda: responder))


if __name__ == "__main__":
    main()
