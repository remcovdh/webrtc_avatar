# Listening avatar plan

Agreed on 2026-09-26. The avatar learns to listen (Dutch and English), reacts
quickly through a "System 1", and answers questions from a personal knowledge
folder through a "System 2". `BACKLOG.md` keeps the ideas for later.

## Decisions

| Area | Decision |
|---|---|
| Languages | Understands Dutch and English; replies in English (Chatterbox Turbo). Dutch replies are in the backlog. |
| Listening | Browser microphone over the existing WebRTC connection -> VAD (CPU) -> Nemotron 3.5 streaming ASR 0.6B -> Nemotron-3-Diarization. |
| Turn-taking | Half-duplex: the microphone is ignored while the avatar speaks. End of turn after 700 ms of silence, 400 ms if the text ends in `?`/`.`, up to 1.2 s after a connective ("and", "maar", ...). Push-to-talk as a fallback. All in config. |
| Diarization | Every utterance gets a speaker label; the avatar only reacts to the primary speaker (first voice in a session). Push-to-talk overrides. |
| System 1 | Laya (`laya-multilingual`, Apache 2.0): typed decisions for intent, emotion and whether System 2 is needed, plus classical CPU topic extraction (spaCy, YAKE, rapidfuzz keyword list). Reacts with pre-written English phrases per class. |
| Feedback | (1) Correct System 1's class in the UI, stored at once; (2) similar utterances follow the correction immediately (nearest-neighbour memory); (3) nightly training. Class descriptions in config take effect at once. |
| Knowledge | A personal Markdown folder (`knowledge/`) with CPU embedding retrieval, behind a swappable `Knowledge` interface. |
| System 2 | A local 3-4B LLM (4-bit, ~3 GB) that only formulates retrieved knowledge into 1-3 spoken English sentences. Nothing relevant retrieved: "I don't know that yet", logged for review. Model chosen by measuring 2-3 candidates. Answers stream into TTS sentence by sentence. |
| Data | Per turn one local JSON line (transcript, speaker, decision, corrections, retrieval, answer). Audio only when opted in. Nothing leaves the VM. `forget` script. |
| Access | HTTPS with a self-signed certificate on port 8443 (VM IP + localhost); HTTP on 8000 stays for local tools. |

## Architecture

Parts are separated at the **code level**, not as REST services. A conductor
process owns the conversation and only knows Python interfaces with plain
dataclasses:

```
Listener   audio frames -> partial/final transcripts with speaker labels
System1    utterance -> Decision(intent, emotion, topic, needs_system2, confidence)
Knowledge  question -> retrieved passages
System2    question + passages -> streamed sentences
Speaker    sentence -> audio (TTS)
Renderer   audio -> video frames (JoyVASA + FasterLivePortrait)
```

Implementations are picked in config. They run in-process by default. Only
where dependencies clash (NeMo listener, Chatterbox TTS, FasterLivePortrait
renderer) does a thin proxy class talk to a worker process over a local Unix
socket, with shared memory for bulk audio/video. Swapping a model means writing
one new class behind the same interface.

## Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Listening: HTTPS, microphone over WebRTC, VAD, streaming ASR, diarization behind the `Listener` interface; live transcript with speaker labels in the page. | Dutch and English speech appears correctly within ~0.5 s; GPU memory and latency measured next to the running avatar. |
| M2 | Conductor + interfaces: conversation logic moves out of `server.py`; turn-taking, push-to-talk, conversation log. The avatar repeats what it heard. | A full loop runs through the interfaces; the text box still works. |
| M3 | System 1: Laya, topic extraction, reaction phrases, correction UI with immediate effect. | Reactions feel right for most sentences; a correction applies to a similar sentence at once. |
| M4 | System 2: knowledge folder, retrieval, LLM comparison, streamed answers, honest "I don't know". | Questions about the Markdown pages get short, correct answers; others get "I don't know". |
| M5 | Nightly learning: LLM review, suggested classes and knowledge gaps, Laya fine-tuning. | A readable morning report and a better System 1. |

Each milestone is its own version with its own explanation and commit.

## Progress

- **M1 done (v2S, 2026-09-26).** The user tested it in the browser: English
  transcripts look good; automatic language detection mixed languages on Dutch,
  so the page now has a language choice (English default, Nederlands,
  Automatic). Measured:
  Nemotron 3.5 ASR + Nemotron-3-Diarization run through Transformers (pinned
  main commit) instead of NeMo, which needed Python 3.13 / CUDA 13. Together
  1.67 GB of GPU memory (the whole GPU went from 6.4 to 8.0 GB). English and
  Dutch test sentences transcribe correctly; partial text follows speech by
  about 1 s; final text arrives ~0.6 s after speech ends (0.5 s of which is
  the VAD's silence); diarization separates two voices and places the switch
  correctly; it costs ~24 ms of GPU per 0.64 s of audio.
