# Backlog

Ideas and follow-ups to pick up later. Newest decisions first; see
`REALTIME_BASELINE.md` for what already works.

## Listening avatar (planned 2026-09-26)

Decided so far:

- Languages: the avatar **understands** Dutch and English, and **replies in
  English** (current Chatterbox Turbo voice).
- Layer 0: browser microphone over WebRTC -> VAD (CPU) -> Nemotron 3.5
  streaming ASR 0.6B -> Nemotron-3-Diarization.
- System 1: Laya (`laya-multilingual`, Apache 2.0) for typed decisions
  (intent, emotion, needs-System-2), plus classical CPU topic extraction
  (spaCy, YAKE, rapidfuzz keyword matching).
- First version reacts with **spoken reactions only**, from pre-written English
  phrases per class.
- System 2: small LLM + RAG in the background for real answers.
- Turn-taking v1: **half-duplex** (microphone ignored while the avatar speaks,
  only a level meter). End of turn after 700 ms of VAD silence; 400 ms if the
  ASR text ends in `?`/`.`; up to 1.2 s if it ends in a connective ("and",
  "but", "en", "maar", "because", "omdat"...). Thresholds in config.
  **Push-to-talk** button as a fallback that bypasses the VAD. System 1 already
  classifies partial ASR text while the user speaks.
- Diarization v1 (mainly one person at a time): **label every utterance with
  its speaker** in transcripts and training data, and **react only to the
  primary speaker** (the first voice in a session); other voices are logged
  without a reaction. Push-to-talk always overrides.
- Knowledge v1: a **personal Markdown folder** (`knowledge/`), indexed with a
  small multilingual embedding model on the CPU, behind the `Knowledge`
  interface so it can be swapped for a much stronger system later.
- System 2 v1: a **local 3-4B LLM, 4-bit** (~2.5-3.5 GB), in its own worker
  process behind the `System2` interface. It is only a **formulator**: it turns
  retrieved knowledge into 1-3 well-formed spoken English sentences and adds no
  knowledge of its own. If retrieval finds nothing relevant, the avatar says it
  doesn't know (without asking the LLM), and the gap is logged for the nightly
  review. The model is chosen by measuring 2-3 current candidates (time to
  first sentence, English quality, faithfulness to the retrieved text).
  Sentences stream into the existing TTS/render pipeline as they are produced.
- Data: every turn is logged locally as one JSON line in
  `results/conversations/` (transcript, speaker, timestamps, System 1 decision,
  UI corrections, retrieved knowledge, answer). Microphone **audio only when
  opted in** (`LISTENER_SAVE_AUDIO=true`). Everything stays on the VM, including
  the nightly review. A `forget` script deletes a session or everything before a
  date; the UI shows whether logging is on.
- Access: the user opens the page from their own machine, so the server gets
  **HTTPS with a self-signed certificate** (SAN for the VM's IP and localhost),
  generated once into a gitignored folder. HTTPS on port 8443; plain HTTP on
  8000 stays for local tools (health checks, `review_recording.sh`). This is
  closer to running it elsewhere later.
- Feedback in three layers: (1) correct System 1's class in the UI, stored at
  once; (2) similar later utterances follow the correction immediately
  (nearest-neighbour override, no training); (3) nightly training.
- Class descriptions (Laya `criteria`) live in config and take effect at once.
- **Every part is swappable** behind a small, stable interface (ASR, VAD,
  diarization, System 1, System 2/RAG, TTS, avatar renderer), because better
  models appear quickly. The RAG/knowledge part stays architecturally separate.
- Separation is at the **code level**, not REST/HTTP services: the conductor
  only knows Python interfaces (`Listener`, `System1`, `Knowledge`, `System2`,
  `Speaker`, `Renderer`) with plain dataclasses. Parts run in-process by
  default and are chosen in config. Only where dependencies clash (NeMo
  listener, Chatterbox TTS, FLP renderer) does a thin proxy talk to a worker
  process over a local Unix socket / shared memory. The conductor is its own
  process with System 1 and knowledge inside it.

Later:

- **Non-verbal reactions while listening**: nod, smile, raised eyebrows,
  interested look, driven by intent/emotion of what is said so far (< 300 ms).
  Needs facial motion without audio, which JoyVASA does not produce yet.
- **Tone-of-voice emotion**: an audio encoder (e.g. an audio JEPA-style model)
  so System 1 also hears laughter, sighs and anger that the words don't carry.
- **Compare other System 1 models** against Laya: Von (claims sub-15 ms),
  OpenJev (Jev-compatible API), and any others from the Jev wave.
- **Fine-tune Laya** on the corrections collected through the feedback loop.
- **Nightly LLM review**: an LLM reads the day's conversations and corrections,
  flags where a conversation went wrong, proposes new categories and phrases,
  and prepares training data for System 1 and improvements for the RAG.
- **Stronger knowledge systems** behind the same `Knowledge` interface: graph
  RAG, wiki import, existing documents (PDF/Word/Confluence), and eventually
  research agents and vision (the avatar looking at what the user shows).
- **Pre-rendered reaction phrases**: cache TTS audio + rendered motion for the
  fixed phrases so they play instantly.
- **Reply in Dutch** when spoken to in Dutch: Chatterbox Multilingual (already
  in the TTS image, supports `nl`) with the language chosen per sentence
  instead of per container; measure speed and preset-voice quality, possibly
  Turbo for English + Multilingual for Dutch (~2-3 GB extra VRAM).
- **Multi-party conversations**: the avatar knows several people are present
  and addresses them ("good point, and what do you think?"); needs real
  conversation state in System 2. Together with **voice enrolment**
  ("this is Remco") so the avatar recognises people by voice.
- **Barge-in**: interrupt the avatar by speaking; VAD keeps running on the CPU
  while it talks.
- **Proper certificates** (e.g. a reverse proxy with Let's Encrypt) when the
  avatar runs somewhere reachable by others.

## Avatar quality

- JoyVASA's mouth follows individual syllables only loosely (correlation with
  loudness 0.2-0.5).
- Stride 1 is close to the render budget (23-35 FPS against 25 needed).
- Faint mirrored-hair specks at the far left edge of the upright crop.
