# Neural WebRTC Avatar

A talking, listening avatar that runs entirely on one machine with an NVIDIA
GPU. A portrait photo speaks with a cloned voice, in real time, in the browser;
it hears you through the browser's microphone (Dutch and English), reacts
quickly, and answers questions from a folder of your own Markdown notes.

Current build: `neural-avatar-v2y-structure` (see [docs/RELEASE.md](docs/RELEASE.md)).

## How it works

Four services, each in its own Docker image because their dependencies clash.
They talk over local Unix sockets and HTTP on the same host.

| Service | Package | What it does |
|---|---|---|
| `webrtc-avatar` | `src/avatar` | Web page, WebRTC audio/video, phrase splitting, JoyVASA (audio to face motion), FasterLivePortrait rendering with TensorRT |
| `tts` | `src/tts` | Chatterbox Turbo text-to-speech behind a small HTTP adapter |
| `listener` | `src/listener` | Voice activity detection, streaming speech recognition and speaker labels |
| `conductor` | `src/conductor` | Turn-taking, System 1 (fast typed decisions and reactions), System 2 (answers from `knowledge/`), conversation log and review |

```
browser  <-- WebRTC -->  avatar  -- HTTP -->  tts
   microphone              |  \
                           |   +-- socket -->  listener  (speech to text)
                           +------ socket -->  conductor (what to say)
```

More detail: [docs/architecture/realtime-rendering.md](docs/architecture/realtime-rendering.md)
for the speaking side and [docs/architecture/listening.md](docs/architecture/listening.md)
for the listening side.

## Requirements

- Linux with Docker Engine and Docker Compose v2
- An NVIDIA GPU with 16 GB (built and tested on an RTX 5080 Laptop GPU), a
  current driver and the NVIDIA Container Toolkit
- Disk space for the images, checkpoints and models (the test machine uses
  about 105 GB in total)
- A clear, front-facing portrait as `inputs/avatar.jpg` and a reference
  recording of the voice as `inputs/voice-preset-nohello.wav`

## Quick start

```bash
docker compose build        # first time: long; the images take tens of GB
docker compose up -d        # first start downloads the model checkpoints
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

Open `http://localhost:8000/` on the machine itself, or
`https://<address>:8443/` from another machine (self-signed certificate; the
microphone needs HTTPS). Type a text, or enable the microphone and talk.

After a restart of the machine, run `docker compose up -d` again: the `tts` and
`webrtc-avatar` containers do not restart on their own.

## Configuration

An empty `.env` is the tested live setup. Every setting, its default and its
meaning is in [CONFIG.md](CONFIG.md); `.env.example` shows the form.

```bash
python3 src/avatar/config.py show      # effective avatar settings
AVATAR_PROFILE=development docker compose up -d webrtc-avatar
```

The avatar refuses to start on an unknown name, a value out of range or an
impossible combination, and lists every problem.

## Repository layout

```
src/            the code, one Python package per service
  avatar/       server.py (WebRTC, rendering), serve.py, config.py, index.html
  conductor/    app.py, system1.py, system2.py, knowledge.py, review.py, forget.py
  listener/     worker.py
  tts/          chatterbox_api.py
  shared/       the socket protocols between the services
tests/          unit tests per service
scripts/        what you run in normal use: benchmarks, review, recording review
tools/          diagnosis and measurement scripts with their test audio
docker/         Dockerfile (one stage per service) and entrypoint.sh
docs/           architecture, changelog and history
config/         system1.json (classes and reactions, reloaded live), reference/
inputs/         the portrait and the reference voice
knowledge/      the Markdown pages the avatar answers from
checkpoints/ models/ results/   downloaded models and output (not in git)
```

## Testing and tools

```bash
tools/run_tests.sh                    # all unit tests on the working tree, no rebuild, no GPU
tools/config_snapshot.py check        # running stack against config/reference/
scripts/review_recording.sh "text"    # record a phrase as frame sheets
scripts/review.sh                     # review the logged conversations
scripts/quality_benchmark.sh          # mouth and gaze quality matrix
```

The same unit tests also run during `docker compose build`.
[tools/README.md](tools/README.md) lists the diagnosis scripts.

## Conversations and privacy

Every turn is logged locally in `results/conversations/`; microphone audio is
kept only with `LISTENER_SAVE_AUDIO=true`. Nothing leaves the machine. Delete
logs with `docker compose exec conductor python -m conductor.forget --all --yes`
(see the header of `src/conductor/forget.py` for other selections).

## Documentation

- [CONFIG.md](CONFIG.md): every setting
- [BACKLOG.md](BACKLOG.md): what is next
- [docs/README.md](docs/README.md): index of everything else, including the
  history of earlier versions

## Upstream projects

[FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait),
[JoyVASA](https://github.com/jdh-algo/JoyVASA),
[Chatterbox](https://github.com/resemble-ai/chatterbox),
[aiortc](https://github.com/aiortc/aiortc),
[llama.cpp](https://github.com/ggml-org/llama.cpp) and the speech and language
models named in [CONFIG.md](CONFIG.md). Review each project's code and model
licence before distributing anything publicly.
