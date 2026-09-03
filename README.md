# Progressive Neural WebRTC Avatar

Current build: `neural-avatar-v2a-warmup-metrics`

This archive is step v2A of the latency optimization series. It deliberately
keeps the rendering algorithm unchanged while adding startup warm-up and enough
instrumentation to decide where the next patch will have the most impact.

This project is a working test of a browser-delivered talking avatar:

- Breeze TTS 2 generates speech.
- JoyVASA converts each speech phrase into facial motion.
- FasterLivePortrait renders the portrait on the NVIDIA GPU.
- FastAPI and aiortc send synchronized audio/video over WebRTC.
- `index.html` shows progress and per-phrase latency measurements.

## Current streaming model

The application now uses **progressive phrase playback**. It splits the input
text into short phrases, generates the first phrase, starts playing it, and
generates later phrases while WebRTC consumes the existing playback queue.

```text
text -> phrase 1: TTS -> render -> playback starts
     -> phrase 2: TTS -> render -> append to queue
     -> phrase 3: TTS -> render -> append to queue
```

This substantially improves time-to-first-speech for multi-sentence input. It
is not yet frame-by-frame neural streaming: Breeze, JoyVASA and
FasterLivePortrait still process each individual phrase as a small batch. If a
later phrase takes longer to generate than the media already buffered, the
client receives silence and holds the last video frame. That gap is measured as
an underrun in the UI.

## Project files

```text
.
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── index.html
├── patch_warping_onnx.py
├── server.py
├── inputs/
│   └── avatar.jpg
├── checkpoints/
├── models/
└── results/
```

## Requirements

- Linux with Docker Engine and Docker Compose v2
- A current NVIDIA driver with RTX 50-series support
- NVIDIA Container Toolkit configured for Docker
- Sufficient storage for Breeze, JoyVASA and FasterLivePortrait checkpoints
- A clear, front-facing portrait in `inputs/avatar.jpg`

The images use PyTorch 2.9.1, CUDA 12.8 and cuDNN 9 for RTX 5080/Blackwell
compatibility.

## Start the application

```bash
mkdir -p inputs checkpoints models results
cp /path/to/avatar.jpg inputs/avatar.jpg

docker compose build --no-cache
docker compose up
```

The first run downloads the model checkpoints. Open the UI when the services
are healthy:

```text
http://localhost:8000/
```

Verify the runtime and progressive settings:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Expected progressive fields:

```json
{
  "server_build": "neural-avatar-v2a-warmup-metrics",
  "startup_warmup_enabled": true,
  "startup_warmup_complete": true,
  "progressive_phrase_mode": true,
  "phrase_first_target_chars": 48,
  "phrase_target_chars": 100,
  "phrase_max_chars": 160
}
```

## Rebuild after this update

`server.py` and `index.html` are copied into the avatar image, so rebuild that
target:

```bash
docker compose build --no-cache webrtc-avatar
docker compose up -d --force-recreate webrtc-avatar
```

The Dockerfile itself is unchanged by the v2A warm-up and metrics update.

## Optimization roadmap and current step

Changes are intentionally introduced and measured one step at a time:

1. **v2A, this archive:** warm Breeze, HuBERT, JoyVASA and FasterLivePortrait;
   expose detailed TTS, pipeline and decode measurements.
2. **v2B, after measuring v2A:** bypass MP4/FFmpeg and feed generated frames
   directly into the WebRTC playback buffer.
3. **v2C:** add configurable render stride and bounded TTS prefetch.
4. **Later experiments:** test a Blackwell-compatible TensorRT path and Ditto
   online in separate containers.

The order matters. v2A establishes a warm, repeatable baseline; otherwise a
later speedup could be confused with one-time model initialization.

## Startup warm-up configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `STARTUP_WARMUP` | `true` | Runs one disposable TTS and avatar request before the server reports ready. |
| `WARMUP_TEXT` | `Hello.` | Short text used to initialize Breeze, HuBERT, JoyVASA and the renderer. |

Warm-up increases container startup time but moves lazy model initialization
out of the first interactive request. Warm-up failure is nonfatal: the server
continues, logs the exception, and exposes it as `startup_warmup_error` in
`/health`.

To roll back only this optimization without changing images or code:

```yaml
STARTUP_WARMUP: "false"
```

## Progressive phrase configuration

These are the only new settings added by this update. They are under the
`webrtc-avatar.environment` section of `docker-compose.yml`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `PROGRESSIVE_PHRASE_MODE` | `true` | Enables phrase splitting and appendable playback. Set `false` to restore one whole-utterance chunk. |
| `PHRASE_FIRST_TARGET_CHARS` | `48` | Approximate maximum size of the first phrase when no punctuation occurs earlier. Smaller values reduce first-response latency but may sound less natural. |
| `PHRASE_TARGET_CHARS` | `100` | Approximate target size for later phrases. Larger values improve prosody and reduce model-call overhead. |
| `PHRASE_MAX_CHARS` | `160` | Hard upper target for a phrase. It is never configured below `PHRASE_TARGET_CHARS`. |

The splitter always prefers sentence punctuation (`.`, `!`, `?`, `;`, `:`).
It may split at a comma after half of the target size. A short punctuated first
phrase such as `Hello!` is emitted immediately even though it is shorter than
`PHRASE_FIRST_TARGET_CHARS`.

Suggested tuning:

```yaml
# Lowest time-to-first-speech; more model calls and more phrase boundaries
PHRASE_FIRST_TARGET_CHARS: "30"
PHRASE_TARGET_CHARS: "70"
PHRASE_MAX_CHARS: "120"

# More natural longer phrases; slower first response
PHRASE_FIRST_TARGET_CHARS: "60"
PHRASE_TARGET_CHARS: "130"
PHRASE_MAX_CHARS: "200"
```

Start with the defaults and use the timing table and gap measurement before
changing them.

## Existing application configuration

These variables already existed in the working application and remain
available in `server.py` or `entrypoint.sh`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AVATAR_PATH` | `/workspace/inputs/avatar.jpg` | Source portrait inside the container. |
| `FLP_CONFIG_PATH` | `/workspace/FasterLivePortrait/configs/onnx_infer.yaml` | FasterLivePortrait ONNX configuration. |
| `RESULTS_ROOT` | `/workspace/results` | Temporary request output. Each phrase gets its own subdirectory. |
| `STARTUP_WARMUP` | `true` | Moves lazy model initialization into container startup. |
| `WARMUP_TEXT` | `Hello.` | Disposable phrase used by startup warm-up. |
| `BREEZE_TTS_URL` | `http://127.0.0.1:7860` | Breeze speech API used by the avatar server. |
| `BREEZE_VOICE_INSTRUCTION` | warm, clear, natural voice | Default instruction if the browser field is empty. |
| `BREEZE_CFG_SCALE` | `4` | Breeze instruction guidance. |
| `BREEZE_SEED` | `42` | Breeze generation seed. |
| `JOYVASA_CFG_SCALE` | `2.8` | JoyVASA audio-to-motion guidance. |
| `MAX_TEXT_LENGTH` | `500` | Maximum accepted browser input. |
| `LOG_LEVEL` | `INFO` | Server logging level. |
| `ICE_SERVERS_JSON` | `[]` | Browser STUN/TURN server array. |
| `PORT` | `8000` | FastAPI port used by `entrypoint.sh`. |
| `BREEZE_PORT` | `7860` | Breeze API port used by `entrypoint.sh`. |
| `FLP_CHECKPOINT_DIR` | FasterLivePortrait `checkpoints` directory | Model-download destination. |
| `BREEZE_MODEL_DIR` | `/workspace/models/Breeze-TTS-2` | Breeze model location. |

### Paste-back stability setting

`AVATAR_PASTE_BACK=false` remains in Compose from the earlier stability fix.
It streams FasterLivePortrait's face crop and avoids the per-frame GPU matrix
inverse that produced `CUSOLVER_STATUS_INTERNAL_ERROR` while Breeze and the
avatar shared limited GPU memory.

No new crop, rotation, expression or avatar-tuning variables are introduced by
the progressive phrase update.

## What the timing table measures

The browser adds one row when a phrase is ready:

| Column | Meaning |
| --- | --- |
| Phrase | Phrase number, total count and text. |
| TTS | Wall time spent waiting for Breeze to generate that phrase. |
| First byte | Time from the TTS request until its first response-body byte. |
| FLP pipeline | Time inside `run_audio_driving`, including JoyVASA, frame rendering and current video/FFmpeg work. |
| Decode | Time spent reopening the completed MP4 and decoding frames for WebRTC. |
| Render | Total wrapper time for the FLP pipeline and decode stage. |
| Media | Duration of the audio/video appended to WebRTC playback. |
| Buffer | Continuous synchronized audio/video remaining immediately after append. |
| Gap | Audio underrun while that phrase was being generated. Zero is ideal. |

The summary shows:

- time until the first phrase became playable;
- cumulative TTS time;
- cumulative render time;
- total request time including final playback;
- cumulative audio gap time.

The same measurements are written to server logs:

```text
Phrase 1/3 timings: tts=...s first_byte=...ms download=...ms render=...s pipeline=...ms decode=...ms frames=... media=...s buffer=...s underrun=...ms
```

### How to interpret measurements

- If `TTS + Render` for phrase 2 is lower than phrase 1's `Media`, playback can
  usually continue without a gap.
- If `Gap` is consistently positive, increase `PHRASE_TARGET_CHARS` so each
  generated phrase provides more playback time, or optimize the slower stage.
- If first-ready time is too long, reduce `PHRASE_FIRST_TARGET_CHARS`.
- A very short phrase such as `Hello!` has low media duration and can expose a
  gap before phrase 2. That is a latency/prosody tradeoff, not a WebRTC failure.

## v2A benchmark procedure

Use the same text used for the v1 baseline:

```text
Hello! This is my first neural streaming avatar test. This is great ...
```

1. Start the stack and wait for `Startup warm-up complete` in the avatar log.
2. Confirm `/health` reports the v2A build and `startup_warmup_complete: true`.
3. Open the browser only after health is ready.
4. Run the text twice without restarting either service.
5. Save the timing table and the `Phrase ... timings` log lines for both runs.
6. Record `nvidia-smi` utilization and memory if a gap or CUDA error appears.

Acceptance criteria for v2A:

- the WebRTC connection and three-phrase split still work;
- both test runs finish without a CUDA or checkpoint error;
- the first interactive request no longer prints `Loading weights` for HuBERT;
- the timing table shows nonzero First byte, FLP pipeline and Decode values;
- the second run is reasonably close to the first warm run.

Do not tune phrase lengths during this comparison. Changing the workload would
make the v1 and v2A measurements incomparable.

## Decision log

### v2A — warm-up and measurement first

- **Observed baseline:** phrase 2 required 3.108 seconds of TTS and 7.341
  seconds of rendering for 3.04 seconds of media; its gap was 9.82 seconds.
- **Decision:** move lazy loading out of the first request and split aggregate
  measurements before changing the rendering algorithm.
- **Reason:** the first render ran at 4.45 FPS while later renders reached about
  11.37 FPS, and the first request printed HuBERT weight loading.
- **Intentionally deferred:** direct frame delivery, frame skipping, TTS
  concurrency, TensorRT and Ditto. These remain separate measurable changes.
- **Rollback:** set `STARTUP_WARMUP=false`, rebuild/recreate only if the Compose
  file is baked into a deployment, and keep the added instrumentation.

## WebRTC behavior

The connection remains open between requests. The server sends:

| Event | Purpose |
| --- | --- |
| `plan` | Lists the phrases detected for the request. |
| `status` | Reports queue, TTS, render and drain phases. |
| `metrics` | Carries one timing record per completed phrase. |
| `playing` | Indicates that the first phrase was appended and playback began. |
| `summary` | Reports totals after all generated media finishes playing. |
| `ready` | Re-enables the browser button. |
| `error` | Reports a generation failure and clears queued media. |

The server holds the last video frame and sends silence if generation falls
behind. Audio packets are mono, 48 kHz, 20 ms. Video output is 512×512 at 30
FPS.

One global inference lock protects the shared neural pipeline. Different peers
are queued rather than executing FasterLivePortrait simultaneously.

## Compatibility fixes retained in the Dockerfile

The working Dockerfile still contains the fixes accumulated during setup:

1. PyTorch 2.9.1 with CUDA 12.8 and cuDNN 9 for RTX 50-series support.
2. Restricted checkpoint safe globals for JoyVASA's `argparse.Namespace` and
   `pathlib.PosixPath` metadata.
3. TorchCodec 0.9.1 plus a build-time WAV decoding smoke test.
4. Eager attention for JoyVASA HuBERT because SDPA does not support its
   requested attention output.
5. ONNX opset-20 conversion for volumetric 5-D GridSample.
6. ONNX Runtime rather than the TensorRT/PyCUDA route.

Only use checkpoints from trusted publishers. This project does not disable
PyTorch's restricted checkpoint loading globally.

## Troubleshooting

### No progressive fields in `/health`

The old `server.py` is still in the image. Rebuild and recreate
`webrtc-avatar`.

### Only one phrase appears

The input may be shorter than the first target, or
`PROGRESSIVE_PHRASE_MODE=false`. Add punctuation to test explicitly:

```text
Hello! This is phrase two. This is phrase three, generated later.
```

### Playback pauses between phrases

Look at the `Gap` column. If it is positive, generation is slower than queued
media playback. Keep the default phrase settings during the v2A benchmark.
Longer phrases can amortize fixed overhead, but cannot solve a sustained
production ratio above 1.0; direct frame streaming is the planned v2B fix.

### Startup warm-up failed

Inspect `startup_warmup_error` in `/health` and the preceding traceback. The
failure is nonfatal, so an interactive request can still be attempted. To
restore the v1 startup behavior while diagnosing it, set:

```yaml
STARTUP_WARMUP: "false"
```

### `CUSOLVER_STATUS_INTERNAL_ERROR`

Confirm that Compose still contains:

```yaml
AVATAR_PASTE_BACK: "false"
```

Then restart the avatar container. The error was in GPU paste-back, not
JoyVASA motion generation.

### Previous compatibility errors return

Use the provided Dockerfile and rebuild without cache. Do not replace it with
an unpatched upstream container recipe.

## Useful commands

```bash
# Build and start
docker compose build --no-cache
docker compose up

# Follow only the runtime services
docker compose logs -f breeze-tts webrtc-avatar

# Inspect service and health status
docker compose ps
curl -s http://127.0.0.1:8000/health | python -m json.tool

# Watch shared GPU memory and utilization
watch -n 0.5 nvidia-smi

# Stop without deleting downloaded host model directories
docker compose down
```

## Next architectural step: Ditto online

Progressive phrases are the practical next step for the existing working
FasterLivePortrait stack. They reduce perceived latency without replacing the
renderer.

For truly continuous sub-second interaction, the later migration target is an
online renderer such as Ditto. That architecture should accept incremental
audio features and emit frames continuously rather than requiring a WAV file,
motion pickle and completed MP4 for every phrase.

Before migrating, collect timing results from representative short and long
requests. Those measurements will show whether Breeze, JoyVASA or portrait
rendering is the dominant latency and provide a baseline for comparing Ditto.

## Upstream projects

- [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait)
- [Breeze TTS](https://github.com/breezeblue-ai/breeze-tts)
- [aiortc](https://github.com/aiortc/aiortc)

Review each project's code and model license before public distribution.
