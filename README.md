# Progressive Neural WebRTC Avatar

Current build: `neural-avatar-v2b-direct-memory`

This archive is v2B of the latency optimization series. It retains the proven
v2A.1 warm-up and phrase playback, then bypasses FasterLivePortrait's temporary
motion pickle, crop/original MP4 files, two FFmpeg mux operations and final MP4
decode. JoyVASA motion and rendered crop frames now pass directly through
memory to the existing WebRTC playback buffer.

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
FasterLivePortrait still complete each phrase before that phrase is appended to
playback. v2B removes file overhead and establishes a direct-frame baseline;
incremental frame delivery is a later, separately measured change. If a later
phrase takes longer to generate than the media already buffered, the client
receives silence and holds the last video frame. That gap is measured as an
underrun in the UI.

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
  "server_build": "neural-avatar-v2b-direct-memory",
  "render_backend": "direct-memory",
  "direct_memory_render": true,
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
docker compose up -d breeze-tts
docker compose up -d --force-recreate webrtc-avatar
```

The Dockerfile is changed in v2B. It now checks that the moving
FasterLivePortrait branch still exposes the direct-motion and direct-frame APIs
used by `server.py`.

## Optimization roadmap and current step

Changes are intentionally introduced and measured one step at a time:

1. **v2A.1, complete:** wait for Breeze, then warm Breeze, HuBERT, JoyVASA and
   FasterLivePortrait; expose detailed TTS, pipeline and decode measurements.
2. **v2B, this archive:** bypass the motion pickle, MP4/FFmpeg work and MP4
   decode; feed completed phrase frames directly into the WebRTC buffer.
3. **v2C:** add configurable render stride and bounded TTS prefetch.
4. **Later experiments:** test a Blackwell-compatible TensorRT path and Ditto
   online in separate containers.

The order matters. v2A establishes a warm, repeatable baseline. v2B isolates
file overhead. Only after measuring both should v2C deliberately trade temporal
detail or GPU concurrency for lower gaps.

## Startup warm-up configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `STARTUP_WARMUP` | `true` | Runs one disposable TTS and avatar request before the server reports ready. |
| `WARMUP_TEXT` | `Hello.` | Short text used to initialize Breeze, HuBERT, JoyVASA and the renderer. |
| `TTS_STARTUP_WAIT_SECONDS` | `300` | Maximum time to wait for Breeze before recording a nonfatal warm-up failure. |
| `TTS_STARTUP_POLL_SECONDS` | `2` | Delay between Breeze readiness checks. |

Warm-up increases container startup time but moves lazy model initialization
out of the first interactive request. Warm-up failure is nonfatal: the server
continues, logs the exception, and exposes it as `startup_warmup_error` in
`/health`.

To roll back only this optimization without changing images or code:

```yaml
STARTUP_WARMUP: "false"
```

## Direct-memory render configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `DIRECT_MEMORY_RENDER` | `true` | Uses JoyVASA motion and FasterLivePortrait crop frames directly in memory. |
| `AVATAR_PASTE_BACK` | `false` | Required for the v2B direct path; avoids the unstable GPU paste-back operation. |

The direct backend removes these per-phrase operations:

1. writing and reopening a motion pickle;
2. encoding both crop and original MP4 files;
3. launching FFmpeg twice to mux audio into both MP4 files;
4. reopening and decoding the completed selected MP4.

It does not change JoyVASA motion, FasterLivePortrait frame inference, output
resolution, audio format or WebRTC timing. It also does not yet append each
frame while inference is running; a phrase becomes playable after its entire
frame loop completes.

Immediate A/B rollback:

```yaml
DIRECT_MEMORY_RENDER: "false"
```

Recreate only the avatar service after changing this runtime setting:

```bash
docker compose up -d --force-recreate webrtc-avatar
```

When `AVATAR_PASTE_BACK=true`, the server automatically uses `legacy-mp4`
because v2B supports the crop path only.

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
| `DIRECT_MEMORY_RENDER` | `true` | Selects the v2B in-memory renderer; `false` selects the v2A.1 legacy MP4 path. |
| `STARTUP_WARMUP` | `true` | Moves lazy model initialization into container startup. |
| `WARMUP_TEXT` | `Hello.` | Disposable phrase used by startup warm-up. |
| `TTS_STARTUP_WAIT_SECONDS` | `300` | Readiness timeout used before startup warm-up. |
| `TTS_STARTUP_POLL_SECONDS` | `2` | Readiness polling interval. |
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

Docker build arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `FASTER_LIVE_PORTRAIT_REF` | `master` | FasterLivePortrait GitHub branch downloaded into the avatar image. The v2B API assertions protect against incompatible branch changes. |
| `BREEZE_REF` | `main` | Breeze TTS GitHub branch downloaded into the Breeze image. |

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
| Backend | `direct-memory` for v2B or `legacy-mp4` for the rollback path. |
| Motion | JoyVASA audio-to-motion time on the direct path. The legacy path reports zero because upstream exposes only its aggregate. |
| Frame loop | Time spent calling FasterLivePortrait for all phrase frames. Legacy mode reports zero because upstream exposes only its aggregate. |
| FPS | Effective FasterLivePortrait frame-loop throughput, not WebRTC's fixed 30 FPS output rate. Legacy mode reports zero. |
| FLP pipeline | Direct motion plus frame-loop time, or legacy `run_audio_driving` time including its file/FFmpeg work. |
| Decode | MP4 decode time. It must be zero for `direct-memory`. |
| Render | Total render wrapper time. On the direct path this should closely match FLP pipeline time. |
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
Phrase 1/3 timings: tts=...s first_byte=...ms download=...ms render=...s backend=direct-memory motion=...ms frame_loop=...ms effective_fps=... pipeline=...ms decode=0ms frames=... media=...s buffer=...s underrun=...ms
```

### How to interpret measurements

- If `TTS + Render` for phrase 2 is lower than phrase 1's `Media`, playback can
  usually continue without a gap.
- If `Gap` is consistently positive, increase `PHRASE_TARGET_CHARS` so each
  generated phrase provides more playback time, or optimize the slower stage.
- If first-ready time is too long, reduce `PHRASE_FIRST_TARGET_CHARS`.
- If `Frame loop FPS` remains below the source motion rate (normally 25 FPS),
  file removal alone cannot eliminate gaps. That result justifies the v2C
  stride or true incremental-render experiment.
- A very short phrase such as `Hello!` has low media duration and can expose a
  gap before phrase 2. That is a latency/prosody tradeoff, not a WebRTC failure.

## Recorded v2A.1 benchmark

Test text:

```text
Hello! This is my first neural streaming avatar test. Love to hear you talking, greetings, love it!
```

| Phrase | TTS | First byte | FLP pipeline | Decode | Media | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Hello!` | 689 ms | 206 ms | 1,805 ms | 16 ms | 0.64 s | 0 ms |
| `This is my first neural streaming avatar test.` | 2,987 ms | 193 ms | 7,436 ms | 45 ms | 3.04 s | 9,840 ms |
| `Love to hear you talking, greetings, love it!` | 2,612 ms | 194 ms | 6,463 ms | 41 ms | 2.64 s | 6,080 ms |

Interpretation:

- Startup warm-up worked: the first `Hello!` phrase was ready in roughly 2.51
  seconds instead of the earlier cold-path result near 7.84 seconds.
- Breeze returned its first body byte in about 0.2 seconds, but completing each
  TTS phrase still took 0.69–2.99 seconds.
- FasterLivePortrait rendered approximately 11.2 source frames per second while
  motion media is 25 FPS. The frame loop, not MP4 decode, is the main sustained
  bottleneck.
- MP4 decode was only 16–45 ms. The repeated MP4 encodes and FFmpeg muxes are
  avoidable, but v2B is expected to be a modest improvement rather than the
  final solution.

## v2B A/B benchmark procedure

Use the exact same text above and do not change phrase settings.

1. Rebuild and start v2B; wait for `Startup warm-up complete`.
2. Confirm `/health` reports `neural-avatar-v2b-direct-memory`,
   `render_backend: direct-memory` and `startup_warmup_complete: true`.
3. Run the text twice without restarting either service. Treat the second run
   as the primary warm measurement.
4. Save the UI table and the three `Phrase ... timings` log lines.
5. Set `DIRECT_MEMORY_RENDER: "false"`, recreate the avatar service, wait for
   warm-up, and run the same text again if an on-machine A/B comparison is
   needed.
6. Record `nvidia-smi` utilization and memory if a gap or CUDA error appears.

Acceptance criteria for v2B:

- the WebRTC connection and three-phrase split still work;
- both test runs finish without a CUDA or checkpoint error;
- every row reports backend `direct-memory` and Decode `0 ms`;
- request-time logs contain no FFmpeg banner;
- phrase directories contain `speech.pcm` and `speech.wav`, but no motion
  pickle or MP4 output;
- frame count, source FPS, lip sync and crop quality match the legacy path;
- FLP pipeline time is lower or equal within normal run-to-run noise.

If direct memory is slower by more than 10% on two warm runs, or output differs,
switch back to `DIRECT_MEMORY_RENDER=false` and keep the logs. Do not proceed to
v2C until the discrepancy is understood.

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

### v2A.1 — wait for the TTS dependency

- **Observed failure:** the avatar was recreated with `--no-deps` while Breeze
  was not running. Warm-up reached port 7860 after 10 ms and received a
  connection failure; `/health` then correctly returned 503 because TTS was
  unavailable.
- **Decision:** poll Breeze for up to 300 seconds before warm-up and document a
  dependency-aware restart sequence.
- **Reason:** Compose dependency conditions do not apply when dependencies are
  explicitly disabled with `--no-deps`.
- **Rollback:** set `TTS_STARTUP_WAIT_SECONDS: "0"` to restore the immediate
  warm-up attempt. This is separate from disabling warm-up entirely.

### v2B — remove file transport between neural stages

- **Observed baseline:** the warm frame loop sustained about 11.2 FPS for 25
  FPS motion. MP4 decode consumed only 16–45 ms, while upstream also created a
  motion pickle, encoded two videos and ran two FFmpeg muxes per phrase.
- **Decision:** call the same JoyVASA `gen_motion_sequence` and
  FasterLivePortrait `run_with_pkl` APIs directly, retain the crop frames in
  memory, and expose motion/frame-loop/FPS metrics.
- **Reason:** this removes unnecessary work without changing model settings,
  frame selection or quality, and produces the clean baseline required before
  frame-stride or concurrency experiments.
- **Intentionally deferred:** incremental per-frame queue append, render stride,
  TTS prefetch, TensorRT and Ditto.
- **Rollback:** set `DIRECT_MEMORY_RENDER=false`; the complete v2A.1 MP4 path is
  still present in `server.py`.

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
media playback. Keep the default phrase settings during the v2B A/B benchmark.
Longer phrases can amortize fixed overhead, but cannot solve a sustained
production ratio above 1.0. v2B removes file overhead but still batches a full
phrase. Render stride and incremental delivery are planned for v2C.

### Backend says `legacy-mp4`

Confirm both values in Compose:

```yaml
AVATAR_PASTE_BACK: "false"
DIRECT_MEMORY_RENDER: "true"
```

Then force-recreate `webrtc-avatar`. Paste-back deliberately selects the legacy
backend because direct-memory v2B supports crop frames only.

### Direct-memory path fails after an upstream update

Set `DIRECT_MEMORY_RENDER=false` and recreate the avatar service. The legacy
path is retained as the compatibility fallback. A fresh Docker build also
checks the upstream API names used by v2B and should fail early if those names
change.

### Startup warm-up failed

Inspect `startup_warmup_error` in `/health` and the preceding traceback. The
failure is nonfatal, so an interactive request can still be attempted. To
recover from `ConnectError` at `127.0.0.1:7860`, start Breeze and then restart
the avatar so warm-up runs again:

```bash
docker compose up -d breeze-tts
docker compose restart webrtc-avatar
```

Do not use `--no-deps` unless `docker compose ps` already shows Breeze as
healthy. To restore the v1 startup behavior while diagnosing another warm-up
problem, set:

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

v2C render stride and bounded prefetch are the next experiments for the working
FasterLivePortrait stack. They should be introduced one at a time using the v2B
direct-memory measurements as the baseline.

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
