# Progressive Neural WebRTC Avatar

Current build: `neural-avatar-v2f-adaptive-stride`

This archive is v2F of the latency optimization series. It retains the v2E.1
benchmark and adds an adaptive stride 2→3 render policy based on measured
playback-buffer pressure. One command can generate
a deterministic Breeze reference fixture, exercise the same WebRTC/API path as
the browser, run identical text through every voice mode, and save JSON, CSV
and Markdown results. Optional stride/prefetch matrices make future changes
comparable without replacing occasional listening tests.

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
playback. Each phrase still waits for its selected frames before being appended;
incremental frame delivery is a later, separately measured change. If a later
phrase takes longer to generate than the media already buffered, the client
receives silence and holds the last video frame. That gap is measured as an
underrun in the UI.

## Project files

```text
.
├── Dockerfile
├── benchmark.sh
├── benchmark_avatar.py
├── test_benchmark_handshake.py
├── docker-compose.yml
├── entrypoint.sh
├── index.html
├── patch_warping_onnx.py
├── server.py
├── inputs/
│   ├── avatar.jpg
│   ├── voice-preset.wav  # optional
│   └── voice-preset.txt  # optional exact transcript
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
  "server_build": "neural-avatar-v2f-adaptive-stride",
  "render_backend": "direct-memory",
  "direct_memory_render": true,
  "render_stride": 2,
  "adaptive_render_stride": true,
  "catchup_render_stride": 3,
  "catchup_buffer_seconds": 0.75,
  "tts_prefetch": false,
  "tts_prefetch_depth": 0,
  "default_voice_mode": "design",
  "voice_modes": ["design", "preset-clone", "preset-direction"],
  "preset_voice_configured": false,
  "design_cfg_scale": 4.0,
  "preset_clone_cfg_scale": 1.0,
  "preset_direction_cfg_scale": 4.0,
  "startup_warmup_enabled": true,
  "startup_warmup_complete": true,
  "progressive_phrase_mode": true,
  "phrase_first_target_chars": 48,
  "phrase_target_chars": 100,
  "phrase_max_chars": 160
}
```

## Repeatable end-to-end benchmark

Build the new avatar image once, then run:

```bash
docker compose build webrtc-avatar benchmark-fixture
./benchmark.sh
```

The harness intentionally uses the production path:

1. starts and health-checks Breeze;
2. generates a dedicated `benchmark-voice-preset.wav` and exact transcript
   when the pair is missing;
3. recreates the avatar with the selected scenario settings;
4. negotiates `/offer` as a headless WebRTC client;
5. consumes the real audio/video tracks and sends requests through the same
   data channel as `index.html`;
6. warms each selected voice mode once;
7. runs the same benchmark text three times per mode in round-robin order;
8. stores raw events, phrase rows and median comparisons;
9. combines all saved scenarios into a timestamped comparison CSV and Markdown
   report.

Your normal `voice-preset.wav/.txt` pair is never replaced by the default test.
To intentionally regenerate only the dedicated canonical benchmark fixture,
run:

```bash
REGENERATE_PRESET=1 ./benchmark.sh
```

The fixture uses a fixed sentence, instruction, CFG 4 and seed 42. Its manifest
records the exact inputs, audio duration, creation timing and SHA-256 hashes.
The runner uses separate but fixed benchmark text for all voice modes, so the
reference prompt is not accidentally favored by repeating its own transcript.

Each scenario creates a timestamped directory:

```text
results/benchmarks/
├── comparison-20260903T211500Z.csv
├── comparison-20260903T211500Z.md
└── stride-2_adaptive-true_catchup-3_prefetch-false/
    └── 20260903T210000Z/
        ├── benchmark.json  # environment, fixture, raw events and summaries
        ├── phrases.csv     # one row per phrase
        └── report.md       # median and per-run tables
```

Useful controls:

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `BENCHMARK_REPEATS` | `3` | Measured runs per mode. |
| `BENCHMARK_WARMUPS` | `1` | Short unmeasured warm-ups per mode. This moves preset cold start out of measured results while retaining its raw warm-up record. |
| `BENCHMARK_MODES` | all three modes | Comma-separated browser mode IDs. |
| `BENCHMARK_RENDER_STRIDES` | `2` | Space-separated render-stride matrix. |
| `BENCHMARK_ADAPTIVE_VALUES` | `true` | Space-separated adaptive-policy matrix. Use `"false true"` for the v2F A/B test. |
| `BENCHMARK_CATCHUP_STRIDE` | `3` | Stride used when adaptive mode detects low buffer. |
| `BENCHMARK_CATCHUP_BUFFER_SECONDS` | `0.75` | Switch to catch-up stride below this buffer level. |
| `BENCHMARK_PREFETCH_VALUES` | `false` | Space-separated `true`/`false` matrix. |
| `REGENERATE_PRESET` | `0` | Set `1` to deliberately regenerate the selected benchmark preset pair. |
| `BENCHMARK_BUILD` | `0` | Set `1` to rebuild benchmark/avatar images before running. |
| `BENCHMARK_PRESET_BASENAME` | `benchmark-voice-preset` | Select the WAV/transcript basename under `inputs/`. Use `voice-preset` to benchmark your normal preset instead. |

Examples:

```bash
# Quick smoke test: one measured run per mode
BENCHMARK_REPEATS=1 ./benchmark.sh

# Compare fixed stride 2 with adaptive stride 2→3
BENCHMARK_MODES=design BENCHMARK_ADAPTIVE_VALUES="false true" ./benchmark.sh

# Reproduce the fixed-stride experiment (disable v2F adaptation)
BENCHMARK_MODES=design BENCHMARK_ADAPTIVE_VALUES=false \
  BENCHMARK_RENDER_STRIDES="2 3 4" ./benchmark.sh

# Reproduce the rejected prefetch experiment as a separate scenario
BENCHMARK_PREFETCH_VALUES="false true" ./benchmark.sh

# Test only preset clone five times
BENCHMARK_MODES="preset-clone" BENCHMARK_REPEATS=5 ./benchmark.sh

# Benchmark your existing production preset without changing it
BENCHMARK_PRESET_BASENAME=voice-preset ./benchmark.sh
```

The Markdown report compares first-ready time, median first byte, TTS real-time
factor, render real-time factor, neural FPS, generated media duration, gaps and
client wall time. TTS RTF is the primary speed comparison because different
voice modes can speak the same text at different speeds. Voice identity,
naturalness and lip-sync quality remain listening/viewing checks; performance
metrics cannot replace them.

The top-level comparison files include every discovered scenario and record
hashes for both the benchmark text and preset WAV. Only treat rows with matching
hashes as a controlled speed comparison.

To run the handshake regression tests in the built image:

```bash
docker compose run --rm --no-deps --entrypoint python benchmark-fixture \
  /workspace/FasterLivePortrait/test_benchmark_handshake.py -v
```

## Configure an optional preset voice

Preset modes are deliberately unavailable until both files exist:

```text
inputs/voice-preset.wav
inputs/voice-preset.txt
```

`voice-preset.txt` must contain the exact words spoken in the WAV, including
filled pauses or vocal-event text where applicable. Use a clean single-speaker
recording that you own or are authorized to clone.

To use an existing recording:

```bash
cp /path/to/authorized-reference.wav inputs/voice-preset.wav
printf '%s\n' 'The exact words spoken in the reference recording.' > inputs/voice-preset.txt
docker compose up -d --force-recreate webrtc-avatar
```

To create a synthetic preset, first ask Breeze to design one good reference
sentence and save that result once. The response is raw mono 24 kHz signed
16-bit PCM:

```bash
curl --fail --silent --show-error \
  -X POST http://127.0.0.1:7860/v1/audio/speech \
  -F 'text=Hello, I am your friendly virtual assistant, ready to help.' \
  -F 'instruction=A warm, clear, natural voice with a calm conversational delivery.' \
  -F 'cfg_scale=4' \
  -F 'seed=42' \
  --output inputs/voice-preset.pcm

ffmpeg -y -f s16le -ar 24000 -ac 1 \
  -i inputs/voice-preset.pcm inputs/voice-preset.wav

printf '%s\n' \
  'Hello, I am your friendly virtual assistant, ready to help.' \
  > inputs/voice-preset.txt

docker compose up -d --force-recreate webrtc-avatar
```

After recreation, `/health` reports `preset_voice_configured: true` and the two
preset options become selectable. The raw `.pcm` file can then be removed.
Changing only these input files does not require an image rebuild.

## Rebuild after this update

`server.py`, `index.html` and the benchmark client are copied into the avatar
image, so rebuild that target:

```bash
docker compose build --no-cache webrtc-avatar
docker compose up -d breeze-tts
docker compose up -d --force-recreate webrtc-avatar
```

The Dockerfile retains the v2B checks that the moving FasterLivePortrait branch
still exposes the direct-motion and direct-frame APIs used by `server.py`.

## Optimization roadmap and current step

Changes are intentionally introduced and measured one step at a time:

1. **v2A.1, complete:** wait for Breeze, then warm Breeze, HuBERT, JoyVASA and
   FasterLivePortrait; expose detailed TTS, pipeline and decode measurements.
2. **v2B, complete:** bypass the motion pickle, MP4/FFmpeg work and MP4
   decode; feed completed phrase frames directly into the WebRTC buffer.
3. **v2C-A, complete:** render alternating motion frames with a configurable
   stride and preserve media duration using frame holds.
4. **v2C-B, rejected:** overlap one future TTS request with the current
   portrait render. On the shared RTX 5080 it approximately halved neural FPS,
   so v2D defaults prefetch off.
5. **v2D, complete:** compare designed voice CFG 4, preset clone CFG 1 and
   preset direction CFG 4 while recording the selected mode per phrase.
6. **v2E, complete:** automate canonical preset creation and repeatable
   headless WebRTC comparisons with saved machine-readable reports.
7. **v2E.1, complete:** make the initial benchmark data-channel handshake
   reliable across aiortc channel-open event ordering.
8. **v2F, this archive:** keep base stride 2 but use the measured stride 3
   catch-up mode whenever the media buffer is below 0.75 seconds.
9. **Later experiments:** add phrase-boundary motion blending, then test a
   Blackwell-compatible TensorRT path and Ditto
   online in separate containers.

The order matters. v2A establishes a warm baseline, v2B isolates file overhead,
and v2C-A measures a deliberate temporal-quality tradeoff. The rejected v2C-B
result establishes that TTS and rendering should not overlap on this GPU. v2D
therefore isolates voice identity and CFG cost without changing scheduling.

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
| `RENDER_STRIDE` | `2` | Base stride used for the first phrase and while the buffer is healthy. |
| `ADAPTIVE_RENDER_STRIDE` | `true` | Enables buffer-aware selection between base and catch-up stride. |
| `CATCHUP_RENDER_STRIDE` | `3` | Lower-cost stride selected while the buffer is below its threshold. |
| `CATCHUP_BUFFER_SECONDS` | `0.75` | Minimum healthy buffered media; below this value selects catch-up mode. |

The direct backend removes these per-phrase operations:

1. writing and reopening a motion pickle;
2. encoding both crop and original MP4 files;
3. launching FFmpeg twice to mux audio into both MP4 files;
4. reopening and decoding the completed selected MP4.

It does not change JoyVASA motion, FasterLivePortrait frame inference, output
resolution, audio format or WebRTC timing. It also does not yet append each
frame while inference is running; a phrase becomes playable after its entire
frame loop completes.

The v2F policy always renders phrase 1 at the base stride for initial visual
quality. Before rendering every later phrase, it samples the synchronized
audio/video buffer. Below 0.75 seconds it selects stride 3; otherwise it keeps
stride 2. The phrase metric records the buffer level and one of
`first-phrase-quality`, `low-buffer-catchup`, `buffer-healthy` or `fixed`.

Immediate v2F rollback while retaining direct-memory rendering:

```yaml
ADAPTIVE_RENDER_STRIDE: "false"
RENDER_STRIDE: "2"
```

Immediate A/B rollback:

```yaml
DIRECT_MEMORY_RENDER: "false"
```

Recreate only the avatar service after changing this runtime setting:

```bash
docker compose up -d --force-recreate webrtc-avatar
```

When `AVATAR_PASTE_BACK=true`, the server automatically uses `legacy-mp4`
with an effective stride of one because direct-memory rendering supports the
crop path only.

### How render stride preserves timing

At the default 25 FPS JoyVASA rate, an example 76-frame phrase represents 3.04
seconds. With `RENDER_STRIDE=2`, v2C-A renders frames 0, 2, 4 … 74: 38 neural
frames at an effective 12.5 FPS, still representing 3.04 seconds. The WebRTC
queue then repeats/holds those frames while emitting its fixed 30 FPS stream.

| Stride | Neural frames rendered | Effective animation rate | Expected effect |
| ---: | ---: | ---: | --- |
| `1` | 100% | 25 FPS | Exact v2B motion detail and rollback. |
| `2` | about 50% | 12.5 FPS | v2C-A default; expected near-half frame-loop time. |
| `3` | about 33% | 8.33 FPS | More speed but visibly less smooth; diagnostic only. |

Stride does not change TTS, JoyVASA motion generation, lip-sync timestamps,
audio duration, WebRTC's 30 FPS transport, or image resolution. It changes how
many intermediate neural portrait frames are inferred. Compare lip sync,
blinks, head motion and visual smoothness before accepting this tradeoff.

## Bounded TTS prefetch configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `TTS_PREFETCH` | `false` | Experimental v2C-B overlap. Keep disabled on the shared RTX 5080. |

Prefetch depth is deliberately fixed at one. There is never more than one
Breeze request active, and at most one future WAV is retained. This bounds GPU
and memory pressure while allowing separate Docker services to overlap TTS and
portrait inference.

```mermaid
sequenceDiagram
    participant Loop as Phrase loop
    participant Breeze
    participant FLP
    Loop->>Breeze: Generate audio 1
    Breeze-->>Loop: Audio 1
    par Current render
        Loop->>FLP: Render phrase 1
    and Next prefetch
        Loop->>Breeze: Generate audio 2
    end
```

The first phrase is never prefetched. For later phrases the UI separates:

- `TTS`: total Breeze wall time, including any GPU-contention slowdown;
- `TTS wait`: time the serial phrase loop actually waited for that result;
- `TTS overlap`: TTS time hidden behind the preceding render.

The v2C-B test confirmed this contention. Neural portrait speed fell from about
11.6–12.0 FPS to 6.4 FPS during the strongest overlap, phrase-two rendering
rose from about 3.35 seconds to 7.08 seconds, and TTS also slowed. Phrase four
recovered to 11.61 FPS when no later TTS request overlapped it. The accepted
v2D scheduling is therefore:

```yaml
TTS_PREFETCH: "false"
```

Then force-recreate `webrtc-avatar`; no image rebuild is required.

## Voice mode configuration

The selected mode is resolved once per browser request and reused for every
phrase. The browser never supplies a reference path; it can only select one of
the server-configured behaviors.

| Browser mode | Reference | Direction | CFG default | Intended use |
| --- | --- | --- | ---: | --- |
| `design` | No | Browser field | `4` | Current reference-free behavior; identity may drift between separate phrase requests. |
| `preset-clone` | WAV + exact transcript | Fixed neutral instruction | `1` | Best first test for stable identity and lower CFG cost. |
| `preset-direction` | WAV + exact transcript | Browser field | `4` | Stable reference identity with controllable emotion, pace or delivery. |

Both preset modes send the configured WAV and transcript to Breeze for every
phrase. Breeze's current API re-encodes uploaded reference audio each request;
v2D intentionally does not patch or cache Breeze internals yet. Compare first
byte and total TTS before deciding whether a server-side audio-token cache is
worth another isolated version.

| Setting | Default | Purpose |
| --- | --- | --- |
| `BREEZE_DEFAULT_VOICE_MODE` | `design` | Initial browser mode. A missing preset automatically falls back to design in the UI and warm-up. |
| `BREEZE_VOICE_INSTRUCTION` | warm, clear, natural voice | Default design/direction text and initial browser value. |
| `BREEZE_DESIGN_CFG_SCALE` | `4` | Guidance used for reference-free voice design. Legacy `BREEZE_CFG_SCALE` remains a fallback. |
| `BREEZE_PRESET_CLONE_CFG_SCALE` | `1` | Preset clone guidance. CFG 1 avoids the additional negative-guidance branch. |
| `BREEZE_PRESET_DIRECTION_CFG_SCALE` | `4` | Guidance used when applying a direction to the preset identity. |
| `BREEZE_PRESET_CLONE_INSTRUCTION` | Speak naturally in the reference voice. | Neutral instruction used when the browser selects clone mode. |
| `BREEZE_PRESET_AUDIO_PATH` | `/workspace/inputs/voice-preset.wav` | Server-controlled preset WAV. |
| `BREEZE_PRESET_TRANSCRIPT_FILE` | `/workspace/inputs/voice-preset.txt` | UTF-8 file containing the exact reference transcript. |
| `BREEZE_PRESET_TRANSCRIPT` | empty | Optional inline transcript overriding the `.txt` file. |

The timing table adds `Voice` and `CFG` columns. For a fair A/B comparison,
keep `TTS_PREFETCH=false`, use identical input text, allow warm-up to finish,
and test each mode at least three times. Record voice consistency, first byte,
total TTS, render FPS and gaps; changing modes should not change render FPS.

## Progressive phrase configuration

These settings are under the `webrtc-avatar.environment` section of
`docker-compose.yml`.

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
| `RENDER_STRIDE` | `2` | Base neural-frame decimation used by direct memory. |
| `ADAPTIVE_RENDER_STRIDE` | `true` | Enables the v2F buffer-aware render policy. |
| `CATCHUP_RENDER_STRIDE` | `3` | Stride used when adaptive catch-up is selected. Never lower than the base stride. |
| `CATCHUP_BUFFER_SECONDS` | `0.75` | Catch-up threshold measured immediately before each phrase render. |
| `TTS_PREFETCH` | `false` | Rejected shared-GPU overlap experiment. `true` restores v2C-B for diagnostics. |
| `STARTUP_WARMUP` | `true` | Moves lazy model initialization into container startup. |
| `WARMUP_TEXT` | `Hello.` | Disposable phrase used by startup warm-up. |
| `TTS_STARTUP_WAIT_SECONDS` | `300` | Readiness timeout used before startup warm-up. |
| `TTS_STARTUP_POLL_SECONDS` | `2` | Readiness polling interval. |
| `BREEZE_TTS_URL` | `http://127.0.0.1:7860` | Breeze speech API used by the avatar server. |
| `BREEZE_VOICE_INSTRUCTION` | warm, clear, natural voice | Default instruction if the browser field is empty. |
| `BREEZE_DEFAULT_VOICE_MODE` | `design` | Initial browser mode; preset modes require configured reference files. |
| `BREEZE_DESIGN_CFG_SCALE` | `4` | Voice-design instruction guidance. |
| `BREEZE_PRESET_CLONE_CFG_SCALE` | `1` | Preset clone guidance without the extra CFG branch. |
| `BREEZE_PRESET_DIRECTION_CFG_SCALE` | `4` | Preset identity plus direction guidance. |
| `BREEZE_PRESET_AUDIO_PATH` | `/workspace/inputs/voice-preset.wav` | Fixed reference audio used by both preset modes. |
| `BREEZE_PRESET_TRANSCRIPT_FILE` | `/workspace/inputs/voice-preset.txt` | Exact reference transcript file. |
| `BREEZE_PRESET_TRANSCRIPT` | empty | Optional inline transcript overriding the file. |
| `BREEZE_CFG_SCALE` | unset | Backward-compatible fallback for `BREEZE_DESIGN_CFG_SCALE`. |
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
| TTS | Total Breeze wall time for the phrase, including any concurrency slowdown. |
| TTS wait | Portion of TTS wall time that blocked the phrase loop. |
| TTS overlap | Portion of TTS wall time hidden behind the preceding render. |
| First byte | Time from the TTS request until its first response-body byte. |
| Backend | `direct-memory` for v2B or `legacy-mp4` for the rollback path. |
| Stride | Interval between rendered JoyVASA motion frames. `2` renders frames 0, 2, 4 and so on. |
| Motion | JoyVASA audio-to-motion time on the direct path. The legacy path reports zero because upstream exposes only its aggregate. |
| Frame loop | Time spent calling FasterLivePortrait for all phrase frames. Legacy mode reports zero because upstream exposes only its aggregate. |
| Neural FPS | Effective FasterLivePortrait compute throughput, not the animation or WebRTC rate. Legacy mode reports zero. |
| Frames | Rendered neural frames divided by original JoyVASA motion frames. |
| Playback FPS | Rate assigned to the decimated frame list so its duration remains synchronized with audio. |
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
Phrase 1/3 timings: tts=...s tts_wait=...ms tts_overlap=...ms prefetched=True first_byte=...ms download=...ms render=...s backend=direct-memory stride=2 motion=...ms frame_loop=...ms effective_fps=... pipeline=...ms decode=0ms frames=8/16 playback_fps=12.50 media=...s buffer=...s underrun=...ms
```

### How to interpret measurements

- With prefetch enabled, compare `TTS wait + Render` with the preceding
  phrase's `Media`. Total TTS includes work already hidden by overlap.
- If `Gap` is consistently positive, increase `PHRASE_TARGET_CHARS` so each
  generated phrase provides more playback time, or optimize the slower stage.
- If first-ready time is too long, reduce `PHRASE_FIRST_TARGET_CHARS`.
- If Neural FPS remains below the source motion rate (normally 25 FPS), full
  rendering cannot keep up with real time. Stride two needs only half as many
  neural frames and can still prepare a phrase faster despite similar Neural
  FPS.
- If total TTS or render time rises materially while overlapping, the two GPU
  workloads are contending. Compare wall time and gaps, not overlap alone.
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

## Recorded v2B benchmark

The supplied full log confirmed CUDA remained active, direct memory produced no
request-time FFmpeg output, decode stayed at zero and frame throughput was
stable across all phrases.

| Phrase | TTS | Motion | Frame loop | Neural FPS | Pipeline | Render change vs v2A.1 | Gap | Gap change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Hello!` | 670 ms | 161 ms | 1,373 ms | 11.65 | 1,534 ms | −15.8% | 0 ms | 0 ms |
| `This is my first neural streaming avatar test.` | 2,921 ms | 162 ms | 6,562 ms | 11.58 | 6,724 ms | −10.1% | 9,000 ms | −840 ms |
| `Love to hear you talking, greetings, love it!` | 2,553 ms | 168 ms | 5,703 ms | 11.57 | 5,872 ms | −9.7% | 5,400 ms | −680 ms |

Conclusion: v2B passed. It removed unnecessary transport work and delivered a
repeatable 10–16% render improvement, but the 11.6 Neural FPS frame loop remains
well below the 25 FPS motion rate. That evidence supports testing stride two.

The startup-only `CoreMLExecutionProvider` warning is harmless on Linux because
that provider is not installed; CUDA and CPU are available. The shape-merge
warnings are also unchanged from the working baseline. There was no traceback,
cuSOLVER failure or ONNX execution error.

## Recorded v2C-A benchmark

The supplied four-phrase run completed without a traceback, CUDA/ONNX error or
FFmpeg fallback. Frame counts, 12.5 Playback FPS and media durations confirm
that stride two behaved as designed.

| Phrase | TTS | Frame loop | Pipeline | Neural FPS | Frames | Media | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Hello!` | 668 ms | 670 ms | 816 ms | 11.94 | 8/16 | 0.64 s | 0 ms |
| `This is my first neural streaming avatar test.` | 2,888 ms | 3,279 ms | 3,439 ms | 11.59 | 38/76 | 3.04 s | 5,680 ms |
| `I love to hear you speaking.` | 1,935 ms | 2,159 ms | 2,292 ms | 11.58 | 25/50 | 2.00 s | 1,200 ms |
| `Thanks for joining` | 1,110 ms | 1,202 ms | 1,363 ms | 11.65 | 14/28 | 1.12 s | 460 ms |

For the two phrases directly comparable to v2B, rendering improved 46.7% and
48.8%. Phrase-two gap fell from 9.0 to 5.68 seconds. v2C-A therefore passed its
latency target; visual smoothness remains a user acceptance decision.

The gap equation now matches the logs almost exactly. For phrase two:

```text
2.888 s TTS + 3.440 s render - 0.640 s prior buffer = 5.688 s predicted gap
5.680 s measured gap
```

This confirms serial scheduling, rather than WebRTC, accounts for the remaining
pause.

## Recorded v2C-B prefetch result

The bounded prefetch behaved correctly, but the shared RTX 5080 could not run
Breeze and FasterLivePortrait efficiently at the same time.

| Phrase | TTS | TTS overlap | Frame loop | Neural FPS | Pipeline | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Hello!` | 748 ms | 0 ms | 1,404 ms | 6.41 | 1,683 ms | 0 ms |
| `This is my first neural streaming avatar test.` | 4,067 ms | 1,684 ms | 6,807 ms | 6.46 | 7,079 ms | 8,740 ms |
| long third phrase | 9,678 ms | 7,080 ms | 9,038 ms | 9.18 | 9,592 ms | 8,680 ms |
| final phrase | 4,682 ms | 4,682 ms | 2,928 ms | 11.61 | 3,053 ms | 0 ms |

Phrase four's recovery after concurrent work ended is the clearest evidence.
The experiment was rejected and prefetch now defaults off.

## Recorded v2D serial baseline before voice comparison

With prefetch disabled, the supplied run completed without CUDA, cuSOLVER,
ONNX execution or application errors. The harmless CoreML-provider and ONNX
shape warnings were unchanged.

| Phrase | TTS | First byte | Frame loop | Neural FPS | Pipeline | Media | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Hello!` | 695 ms | 208 ms | 665 ms | 12.03 | 822 ms | 0.64 s | 0 ms |
| `This is a neural streaming avatar test.` | 2,941 ms | 196 ms | 3,194 ms | 11.58 | 3,352 ms | 2.96 s | 5,660 ms |
| `I love these test.` | 1,693 ms | 195 ms | 1,796 ms | 11.69 | 1,918 ms | 1.68 s | 660 ms |
| `It makes me feel to come alive` | 2,078 ms | 194 ms | 2,252 ms | 11.54 | 2,416 ms | 2.08 s | 2,800 ms |

The gaps match `current TTS + current render − previous media` within normal
packet timing. This is the control run for the v2D voice-mode comparison.

## Recorded initial v2D voice-mode result

The first manual comparison confirmed both modes worked and rendering remained
stable near 11.5 FPS. The texts differed slightly, so total milliseconds were
not accepted as a controlled comparison, but two important behaviors were
clear:

- Designed voice first byte stayed between 190 and 211 ms.
- The first-ever preset clone request took 6,782 ms to its first byte, while
  later preset phrases returned their first bytes in 227–261 ms. This indicates
  approximately 6.5 seconds of one-time reference/audio-tokenizer cold work.
- Warm preset TTS ran at roughly 0.94× generated media duration, compared with
  roughly 0.96× for design. CFG 1 did not create a large steady-state penalty.
- Preset speech was substantially longer for comparable phrases: `Hello!` was
  1.84 seconds instead of 0.64 seconds. That increased JoyVASA frames, portrait
  renders and gaps even though per-media-second efficiency was similar.

This result motivated v2E: warm every mode separately, require identical test
text, calculate TTS/render RTF, and retain raw output automatically.

## v2D voice comparison procedure

1. Keep `TTS_PREFETCH=false`, `RENDER_STRIDE=2` and all phrase thresholds
   unchanged.
2. Confirm `/health` reports build `neural-avatar-v2f-adaptive-stride` and
   `preset_voice_configured: true` after adding the preset files.
3. Use the same text for Designed voice, Preset voice (clone), and Preset voice
   + direction. Run each mode three times without restarting services.
4. Compare Voice, CFG, TTS, First byte, Neural FPS and Gap. Also judge whether
   identity remains stable at phrase boundaries.
5. Accept preset clone as the latency default only if identity improves and its
   first-byte/total-TTS measurements are no worse than the design median.
   Otherwise retain it as a quality option and investigate reference-token
   caching separately.

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

### v2C-A — render alternating motion frames

- **Observed v2B result:** direct mode improved rendering 10–16%, but the neural
  frame loop remained stable at only 11.57–11.65 FPS and later gaps remained
  9.0 and 5.4 seconds.
- **Decision:** default `RENDER_STRIDE=2`, render half of the 25 FPS motion
  sequence, and label those frames as 12.5 FPS before WebRTC resampling.
- **Reason:** frame inference is the measured bottleneck. Halving only that work
  is the smallest reversible experiment likely to make a large difference.
- **Tradeoff:** held frames can make fast head, eye or mouth motion look less
  smooth even though duration and audio synchronization remain unchanged.
- **Intentionally deferred:** TTS prefetch, frame interpolation, incremental
  append, TensorRT and Ditto.
- **Rollback:** set `RENDER_STRIDE=1` and recreate `webrtc-avatar`; no rebuild is
  required.

### v2C-B — prefetch one future TTS phrase

- **Observed v2C-A result:** stride two cut comparable render time by 46.7–48.8%,
  but serial `TTS + render − buffer` still explained every remaining gap.
- **Experiment:** start only phrase N+1 TTS while rendering phrase N and
  measure total TTS, blocking wait and hidden overlap separately.
- **Reason:** later TTS calls can fit entirely inside the preceding long render
  in the measured four-phrase workload, potentially eliminating later gaps.
- **Risk:** both services share one GPU. Concurrent kernels or memory pressure
  can slow first-ready/render/TTS or cause CUDA failure.
- **Bound:** prefetch depth is fixed at one; multiple simultaneous Breeze
  requests are never issued, and unfinished work is cancelled on failure.
- **Measured result:** phrase-one and phrase-two neural FPS fell to 6.41 and
  6.46 while comparable serial runs sustained about 12.03 and 11.58 FPS.
  Phrase-two render rose to 7.079 seconds. Phrase four recovered to 11.61 FPS
  after overlap ended.
- **Decision:** reject shared-GPU prefetch and make `TTS_PREFETCH=false` the v2D
  default. It can still be enabled for diagnostics.

### v2D — selectable designed and preset voices

- **Observed serial result:** with prefetch disabled, neural FPS recovered to
  11.54–12.03 and the four measured gaps matched
  `current TTS + current render − previous buffered media`.
- **Observed voice behavior:** reference-free Voice Design starts a separate
  sampled generation for every phrase; a fixed description and seed do not
  provide persistent speaker identity.
- **Decision:** expose `design`, `preset-clone` and `preset-direction` without
  allowing the browser to choose arbitrary files.
- **Reason:** a fixed reference WAV and exact transcript should stabilize
  identity. Clone mode also permits CFG 1, while design/direction retain CFG 4.
- **Measurement:** add Voice and CFG to every phrase row and log entry; compare
  first byte and total TTS because the stock Breeze API re-encodes reference
  audio on each request.
- **Intentionally deferred:** Breeze reference-token caching, simultaneous
  TTS/render work, phrase-boundary motion blending and incremental PCM/motion
  streaming.
- **Rollback:** select Designed voice in the UI or set
  `BREEZE_DEFAULT_VOICE_MODE=design`; the v2C-A serial render path is unchanged.

### v2E — repeatable headless WebRTC benchmark

- **Observed problem:** manually entered test sentences differed between voice
  modes, the first preset request included a one-time cold penalty, and total
  TTS time was misleading when cloned speech duration changed.
- **Decision:** generate a deterministic Breeze reference fixture and drive the
  same `/offer`, media-track and data-channel path as the browser.
- **Measurement:** warm each mode explicitly, rotate measured modes per repeat,
  calculate TTS/render RTF and weighted neural FPS, and store JSON, CSV and
  Markdown artifacts with the complete `/health` configuration.
- **Scenario control:** Compose exposes stride, prefetch and phrase thresholds
  through namespaced host variables; the default matrix retains stride two and
  prefetch false.
- **Safety:** existing preset files are never overwritten unless
  `REGENERATE_PRESET=1` is explicitly supplied. Incomplete preset pairs fail
  instead of guessing.
- **Intentionally deferred:** automated perceptual speaker similarity, visual
  lip-sync scoring, reference-token caching and incremental neural streaming.

### v2E.1 — benchmark readiness handshake

- **Observed problem:** WebRTC and the local data channel connected, but the
  benchmark timed out waiting 30 seconds for the server's initial `ready`
  message.
- **Cause:** for a remotely created aiortc data channel, the server can receive
  the channel after it is already open. Attaching only an `open` handler at that
  point misses the transition.
- **Fix:** the server checks the current channel state as well as listening for
  `open`, with a guard preventing duplicate `ready` messages. The benchmark
  also accepts its own open channel after a two-second compatibility wait.
- **Scope:** this changes only connection setup; benchmark text, warm-ups,
  metrics, voice modes and renderer settings remain identical to v2E.

### v2F — adaptive stride 2→3

- **Measured control:** fixed stride 2 produced render RTF 1.146, first-ready
  1.146 seconds and 9.42 seconds of gaps across three repeatable design runs.
- **Measured alternative:** fixed stride 3 reduced render RTF to 0.781 and gaps
  to 7.34 seconds while neural throughput remained stable near 11.8 rendered
  frames per second. Stride 4 reduced gaps further to 6.32 seconds but playback
  fell to 6.25 FPS.
- **Decision:** retain stride 2 for phrase 1 and a healthy media queue; select
  stride 3 below 0.75 seconds buffered. This takes the best measured speed gain
  without making stride 4's lower temporal quality the default.
- **Measurement:** every phrase reports whether adaptation was enabled, buffer
  seconds before render, selected stride, and decision reason. Benchmark
  scenarios include base stride, adaptive state and catch-up stride.
- **Rollback:** set `ADAPTIVE_RENDER_STRIDE=false`; fixed stride behavior is
  unchanged.
- **Known limit:** TTS remains near 1.0 RTF, so v2F can shorten but cannot
  eliminate gaps. True continuous delivery still needs a streaming renderer
  such as Ditto and/or substantially faster TTS.

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

### Benchmark times out in `AvatarBenchmarkClient.connect`

Confirm `/health` reports
`server_build: neural-avatar-v2f-adaptive-stride`. If it reports v2E.1,
the old image is still running. Rebuild and force-recreate the avatar. v2E.1
does not require a server `ready` event once the client data channel is open.

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
media playback. Longer phrases can amortize fixed overhead, but cannot solve a
sustained production ratio above 1.0. v2D keeps prefetch disabled because the
shared-GPU overlap made total work slower. Incremental PCM/motion/frame delivery
or an online renderer remains necessary for truly continuous playback.

### Prefetch makes performance worse or causes a CUDA error

Set `TTS_PREFETCH: "false"` and force-recreate `webrtc-avatar`. Compare total
TTS, TTS wait, TTS overlap, render time and Neural FPS. A high overlap value is
not a win if GPU contention increases the combined wall time or destabilizes
the driver.

### Preset voice options say `not configured`

Both `inputs/voice-preset.wav` and `inputs/voice-preset.txt` must exist, and the
transcript file cannot be empty. Confirm from the project directory:

```bash
test -s inputs/voice-preset.wav
test -s inputs/voice-preset.txt
docker compose up -d --force-recreate webrtc-avatar
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Look for `preset_voice_configured: true`. Do not use an approximate transcript;
it must match the spoken reference.

### Preset voice is consistent but slower

Compare `First byte` and `TTS` against Designed voice with identical text.
Reference audio is encoded for every phrase in this version. If first-byte time
increases materially while CFG 1 does not compensate during generation, the
next isolated experiment is a stable preset ID with cached reference tokens in
the Breeze service. Do not re-enable TTS prefetch during this comparison.

### Backend says `legacy-mp4`

Confirm both values in Compose:

```yaml
AVATAR_PASTE_BACK: "false"
DIRECT_MEMORY_RENDER: "true"
```

Then force-recreate `webrtc-avatar`. Paste-back deliberately selects the legacy
backend because direct-memory rendering supports crop frames only.

### Motion is visibly stepped

Set `RENDER_STRIDE: "1"` and force-recreate `webrtc-avatar`. This restores the
exact v2B frame selection without disabling direct memory. Keep both timing
tables so the smoothness/latency tradeoff can be evaluated explicitly.

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

After the v2E benchmark baseline, the remaining FasterLivePortrait experiment is
phrase-boundary blending followed by incremental frame append so playback can
begin before a complete phrase is rendered. That requires a thread-safe bounded
frame queue and careful audio start timing; it should remain separate from the
voice measurement.

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
