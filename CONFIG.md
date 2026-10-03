# Configuration

Every setting of the stack: its name, default and meaning. The avatar service's
settings are defined in one place, `src/avatar/config.py`; the tables for it below
mirror that file.

## How it works

- **One source of defaults.** The defaults are in the code. `docker-compose.yml`
  holds no values: it only passes on what you set in the shell or in `.env`.
  An empty `.env` is the tested live setup.
- **One name per setting.** The name is the same in `.env`, in the shell and in
  the container. Avatar settings start with `AVATAR_`.
- **Profiles.** `AVATAR_PROFILE` picks a named set of values; single settings
  can still be overridden on top of it.
- **Checked at start-up.** The avatar refuses to start, and lists every
  problem, when a name is unknown, renamed or removed, a value is out of range,
  or a combination is impossible. A setting that is set but cannot have any
  effect gives a warning in the log and in `/health`.
- **Traceable.** `/health`, every benchmark result and every conversation log
  record the profile, the overrides and a hash of the behaviour settings. Equal
  hashes mean equal render, timing and motion settings. A run with a diagnosis
  switch changed is marked as an experiment.

```bash
python3 src/avatar/config.py show       # effective settings, overrides, warnings
python3 src/avatar/config.py profiles   # what each profile changes
tools/config_snapshot.py check      # running stack against config/reference/
```

## Profiles

| Profile | Changes against the defaults |
|---|---|
| `live` | nothing: the defaults are this profile |
| `development` | `AVATAR_TENSORRT=false`, `AVATAR_DEBUG_DUMP_DIR=/workspace/results/motion-dumps` |
| `benchmark-runtime` | `AVATAR_TTS_PREFETCH=false` |
| `benchmark-visual` | `AVATAR_TTS_PREFETCH=false`, `AVATAR_RENDER_STRIDE=1`, `AVATAR_ADAPTIVE_RENDER_STRIDE=false`, `AVATAR_INCREMENTAL_FRAME_WINDOWS=false` |

`live` is the default. `development` starts without building a TensorRT engine
and keeps every phrase's audio and motion for analysis. The benchmark profiles
run phrases serially so timings are comparable: `benchmark-runtime` measures
the live render path, `benchmark-visual` renders every frame of a whole phrase
before playing it. `scripts/quality_benchmark.sh` and `scripts/benchmark.sh` use them.

## Avatar: behaviour settings

These define how the avatar renders, times and moves, and are part of the
configuration hash.

| Variable | Default | Meaning |
|---|---|---|
| `AVATAR_RENDER_STRIDE` | 1 | Render every Nth of the 25 motion frames per second; 1 = all. |
| `AVATAR_ADAPTIVE_RENDER_STRIDE` | true | Switch to the catch-up stride while little media is buffered. |
| `AVATAR_CATCHUP_RENDER_STRIDE` | 2 | Stride used while the buffer is low. Must be >= the render stride. |
| `AVATAR_CATCHUP_BUFFER_SECONDS` | 0.75 | Below this much buffered media the catch-up stride is used. |
| `AVATAR_INCREMENTAL_FRAME_WINDOWS` | true | Publish frames per window instead of per whole phrase. |
| `AVATAR_RENDER_WINDOW_FRAMES` | 8 | Frames per window; smaller starts sooner, larger absorbs GPU jitter. |
| `AVATAR_SPEECH_START_GATE` | true | Hold a phrase's speech until enough video exists to stay in sync. |
| `AVATAR_SPEECH_START_SAFETY` | 1.15 | Safety factor on that hold. |
| `AVATAR_EXPECTED_RENDER_FPS` | 9.0 | Render speed assumed until the first measurement. |
| `AVATAR_TENSORRT` | true | Render warping_spade with TensorRT FP16 (~27 ms instead of ~97 ms per frame). |
| `AVATAR_TTS_PREFETCH` | true | Synthesize the next phrase while the current one renders. |
| `AVATAR_TTS_PREFETCH_POLICY` | adaptive | `adaptive` waits for the first video window and a minimum buffer; `eager` starts at once. |
| `AVATAR_TTS_PREFETCH_MIN_BUFFER_SECONDS` | 0.5 | Buffer the adaptive policy requires. |
| `AVATAR_PHRASE_FIRST_TARGET_CHARS` | 48 | Target length of the first phrase (short, so playback starts soon). |
| `AVATAR_PHRASE_MIN_FIRST_CHARS` | 24 | A first phrase shorter than this is merged with the second. |
| `AVATAR_PHRASE_TARGET_CHARS` | 100 | Target length of later phrases. |
| `AVATAR_PHRASE_MAX_CHARS` | 160 | Hard limit per phrase. Must be >= the target. |
| `AVATAR_MERGE_SHORT_OPENING_PHRASE` | true | Merge a tiny opening ("Hello!") with the next phrase. |
| `AVATAR_DEFAULT_VOICE_MODE` | preset-clone | `preset-clone` clones the reference recording; `design` is Chatterbox's built-in voice. |
| `AVATAR_EYE_MOTION_SCALE` | 0.3 | JoyVASA eye motion: 1.0 as generated, 0.0 the portrait's own eyes. |
| `AVATAR_HEAD_MOTION_SCALE` | 0.3 | JoyVASA head rotation and translation around the first pose. |
| `AVATAR_LIP_MOTION_MODE` | absolute | `absolute` uses JoyVASA's mouth shapes directly; `relative` applies their change to the portrait's lips. |
| `AVATAR_LIP_MOTION_SCALE` | 1.0 | Mouth opening in `relative` lip mode only. |
| `AVATAR_LIP_SYNC_OFFSET_MS` | 80.0 | Positive shows the mouth later than the voice. |
| `AVATAR_AUDIO_KEEPALIVE_DBFS` | -60.0 | Inaudible noise between sentences so speaker amplifiers stay awake; `off` sends silence. |
| `AVATAR_DEBUG_DUMP_DIR` | (empty) | Save every phrase's WAV and motion here; empty disables it. |

## Avatar: diagnosis and rollback switches

Not part of normal use. Changing one marks the run as an experiment.

| Variable | Default | Meaning |
|---|---|---|
| `AVATAR_PERSISTENT_PHRASE_MOTION` | true | Keep one motion reference for all phrases of an utterance. |
| `AVATAR_RELATIVE_MOTION` | true | FLP relative motion; false disables the eye, head and lip settings. |
| `AVATAR_ANIMATION_REGION` | all | Which part FLP animates (quality matrix). |
| `AVATAR_DRIVING_MULTIPLIER` | 1.0 | FLP driving multiplier (quality matrix). |
| `AVATAR_NORMALIZE_LIP` | true | FLP lip normalisation (quality matrix). |
| `AVATAR_EYE_RETARGETING` | false | FLP eye retargeting (quality matrix). |
| `AVATAR_LIP_RETARGETING` | false | FLP lip retargeting (quality matrix). |
| `AVATAR_CROP_ROTATION` | false | Let FLP rotate the crop upright (tilts the frame, black wedges). |
| `AVATAR_USE_NEURAL_IDLE_FRAME` | true | Show a rendered frame while idle instead of the photo. |
| `AVATAR_WARMUP_IDLE_FRAME_INDEX` | 0 | Which warm-up frame becomes the idle image. |

## Avatar: deployment

Where things are. Not part of the configuration hash.

| Variable | Default | Meaning |
|---|---|---|
| `AVATAR_IMAGE_PATH` | /workspace/inputs/avatar.jpg | The portrait. |
| `AVATAR_PRESET_AUDIO_PATH` | /workspace/inputs/voice-preset-nohello.wav | Reference recording of the cloned voice. |
| `AVATAR_FLP_CONFIG_PATH` | /workspace/FasterLivePortrait/configs/onnx_infer.yaml | FasterLivePortrait configuration. |
| `AVATAR_RESULTS_ROOT` | /workspace/results | Where temporary and result files go. |
| `AVATAR_TENSORRT_CACHE` | /workspace/trt-cache | TensorRT engine cache. |
| `AVATAR_TTS_URL` | http://127.0.0.1:7860 | The TTS service. |
| `AVATAR_LISTENER_SOCKET` | /run/avatar/listener.sock | Listener worker socket; empty disables listening. |
| `AVATAR_CONDUCTOR_SOCKET` | /run/avatar/conductor.sock | Conductor socket; empty disables conversation. |
| `AVATAR_PORT` | 8000 | HTTP port. |
| `AVATAR_HTTPS_PORT` | 8443 | HTTPS port (microphone from other machines); empty disables HTTPS. |
| `AVATAR_HTTPS_CERT_DIR` | /workspace/certs | Where the self-signed certificate is kept. |
| `AVATAR_HTTPS_CERT_HOSTS` | (empty) | Names/addresses in the certificate; empty = this host's addresses. |
| `AVATAR_ICE_SERVERS_JSON` | [] | TURN/STUN servers for remote clients, as a JSON array. |
| `AVATAR_LOG_LEVEL` | INFO | Log level. |

## Settings without effect in some combinations

The avatar warns when one of these is set while its switch is off:

- Eye and head scale and the lip mode need `AVATAR_RELATIVE_MOTION=true`.
- `AVATAR_LIP_MOTION_SCALE` needs `AVATAR_LIP_MOTION_MODE=relative`.
- The prefetch policy needs `AVATAR_TTS_PREFETCH=true`; the minimum buffer also
  needs the `adaptive` policy.
- The catch-up stride and buffer need `AVATAR_ADAPTIVE_RENDER_STRIDE=true`.
- The speech start gate, its safety factor and the window size need
  `AVATAR_INCREMENTAL_FRAME_WINDOWS=true`.
- The idle frame index needs `AVATAR_USE_NEURAL_IDLE_FRAME=true`.
- `AVATAR_PHRASE_MIN_FIRST_CHARS` needs `AVATAR_MERGE_SHORT_OPENING_PHRASE=true`.

## TTS (`src/tts/chatterbox_api.py`, `docker/entrypoint.sh`, build arguments)

| Variable | Default | Class |
|---|---|---|
| `CHATTERBOX_VARIANT` | turbo | deployment |
| `CHATTERBOX_DEVICE` | cuda | deployment |
| `CHATTERBOX_LANGUAGE` | en | profile |
| `CHATTERBOX_EXAGGERATION` | 0.5 | profile |
| `CHATTERBOX_CFG_WEIGHT` | 0.5 | profile |
| `CHATTERBOX_PORT` | 7860 | deployment |
| `CHATTERBOX_VERSION` (build) | 0.1.7 | deployment |

`CHATTERBOX_LANGUAGE` applies to the multilingual variant only, and
`CHATTERBOX_EXAGGERATION` and `CHATTERBOX_CFG_WEIGHT` to the `original` variant
only.

## Listener (`src/listener/worker.py`)

| Variable | Default | Class |
|---|---|---|
| `LISTENER_SOCKET` | /run/avatar/listener.sock | deployment |
| `LISTENER_ASR_MODEL` | nvidia/nemotron-3.5-asr-streaming-0.6b | deployment |
| `LISTENER_DIAR_MODEL` | nvidia/Nemotron-3-Diarization | deployment |
| `LISTENER_AUDIO_DIR` | results/conversations/audio | deployment |
| `LISTENER_SAVE_AUDIO` | false | deployment |
| `LISTENER_ASR_LOOKAHEAD` | 3 | profile |
| `LISTENER_ASR_LANGUAGE` | en-US | profile |
| `LISTENER_DIAR_MODE` | very_low_latency | profile |
| `LISTENER_VAD_THRESHOLD` | 0.5 | profile |
| `LISTENER_MIN_SILENCE_MS` | 300 | profile |
| `LISTENER_PREROLL_MS` | 300 | profile |
| `LISTENER_FLUSH_MS` | 700 | profile |

## Conductor (`src/conductor/`: `app.py`, `system1.py`, `system2.py`, `knowledge.py`, `review.py`)

| Variable | Default | Class |
|---|---|---|
| `CONDUCTOR_SOCKET` | /run/avatar/conductor.sock | deployment |
| `CONVERSATION_LOG_DIR` | results/conversations | deployment |
| `SYSTEM1_CONFIG` | config/system1.json | deployment |
| `SYSTEM1_MEMORY` | results/system1/corrections.jsonl | deployment |
| `SYSTEM1_MODEL` | JevK5 4B v0.3 Q4_K_M | deployment |
| `SYSTEM2_MODEL` | Llama 3.2 3B Q4_K_M | deployment |
| `KNOWLEDGE_DIR` | /workspace/knowledge | deployment |
| `KNOWLEDGE_EMBED_MODEL` | intfloat/multilingual-e5-small | deployment |
| `REVIEW_RESULTS` | /workspace/results | deployment |
| `REVIEW_MODEL` | Qwen3 4B Q4_K_M | deployment |
| `CONDUCTOR_RESPONDER` | system1 | profile |
| `SYSTEM2_ENABLED` | true | profile |
| `TURN_SILENCE_MS` | 700 | profile |
| `TURN_END_PUNCTUATION_MS` | 400 | profile |
| `TURN_CONNECTIVE_MS` | 1200 | profile |
| `TURN_CONNECTIVES` | and,but,…,als | profile |
| `LISTENER_MIN_SILENCE_MS` | 300 | profile |
| `PTT_FINAL_WAIT_MS` | 1500 | profile |
| `SYSTEM2_MAX_TOKENS` | 120 | profile |
| `REVIEW_UNSURE_BELOW` | 0.6 | profile |
| `REVIEW_REPEAT_SIMILARITY` | 0.88 | profile |

The listener, conductor and TTS settings have one name and agreeing defaults;
the problems above are concentrated in the avatar service.

`LISTENER_MIN_SILENCE_MS` is read by both the listener and the conductor and
must be smaller than the three `TURN_*_MS` values; the conductor checks this at
start-up. These three services keep their existing names and read their
settings in their own modules; their defaults were already consistent.

## Not service configuration

- `config/system1.json` is domain content (classes, reactions) with hot reload.
- `BENCHMARK_*`, `QUALITY_BENCHMARK_*`, `REVIEW_FPS` and `REVIEW_MODE` are
  arguments of the host scripts.

## History of the clean-up

Before this (v2W) the avatar read 60 variables with defaults in four places
that disagreed, most with a different name on the host and in the container.

- Renamed: container-side names without a prefix got the host name they
  already had (`RENDER_STRIDE` → `AVATAR_RENDER_STRIDE`, `TTS_PREFETCH` →
  `AVATAR_TTS_PREFETCH`, `HTTPS_PORT` → `AVATAR_HTTPS_PORT`, and so on). Also
  `TTS_DEFAULT_VOICE_MODE` → `AVATAR_DEFAULT_VOICE_MODE`, `TTS_URL` →
  `AVATAR_TTS_URL`, `AVATAR_PATH` → `AVATAR_IMAGE_PATH`, `LOG_LEVEL` →
  `AVATAR_LOG_LEVEL` (avatar only). The full list is `RENAMED` in
  `src/avatar/config.py`; an old name stops the avatar with a message naming the
  new one.
- Became code constants (never varied in practice): `STARTUP_WARMUP`, `WARMUP_TEXT`, `TTS_STARTUP_WAIT_SECONDS`, `TTS_STARTUP_POLL_SECONDS`, `MAX_TEXT_LENGTH`, `JOYVASA_CFG_SCALE`, `BREEZE_SEED`, `CHATTERBOX_MAX_REFERENCE_BYTES`, `SYSTEM1_TEMPERATURE`.
- Dead flags that only kept the legacy MP4 render route alive:
  `AVATAR_PASTE_BACK`, `DIRECT_MEMORY_RENDER`, `PROGRESSIVE_PHRASE_MODE`. The
  legacy renderer went with them.
- Breeze TTS (too heavy to share the GPU with the renderer; unused since
  Chatterbox became the default in v2L): `TTS_PROVIDER`, `BREEZE_TTS_URL`, `TTS_HEALTH_PATH`, `BREEZE_DEFAULT_VOICE_MODE`, `BREEZE_VOICE_INSTRUCTION`, `BREEZE_DESIGN_CFG_SCALE`, `BREEZE_CFG_SCALE`, `BREEZE_PRESET_CLONE_CFG_SCALE`, `BREEZE_PRESET_DIRECTION_CFG_SCALE`, `BREEZE_PRESET_CLONE_INSTRUCTION`, `BREEZE_PRESET_TRANSCRIPT`, `BREEZE_PRESET_TRANSCRIPT_FILE`, `BREEZE_FAST_ARGS`, `BREEZE_MODEL_DIR`, `BREEZE_PORT`, `BREEZE_INSTALL_FLASH_ATTN`, `FLASH_ATTN_CUDA_ARCHS`, `BREEZE_REF` (build).
  With it went the voice direction, the CFG scales, the preset transcript and
  the `preset-direction` voice mode; `design` (Chatterbox's built-in voice) and
  `preset-clone` remain.
- `.env.before-facial-test`. `.env.example` no longer repeats defaults.

The avatar now has 26 behaviour settings, 10 diagnosis switches and
14 deployment settings, each with one name and one default.
