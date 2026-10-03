# Configuration inventory

Every environment variable the stack reads, who owns it, where its default
comes from, and what kind of setting it is. This is the working document of the
configuration refactoring; `tools/config_snapshot.py check` verifies that the
running stack still matches `config/reference/`.

State: commit `1baa50c` (v2W), before pruning. The **Decision** column says
what happens to the variable.

## Why this exists

- Defaults live in four places that disagree: the Python code, the
  `${VAR:-default}` expressions in `docker-compose.yml`, `.env.example`, and the
  local (gitignored) `.env`. Running `server.py` outside Compose gives a setup
  the README calls outdated (stride 2, no TensorRT, no prefetch, full eye and
  head motion, lip offset 0).
- Most avatar settings have two names: `AVATAR_X` on the host, `X` in the
  container.
- Combinations are not validated; 13 values are silently corrected with
  `max(...)`.

## Classes

| Class | Meaning | Where it ends up |
|---|---|---|
| **profile** | Changes render, timing or motion behaviour | A named profile; override allowed and reported |
| **experiment** | Diagnosis or rollback switch, not part of normal use | Settings class, reported loudly when changed |
| **deployment** | Paths, ports, sockets, URLs, model names | Compose / `.env` |
| **constant** | Never varied in practice | Code constant, no environment variable |
| **dead** | Only one value is ever used, or it serves a removed feature | Removed |

## Avatar (`server.py`, `serve.py`, `entrypoint.sh`)

"Code" is the default when the variable is absent; "Compose" is what the
container actually gets. **Bold** marks a disagreement.

| Container name | Host name | Code | Compose | Class | Decision |
|---|---|---|---|---|---|
| `RENDER_STRIDE` | `AVATAR_RENDER_STRIDE` | **2** | **1** | profile | keep |
| `ADAPTIVE_RENDER_STRIDE` | `AVATAR_ADAPTIVE_RENDER_STRIDE` | true | true | profile | keep |
| `CATCHUP_RENDER_STRIDE` | `AVATAR_CATCHUP_RENDER_STRIDE` | **3** | **2** | profile | keep; must be >= render stride |
| `CATCHUP_BUFFER_SECONDS` | `AVATAR_CATCHUP_BUFFER_SECONDS` | 0.75 | 0.75 | profile | keep |
| `INCREMENTAL_FRAME_WINDOWS` | `AVATAR_INCREMENTAL_FRAME_WINDOWS` | true | true | profile | keep (`benchmark-visual` turns it off) |
| `RENDER_WINDOW_FRAMES` | `AVATAR_RENDER_WINDOW_FRAMES` | 8 | 8 | profile | keep |
| `SPEECH_START_GATE` | `AVATAR_SPEECH_START_GATE` | true | true | profile | keep |
| `SPEECH_START_SAFETY` | `AVATAR_SPEECH_START_SAFETY` | 1.15 | 1.15 | profile | keep |
| `EXPECTED_RENDER_FPS` | `AVATAR_EXPECTED_RENDER_FPS` | 9.0 | 9.0 | profile | keep (only the estimate before the first measurement) |
| `AVATAR_TENSORRT` | same | **false** | **true** | profile | keep |
| `TTS_PREFETCH` | `AVATAR_TTS_PREFETCH` | **false** | **true** | profile | keep (`.env.example` says false) |
| `TTS_PREFETCH_POLICY` | `AVATAR_TTS_PREFETCH_POLICY` | adaptive | adaptive | profile | keep; only meaningful with prefetch on |
| `TTS_PREFETCH_MIN_BUFFER_SECONDS` | `AVATAR_TTS_PREFETCH_MIN_BUFFER_SECONDS` | 0.50 | 0.50 | profile | keep; adaptive policy only |
| `PHRASE_FIRST_TARGET_CHARS` | `AVATAR_PHRASE_FIRST_TARGET_CHARS` | 48 | 48 | profile | keep |
| `PHRASE_MIN_FIRST_CHARS` | `AVATAR_PHRASE_MIN_FIRST_CHARS` | 24 | 24 | profile | keep |
| `PHRASE_TARGET_CHARS` | `AVATAR_PHRASE_TARGET_CHARS` | 100 | 100 | profile | keep |
| `PHRASE_MAX_CHARS` | `AVATAR_PHRASE_MAX_CHARS` | 160 | 160 | profile | keep; must be >= target |
| `MERGE_SHORT_OPENING_PHRASE` | `AVATAR_MERGE_SHORT_OPENING_PHRASE` | true | true | profile | keep |
| `AVATAR_EYE_MOTION_SCALE` | same | **1.0** | **0.3** | profile | keep; needs relative motion |
| `AVATAR_HEAD_MOTION_SCALE` | same | **1.0** | **0.3** | profile | keep; needs relative motion |
| `AVATAR_LIP_MOTION_MODE` | same | absolute | absolute | profile | keep; needs relative motion |
| `AVATAR_LIP_MOTION_SCALE` | same | 1.0 | 1.0 | profile | keep; only in `relative` lip mode |
| `AVATAR_LIP_SYNC_OFFSET_MS` | same | **0** | **80** | profile | keep |
| `AVATAR_AUDIO_KEEPALIVE_DBFS` | same | -60 | -60 | profile | keep |
| `TTS_DEFAULT_VOICE_MODE` | same | preset-clone | preset-clone | profile | keep |
| `AVATAR_DEBUG_DUMP_DIR` | same | (empty) | (empty) | profile | keep (`development` turns it on) |
| `PERSISTENT_PHRASE_MOTION` | `AVATAR_PERSISTENT_PHRASE_MOTION` | true | true | experiment | keep (v2H rollback) |
| `AVATAR_RELATIVE_MOTION` | same | true | true | experiment | keep; false disables all eye/head/lip settings |
| `AVATAR_ANIMATION_REGION` | same | all | all | experiment | keep (quality matrix) |
| `AVATAR_DRIVING_MULTIPLIER` | same | 1.0 | 1.0 | experiment | keep (quality matrix) |
| `AVATAR_NORMALIZE_LIP` | same | true | true | experiment | keep (quality matrix) |
| `AVATAR_EYE_RETARGETING` | same | false | false | experiment | keep (quality matrix) |
| `AVATAR_LIP_RETARGETING` | same | false | false | experiment | keep (quality matrix) |
| `AVATAR_CROP_ROTATION` | same | false | false | experiment | keep |
| `USE_NEURAL_IDLE_FRAME` | `AVATAR_USE_NEURAL_IDLE_FRAME` | true | true | experiment | keep |
| `WARMUP_IDLE_FRAME_INDEX` | `AVATAR_WARMUP_IDLE_FRAME_INDEX` | 0 | 0 | experiment | keep |
| `AVATAR_PATH` | – | inputs/avatar.jpg | same | deployment | keep |
| `FLP_CONFIG_PATH` | – | configs/onnx_infer.yaml | same | deployment | keep |
| `RESULTS_ROOT` | – | /workspace/results | (unset) | deployment | keep |
| `AVATAR_TENSORRT_CACHE` | – | /workspace/trt-cache | same | deployment | keep |
| `TTS_URL` | same | http://127.0.0.1:7860 | same | deployment | keep |
| `BREEZE_PRESET_AUDIO_PATH` | `AVATAR_PRESET_AUDIO_PATH` | **voice-preset.wav** | **voice-preset-nohello.wav** | deployment | keep, renamed to a neutral name |
| `LISTENER_SOCKET` | `AVATAR_LISTENER_SOCKET` | (empty = off) | /run/avatar/listener.sock | deployment | keep |
| `CONDUCTOR_SOCKET` | `AVATAR_CONDUCTOR_SOCKET` | (empty = off) | /run/avatar/conductor.sock | deployment | keep |
| `ICE_SERVERS_JSON` | – | [] | [] | deployment | keep |
| `PORT` | – | 8000 | (unset) | deployment | keep |
| `HTTPS_PORT` | `AVATAR_HTTPS_PORT` | (empty = off) | 8443 | deployment | keep |
| `HTTPS_CERT_DIR` | – | /workspace/certs | same | deployment | keep |
| `HTTPS_CERT_HOSTS` | – | host addresses | (unset) | deployment | keep |
| `HTTPS_CERT_FILE`, `HTTPS_KEY_FILE` | – | set by `entrypoint.sh` | – | deployment | keep (internal) |
| `FLP_CHECKPOINT_DIR` | – | FLP checkpoints | (unset) | deployment | keep |
| `LOG_LEVEL` | – | INFO | (unset) | deployment | keep (all services) |
| `STARTUP_WARMUP` | – | true | "true" (fixed) | constant | always warm up |
| `WARMUP_TEXT` | – | Hello. | "Hello." (fixed) | constant | |
| `TTS_STARTUP_WAIT_SECONDS` | – | 300 | "300" (fixed) | constant | |
| `TTS_STARTUP_POLL_SECONDS` | – | 2 | "2" (fixed) | constant | |
| `MAX_TEXT_LENGTH` | – | 500 | (unset) | constant | |
| `JOYVASA_CFG_SCALE` | – | 2.8 | (unset) | constant | |
| `BREEZE_SEED` | – | 42 | (unset) | constant | becomes `TTS_SEED = 42` |
| `AVATAR_PASTE_BACK` | – | false | "false" (fixed) | dead | remove with the legacy render route |
| `DIRECT_MEMORY_RENDER` | – | true | "true" (fixed) | dead | remove with the legacy render route |
| `PROGRESSIVE_PHRASE_MODE` | – | true | "true" (fixed) | dead | remove |
| `TTS_PROVIDER` | same | chatterbox | chatterbox | dead | Breeze removed; Chatterbox is the only provider |
| `BREEZE_TTS_URL` | – | fallback for `TTS_URL` | (unset) | dead | Breeze |
| `TTS_HEALTH_PATH` | – | /health (/docs for Breeze) | (unset) | dead | constant `/health` |
| `BREEZE_DEFAULT_VOICE_MODE` | – | fallback | (unset) | dead | Breeze |
| `BREEZE_VOICE_INSTRUCTION` | – | "A warm, clear…" | fixed | dead | Chatterbox ignores it |
| `BREEZE_DESIGN_CFG_SCALE`, `BREEZE_CFG_SCALE` | – | 4 | "4" (fixed) | dead | Chatterbox ignores it |
| `BREEZE_PRESET_CLONE_CFG_SCALE` | – | 1 | "1" (fixed) | dead | Chatterbox ignores it |
| `BREEZE_PRESET_DIRECTION_CFG_SCALE` | – | 4 | "4" (fixed) | dead | Breeze-only voice mode |
| `BREEZE_PRESET_CLONE_INSTRUCTION` | – | "Speak naturally…" | fixed | dead | Chatterbox ignores it |
| `BREEZE_PRESET_TRANSCRIPT` | – | (empty) | "" (fixed) | dead | only Breeze needs the transcript |
| `BREEZE_PRESET_TRANSCRIPT_FILE` | `AVATAR_PRESET_TRANSCRIPT_FILE` | voice-preset.txt | voice-preset-nohello.txt | dead | only Breeze needs the transcript |

## TTS (`chatterbox_api.py`, `entrypoint.sh`, build arguments)

| Name | Code | Compose | Class | Decision |
|---|---|---|---|---|
| `CHATTERBOX_VARIANT` | turbo | turbo | deployment | keep (model choice) |
| `CHATTERBOX_DEVICE` | cuda | cuda | deployment | keep |
| `CHATTERBOX_LANGUAGE` | en | en | profile | keep (multilingual variant only) |
| `CHATTERBOX_EXAGGERATION` | 0.5 | 0.5 | profile | keep (`original` variant only) |
| `CHATTERBOX_CFG_WEIGHT` | 0.5 | 0.5 | profile | keep (`original` variant only) |
| `CHATTERBOX_MAX_REFERENCE_BYTES` | 20000000 | (unset) | constant | |
| `CHATTERBOX_PORT` | 7860 | (unset) | deployment | keep |
| `CHATTERBOX_VERSION` (build) | 0.1.7 | 0.1.7 | deployment | keep |
| `TTS_PROVIDER` | – | chatterbox | dead | Breeze |
| `BREEZE_FAST_ARGS` | – | fast-path flags | dead | Breeze |
| `BREEZE_MODEL_DIR`, `BREEZE_PORT` | entrypoint | (unset) | dead | Breeze |
| `BREEZE_INSTALL_FLASH_ATTN`, `FLASH_ATTN_CUDA_ARCHS`, `BREEZE_REF` (build) | – | 0 / 120 / main | dead | Breeze |

## Listener (`listener_worker.py`)

| Name | Code | Compose | Class | Decision |
|---|---|---|---|---|
| `LISTENER_SOCKET` | /run/avatar/listener.sock | same | deployment | keep |
| `LISTENER_ASR_MODEL` | nvidia/nemotron-3.5-asr-streaming-0.6b | (unset) | deployment | keep |
| `LISTENER_DIAR_MODEL` | nvidia/Nemotron-3-Diarization | (unset) | deployment | keep |
| `LISTENER_AUDIO_DIR` | results/conversations/audio | (unset) | deployment | keep |
| `LISTENER_SAVE_AUDIO` | false | false | deployment | keep (privacy opt-in) |
| `LISTENER_ASR_LOOKAHEAD` | 3 | 3 | profile | keep |
| `LISTENER_ASR_LANGUAGE` | en-US | (unset) | profile | keep |
| `LISTENER_DIAR_MODE` | very_low_latency | same | profile | keep |
| `LISTENER_VAD_THRESHOLD` | 0.5 | (unset) | profile | keep |
| `LISTENER_MIN_SILENCE_MS` | 300 | 300 | profile | keep; shared with the conductor |
| `LISTENER_PREROLL_MS` | 300 | (unset) | profile | keep |
| `LISTENER_FLUSH_MS` | 700 | (unset) | profile | keep |

## Conductor (`conductor.py`, `system1.py`, `system2.py`, `knowledge.py`, `review.py`)

| Name | Code | Compose | Class | Decision |
|---|---|---|---|---|
| `CONDUCTOR_SOCKET` | /run/avatar/conductor.sock | same | deployment | keep |
| `CONVERSATION_LOG_DIR` | results/conversations | (unset) | deployment | keep |
| `SYSTEM1_CONFIG` | config/system1.json | same | deployment | keep |
| `SYSTEM1_MEMORY` | results/system1/corrections.jsonl | same | deployment | keep |
| `SYSTEM1_MODEL` | JevK5 4B v0.3 Q4_K_M | same | deployment | keep |
| `SYSTEM2_MODEL` | Llama 3.2 3B Q4_K_M | same | deployment | keep |
| `KNOWLEDGE_DIR` | /workspace/knowledge | same | deployment | keep |
| `KNOWLEDGE_EMBED_MODEL` | intfloat/multilingual-e5-small | (unset) | deployment | keep |
| `REVIEW_RESULTS` | /workspace/results | (unset) | deployment | keep |
| `REVIEW_MODEL` | Qwen3 4B Q4_K_M | (unset) | deployment | keep |
| `CONDUCTOR_RESPONDER` | system1 | system1 | profile | keep (`echo` for tests) |
| `SYSTEM2_ENABLED` | true | true | profile | keep |
| `TURN_SILENCE_MS` | 700 | 700 | profile | keep |
| `TURN_END_PUNCTUATION_MS` | 400 | 400 | profile | keep |
| `TURN_CONNECTIVE_MS` | 1200 | 1200 | profile | keep |
| `TURN_CONNECTIVES` | and,but,…,als | (unset) | profile | keep |
| `LISTENER_MIN_SILENCE_MS` | 300 | 300 | profile | keep; must equal the listener's value |
| `PTT_FINAL_WAIT_MS` | 1500 | (unset) | profile | keep |
| `SYSTEM1_TEMPERATURE` | 1.22 | (unset) | constant | belongs to the model file |
| `SYSTEM2_MAX_TOKENS` | 120 | (unset) | profile | keep |
| `REVIEW_UNSURE_BELOW` | 0.6 | (unset) | profile | keep |
| `REVIEW_REPEAT_SIMILARITY` | 0.88 | (unset) | profile | keep |

The listener, conductor and TTS settings have one name and agreeing defaults;
the problems above are concentrated in the avatar service.

## Not part of this

- `config/system1.json` is domain content (classes, reactions) with hot reload.
  It stays a separate mechanism and gets its own schema check.
- `BENCHMARK_*`, `QUALITY_BENCHMARK_*`, `REVIEW_FPS`, `REVIEW_MODE` are
  arguments of the host scripts, not service configuration. The benchmark
  profiles (`runtime`, `visual`) inside `quality_benchmark.sh` do move to the
  shared profiles; today `runtime` still uses stride 2 while live runs stride 1.

## Hidden dependencies to validate

- Eye, head and lip scales and the lip mode do nothing with
  `AVATAR_RELATIVE_MOTION=false`.
- `AVATAR_LIP_MOTION_SCALE` only applies with `AVATAR_LIP_MOTION_MODE=relative`.
- `TTS_PREFETCH_POLICY` and `TTS_PREFETCH_MIN_BUFFER_SECONDS` do nothing with
  `TTS_PREFETCH=false`; the minimum buffer only applies to `adaptive`.
- `CATCHUP_RENDER_STRIDE` and `CATCHUP_BUFFER_SECONDS` do nothing with
  `ADAPTIVE_RENDER_STRIDE=false`; the catch-up stride must be >= the render
  stride (today silently raised).
- `SPEECH_START_GATE` and `RENDER_WINDOW_FRAMES` only apply with
  `INCREMENTAL_FRAME_WINDOWS=true`.
- `PHRASE_MAX_CHARS` must be >= `PHRASE_TARGET_CHARS` (today silently raised).
- `WARMUP_IDLE_FRAME_INDEX` does nothing with `USE_NEURAL_IDLE_FRAME=false`.
- `LISTENER_MIN_SILENCE_MS` must be the same in the listener and the conductor,
  and smaller than the three `TURN_*_MS` values.
- `CHATTERBOX_LANGUAGE`, `CHATTERBOX_EXAGGERATION` and `CHATTERBOX_CFG_WEIGHT`
  each apply to one variant only.
