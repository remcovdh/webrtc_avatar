# neural-avatar-v3a-reproducible

## v3A smooth phrase changes, reproducible builds, licence clean-up

- **No more gap between phrases or skipped frames.** Synthesizing the next
  phrase's speech on the same GPU dropped rendering below real time, so the
  video fell behind the voice (200-430 ms of frames skipped) and a 220 ms
  silence followed the first phrase. Each window of frames now reports whether
  rendering must catch up; the next window then renders every second frame and
  shows it twice. On the three-phrase test text both are gone, for 8-12 held
  frames out of 72 in the first phrase; time to first playback is unchanged.
  `tools/timing_compare.py` compares timing runs per phrase.
- **Reproducible builds.** Base image pinned by digest; FasterLivePortrait
  (was `master`), Perth (was `master`) and jevk5 by commit; every Python
  package by `docker/locks/<service>.txt`, used as pip constraints and verified
  by `docker/check_lock.py` at the end of each stage; TensorRT by wheel
  checksum; every model by Hugging Face revision (`repo@revision`) or
  checksum. A rebuild from scratch reproduced all four lock files exactly.
- **System 2 runs on IBM Granite 4.0 1B** (Apache-2.0) instead of Llama 3.2 3B
  (Llama Community License): median about 110 ms instead of 260 ms and 1.5 GB
  of GPU memory instead of 2.9 GB, with answers in English that stay with the
  notes. The prompt now ends with a short reminder of the rules.
- **The InsightFace weights are no longer used** (non-commercial research
  only). The face in the portrait is found with MediaPipe (Apache-2.0); its
  landmarks are laid out so FasterLivePortrait crops the same region (within
  2% in scale on five test portraits). `AVATAR_FACE_DETECTOR=insightface`
  switches back.
- `THIRD_PARTY_NOTICES.md` lists every upstream project and model with its
  licence.

# neural-avatar-v2z-server-split

## v2Z `server.py` split into modules

No behaviour change. `src/avatar/server.py` was 2,172 lines with seven
concerns; it is now 261 lines that wire these modules together:

| Module | What it holds |
|---|---|
| `phrases.py` | splitting a text into phrases |
| `playback.py` | `PlaybackBuffer`: per-browser audio/video queue on one clock; `IdleFrame` |
| `tracks.py` | the WebRTC audio and video tracks, keep-alive noise |
| `tts_client.py` | `TtsClient`: voice modes, synthesize, readiness |
| `renderer.py` | `Renderer`: FasterLivePortrait load, TensorRT, upright crop, the frame loop; motion adjustment |
| `metrics.py` | the per-phrase metrics event and log line, from one dictionary |
| `speech.py` | `Speaker`: speaking one text (stride choice, windows, prefetch) and the warm-up |
| `session.py` | `Session`: one browser connection |

- State that was changed through `global` now belongs to objects (`Renderer`,
  `Speaker`, `IdleFrame`, `Runtime`); no `global` statement is left. The
  modules take the settings object instead of 45 module constants.
- The 399-line `_create_clip` became `Speaker.speak` plus `_speak_phrase`; the
  14 nested functions in `/offer` became the `Session` class.
- Tests: the 18 assertions that matched literal lines of `server.py` are
  replaced by behaviour tests with stand-ins for the renderer and the TTS
  (motion reference reset per utterance, prefetch timing for each policy,
  window order from the render thread, the idle image from the warm-up,
  refusals and failures). The avatar suite went from 60 to 103 tests.
- Three metric fields that were constant since the legacy renderer was removed
  are dropped: `render_backend`, `render_decode_ms`,
  `render_pipeline_reported_ms`.
- `tools/smoke.sh` rebuilds and restarts the avatar and checks it end to end.

# neural-avatar-v2y-structure

## v2Y repository structure

No behaviour change; the files moved.

- Code is one Python package per service under `src/`: `avatar`, `conductor`,
  `listener`, `tts`, plus `shared` for the two protocol modules. Imports are
  package imports and the services start as modules (`python -m avatar.serve`,
  `python -m conductor.app`, `python -m listener.worker`). Renamed on the way:
  `avatar_config.py` -> `src/avatar/config.py`, `conductor.py` ->
  `src/conductor/app.py`, `listener_worker.py` -> `src/listener/worker.py`.
- Tests are in `tests/<service>/`. The images keep the repo's layout under
  `/workspace/app`, so tests run the same on the host and in the build.
  `tools/run_tests.sh` runs every service's tests against the working tree
  without a rebuild. `test_benchmark_handshake.py` was never part of the
  build's test run and had gone stale; it is fixed and runs now.
- `scripts/` holds what you run in normal use (`benchmark.sh`,
  `quality_benchmark.sh`, `review.sh`, `review_recording.sh`,
  `benchmark_avatar.py`); `tools/` keeps the diagnosis scripts (now including
  `lip_sync_analysis.py`). `docker/` holds the Dockerfile and entrypoint.
- Documentation: a short current `README.md`; `CONFIG.md` and `BACKLOG.md` stay
  in the root; `docs/` has an index, `architecture/` and `history/`. The old
  README body is kept unchanged as `docs/history/readme-until-v2w.md`.
- `SHA256SUMS` is gone; git already guarantees file integrity.

Earlier entries below use the file names of their time.

# neural-avatar-v2x-config

## v2X one configuration

- `avatar_config.py` is the single source of the avatar's settings: defaults
  (the tested live setup), profiles (`live`, `development`,
  `benchmark-runtime`, `benchmark-visual`) and validation. Before, defaults
  lived in the code, in Compose, in `.env.example` and in the local `.env`, and
  disagreed (stride 2 vs 1, TensorRT off vs on, lip offset 0 vs 80 ms, ...).
- One name per setting, `AVATAR_...`, the same on the host and in the
  container. Compose passes settings on and holds no values. An old name stops
  the avatar with a message naming the new one.
- Start-up validation instead of silent correction: unknown names, values out
  of range and impossible combinations (catch-up stride below the render
  stride, phrase maximum below the target) are errors; a setting that cannot
  have an effect is a warning. The conductor checks that the listener's
  silence is shorter than the turn-ending silences.
- `/health`, benchmark results and conversation logs carry the profile, the
  overrides and a configuration hash; a changed diagnosis switch marks the run
  as an experiment.
- Removed: Breeze TTS (Docker target, benchmark scripts, voice direction, CFG
  scales, preset transcript, `preset-direction` mode), the dead flags
  `AVATAR_PASTE_BACK`, `DIRECT_MEMORY_RENDER`, `PROGRESSIVE_PHRASE_MODE` with
  the legacy MP4 renderer, and nine never-varied settings (now constants).
- `benchmark-runtime` now measures the live render path (stride 1 with
  catch-up 2); it used stride 2, so new results are not comparable with old
  ones. `benchmark.sh` defaults follow.
- New tools: `tools/config_snapshot.py` (running stack against a reference),
  `tools/run_tests.sh` (avatar tests on the working tree without a rebuild).
  `test_avatar_config.py` covers profiles, invalid combinations and the
  Compose pass-through; it needs no GPU.
- `CONFIG.md` describes every setting.

# neural-avatar-v2w-jevk5-review

## v2W JevK5 System 1 and the conversation review (milestone M5)

- System 1 now uses JevK5 v0.3 4B (Qwen3.5-4B + LoRA, option-logit readout,
  Apache-2.0) in-process with llama.cpp, behind a new `Decider` interface.
  It replaced Laya after research: JevBench scores Laya multilingual at
  Intelligence 2.4, and on our 18 Dutch/English utterances JevK5 got intent
  15/18 and emotion 17/18 (Laya 11/16 and 4-9/16); 168 ms per utterance,
  +3.4 GB GPU (total ~14.2 of 16.3 GB). The correction memory now uses the
  multilingual-e5-small embeddings. torch must load before llama.cpp.
- `review.py` / `./review.sh` (M5, manual): facts from the conversation logs
  (counts, unsure decisions, corrections, knowledge gaps, turns that went
  wrong: pushback or a repeated question) and suggestions from Qwen3 4B
  (knowledge pages, class descriptions, new classes, diagnoses). Writes
  `results/review/<time>.md` and `.json`; changes nothing itself. The script
  pauses the avatar and TTS while the review model runs.
- `tools/` keeps the test and measurement scripts and test audio in the repo
  (they were lost twice in a cleared scratch folder); see `tools/README.md`.
- Decided (see `LISTENING_PLAN.md`): no model training yet, manual review
  only, voice emotion on the backlog.

# neural-avatar-v2v-system2

## v2V System 2 answers (milestone M4 of `LISTENING_PLAN.md`) and audio fixes

- `knowledge.py` (`Knowledge` interface): the Markdown pages in `knowledge/`
  are split per heading and embedded with multilingual-e5-small on the CPU;
  re-indexed on change. Retrieval threshold 0.8 (answerable questions scored
  0.83-0.90, unanswerable 0.71-0.77).
- `system2.py` (`System2` interface): Llama 3.2 3B Instruct (Q4_K_M) through
  llama.cpp in the conductor (compiled for sm_120), only rephrasing retrieved
  passages into 1-3 spoken English sentences. Chosen over Qwen3 4B (too long
  for speech) and Phi-4-mini (answered Dutch questions in Dutch); median 260 ms.
- No relevant passage: an honest "I don't know" without calling the LLM,
  logged as a knowledge gap. A filler phrase is spoken first only if the answer
  takes longer than 600 ms; the avatar speaks queued lines in order.
- Keyword rules now catch question and request starters (EN/NL) and a trailing
  "?": Laya missed short questions whose "?" the ASR dropped.
- Fix: the reference voice recording started with "Hello, I am your friendly
  virtual assistant"; any reply starting with "Hello" made Chatterbox Turbo
  speak that sentence instead. The preset is now the same recording without
  "Hello," (`inputs/voice-preset-nohello.wav`).
- Fix: laptop speaker amplifiers powered down during digital silence and cut
  the first word of the next sentence. Pauses now carry inaudible noise
  (`AVATAR_AUDIO_KEEPALIVE_DBFS`, -60).
- The page shows the knowledge source of each answer, a meter for avatar sound
  arriving in the browser, and a "Test speakers" button.

# neural-avatar-v2u-system1

## v2U System 1 reactions (milestone M3 of `LISTENING_PLAN.md`)

- `system1.py` behind the `System1` interface: Laya multilingual (on the CPU,
  ~120 ms) decides intent and emotion; keyword rules on the ASR text (short
  utterances only) and a correction memory (Laya embeddings, cosine >= 0.9)
  override it. spaCy + YAKE + rapidfuzz pick the topic, with the page's
  language choice. Questions and requests are marked for System 2.
- The conductor's `System1Responder` answers with an English phrase per
  intent (+emotion), naming the topic where possible.
- The page shows each decision with intent/emotion dropdowns; a correction is
  stored in `results/system1/corrections.jsonl` and applies at once.
- Classes, criteria, keyword rules, watch words and phrases live in
  `config/system1.json`, re-read on change.
- Measured: zero-shot intent 11/16, emotion weak (bias to "surprised", so
  low-confidence emotions become "neutral"); a decision takes ~230-280 ms.

# neural-avatar-v2t-conductor

## v2T conductor and turn-taking (milestone M2 of `LISTENING_PLAN.md`)

- New `conductor` image/process owns the conversation. The avatar reaches it
  through `ConductorClient` (the `Conductor` interface, thin Unix-socket proxy)
  and receives `say`/`show` commands (the `AvatarOutput` interface).
- Turn-taking: half-duplex (speech that starts while the avatar talks is
  ignored); the turn ends after 400 ms of silence when the text ends in
  `?`/`.`/`!`, 700 ms otherwise, 1.2 s after a connective; the listener's own
  utterance silence (now 300 ms) is subtracted. Push-to-talk in the page
  bypasses thresholds and the speaker filter. Only the primary speaker (first
  labelled voice) is answered.
- The reply is still an echo ("You said: ..."); System 1 and 2 replace the
  `Responder` in M3/M4. Measured: turn over 0.4 s after the final text, avatar
  speaking 1.4 s later.
- Every turn is one local JSON line in `results/conversations/`; utterance
  audio is opt-in (`LISTENER_SAVE_AUDIO=true`); `forget.py` deletes logs.
- Fix found by the tests: an utterance begun during push-to-talk whose final
  text arrived after the release was dropped by the speaker filter.

# neural-avatar-v2s-listening

## v2S listening (milestone M1 of `LISTENING_PLAN.md`)

- New `listener` image and process: Silero VAD (CPU), Nemotron 3.5 streaming
  ASR 0.6B and Nemotron-3-Diarization, through Transformers (pinned main
  commit) rather than NeMo, whose integration needs Python 3.13 / CUDA 13.
  1.67 GB of GPU memory; ~24 ms of GPU per 0.64 s of audio for diarization.
- The avatar codes against the `Listener` interface (`listener_protocol.py`);
  `SocketListener` proxies it over a Unix socket (tiny binary framing, no HTTP).
  Without a running listener the avatar works as before.
- The page streams the microphone over the existing WebRTC connection and shows
  a live transcript with speaker labels. Spoken language is chosen in the page:
  English (default), Nederlands or Automatic (which mixed languages on Dutch).
- HTTPS on port 8443 with a self-signed certificate for the host's IP addresses,
  so the microphone works from other machines (`serve.py`, `entrypoint.sh`).
- Workaround: the pinned processor gives one mel frame too many for the first
  streaming chunk; `fit_frames` trims chunks to the size the model requires.

# neural-avatar-v2r-upright-crop

## v2R upright image without black borders; offset hang fix

- The output frame was tilted with black corner wedges: FLP rotates the source
  crop to straighten a slightly tilted face, and never passes its own
  `flag_do_rot` to `crop_image`. `_crop_source_image` now keeps the crop upright
  (`AVATAR_CROP_ROTATION=false`, default); the face keeps its natural tilt.
- The 2.3x face crop reached 23 px past the left and 16 px past the top of the
  512x512 portrait, which showed as black bands. The crop is now redone with
  mirrored edges (`BORDER_REFLECT_101`), so background and hair continue.
- Fix: with `AVATAR_LIP_SYNC_OFFSET_MS=80`, the last 80 ms of video frames never
  became due once the voice ended, so the request waited forever (the browser's
  "Generate & speak" button stayed disabled). During silence the video clock now
  catches up by up to the offset.

# neural-avatar-v2q-realtime-baseline

## v2Q real-time baseline

First build that starts speaking in ~0.9 s, plays phrases without pauses, keeps
the mouth in sync and looks natural. See `REALTIME_BASELINE.md`.

- Audio and video share one WebRTC stream id, so browsers synchronise them
  with RTCP sender reports (aiortc gave each track its own random stream).
- `AVATAR_LIP_SYNC_OFFSET_MS=80` (default) shows the mouth 80 ms later; JoyVASA's
  mouth leads the voice by 40-160 ms. Browser buffers went from 34/290 ms
  (audio/video) to 53/66 ms.
- Stride 1 (all 25 motion FPS) is the default, with catch-up stride 2.
- The page shows the browser's live audio/video jitter-buffer delays.
- `lip_sync_analysis.py` measures JoyVASA's mouth lead/lag per dumped phrase.
- `REALTIME_BASELINE.md` explains all changes since v2N.2, including TensorRT.

# neural-avatar-v2p-tensorrt-fp16

## v2P TensorRT FP16 face rendering

- `warping_spade`, the per-frame bottleneck, now runs through ONNX Runtime's
  TensorRT execution provider in FP16: 27 ms instead of 97 ms per frame on the
  RTX 5080 (FP32 TensorRT only reached 87 ms). Mean output difference versus
  the CUDA provider is 0.00125 on a 0-1 scale; no visible change.
- TensorRT cannot import the 5-D GridSample, so ONNX Runtime keeps that node on
  the CUDA provider and runs the rest in TensorRT.
- End-to-end rendering rose from ~9 to 22-35 frames/s. The user's three-phrase
  test text now plays in 10.2 s instead of 15.4 s, with no speech wait and no
  gaps between phrases.
- `AVATAR_TENSORRT=true` (default); engines are cached in `./models/trt-cache`
  (first start builds them in ~30 s). Any TensorRT failure falls back to the
  CUDA provider; `/health` reports `warping_backend`.
- The speech-start estimate is raised by the warm-up's measured render speed,
  removing the first phrase's wait for the assumed `EXPECTED_RENDER_FPS`.
- The image installs `tensorrt-cu12-libs==10.16.1.11` but keeps only the
  sm_120 builder resources: 1.1 GB instead of 6.2 GB of libraries.

# neural-avatar-v2o-audio-clock-expression

## v2O audio-clocked playback, speech start gate and expression control

- Video frames carry their time on the audio timeline; frames the voice has
  already passed are skipped, so the mouth can no longer drift behind speech
  when FLP renders below real time (it lagged up to ~0.8 s per phrase).
- Speech start gate: each phrase's speech waits until
  `(W + (D - W) * (1 - render speed)) * 1.15` seconds of video exist, so the
  rest arrives in time without skipped frames. Render speed is measured per
  phrase. Costs ~1-2 s extra start latency and short pauses between phrases on
  the RTX 5080 (~9 rendered FPS vs 12.5 needed at stride 2).
- `AVATAR_LIP_MOTION_MODE=absolute` (default) uses JoyVASA's mouth shapes
  directly. Relative lips pressed this closed-smile portrait's lips shut.
- `AVATAR_EYE_MOTION_SCALE=0.3` (default) damps JoyVASA eye motion, removing
  the staring and winking while keeping some eye life.
- `AVATAR_HEAD_MOTION_SCALE=0.3` (default) damps JoyVASA head rotation and
  translation around the first pose; at 1.0 pitch drifted 1-5 degrees and roll
  up to 5 degrees, so the avatar looked above the camera.
- `review_recording.sh` records one phrase over WebRTC and writes frame,
  close-up and mouth/waveform sheets for visual review.
- Optional `AVATAR_DEBUG_DUMP_DIR` saves each phrase's WAV and JoyVASA motion.
- New metrics: `video_dropped_ms`, `speech_hold_ms`, `speech_lead_ms`.

# neural-avatar-v2n2-1-build-fixture-fix

## v2N.2.1 build correction

- Copies `quality_benchmark.sh` into `/workspace/FasterLivePortrait` before
  the Docker build executes `test_visual_quality.py`.
- Keeps all v2N.2 runtime and benchmark-profile behavior unchanged.

# neural-avatar-v2n2-benchmark-profiles

## v2N.2 benchmark correction

- Makes the visual-quality runner default to a deployable stride-2 adaptive
  incremental runtime profile.
- Adds a non-incremental stride-1 visual profile that renders before playback,
  preserving synchronization for full-detail artifact comparison.
- Names result roots with `-runtime` or `-visual` to prevent invalid comparisons.
- Documents why a roughly 11.5 neural-FPS renderer cannot stream 25 unique
  frames per second without progressively falling behind audio.

# neural-avatar-v2n1-av-sync-gate

## v2N.1 synchronization correction

- Gates incremental audio consumption on the same `started` state set by the
  first completed FLP video window.
- Prevents speech from leading facial motion by one first-window render delay.
- Replaces the previous source-order test with a regression check for the
  actual playback gate.
- Removes the black-producing lip-retargeting case from the default visual
  matrix while leaving it available through an explicit matrix override.

# neural-avatar-v2n-visual-quality-benchmark

## v2N visual-quality benchmark

- Exposes animation region, driving multiplier, lip normalization, and eye/lip
  retargeting through `.env` and Compose.
- Reports the active visual configuration through `/health` and phrase metrics.
- Adds `quality_benchmark.sh` with a controlled six-scenario mouth/gaze matrix.
- Records every headless WebRTC run as an MP4 alongside exact resolved settings,
  raw events, phrase CSV, logs, and comparison reports.
- Forces render stride 1 and disables adaptive catch-up/prefetch during visual
  comparisons so performance shortcuts do not confound articulation quality.
- Keeps the accepted v2M runtime defaults unchanged.

# neural-avatar-v2i-persistent-motion

## v2I optimization

- Defers prepared-avatar preview work as requested.
- Makes the v2H-winning Breeze combination the Compose default.
- Enables configurable relative motion.
- Resets FasterLivePortrait's driving reference only on phrase 1 of an
  utterance, preserving it for later progressive phrases.
- Resets again for each new user action.
- Exposes motion policy in health, phrase metrics, logs, and benchmark reports.
- Adds a build-time regression test for continuity wiring.

## v2H optimization

- Adds the `depth-codec` and `depth-codec-backbone` combination profiles.
- Captures the first measured raw PCM response as a proper WAV per profile.
- Adds capture filename and SHA-256 to JSON and CSV output.
- Links each WAV from the generated Markdown report.
- Does not read, rename, overwrite, or otherwise change files under `inputs/`.

## v2G.1 correction

- Passes `--fast-args=<value>` so values such as
  `--fast-backbone-decode` cannot be parsed as benchmark options.
- No Breeze, avatar, rendering, or benchmark-policy behavior changed.

## v2G optimization

- Preserves the proven v2F adaptive stride behavior.
- Adds `breeze_benchmark.sh` and `breeze_benchmark.py`.
- Tests Breeze eager and each official fast stage with fixed inputs.
- Captures warmed first byte, RTF, total time, and sampled GPU memory.
- Continues after unsupported/OOM profiles and retains their logs.
- Adds an opt-in `fast-all` test and documents the two-laptop topology.
- Makes `BREEZE_TTS_URL` configurable for a remote Breeze server.

## v2F optimization

- Adds an adaptive render policy using base stride 2 and catch-up stride 3.
- Phrase 1 retains base-stride quality. Later phrases select catch-up below
  0.75 seconds buffered and return to base stride when the queue is healthy.
- Adds `ADAPTIVE_RENDER_STRIDE`, `CATCHUP_RENDER_STRIDE` and
  `CATCHUP_BUFFER_SECONDS` with Compose overrides.
- Adds selected stride, buffer-before-render and decision reason to logs, UI,
  JSON and phrase CSV metrics.
- Extends the benchmark with fixed/adaptive scenario matrices and policy-aware
  scenario names.
- Based on the three-repeat RTX 5080 result: stride 3 reduced render RTF from
  1.146 to 0.781 and gaps from 9.42 to 7.34 seconds. Stride 4 was not selected
  as the default because its playback rate is only 6.25 FPS.

## v2E.1 correction

- Fixes the smoke-test timeout after WebRTC connected successfully but the
  server did not emit its initial `ready` event.
- The server now announces readiness both from the channel `open` callback and
  from an immediate ready-state check, guarded against duplicate events.
- The benchmark treats its locally open data channel as authoritative and
  continues after a two-second compatibility wait when an older server omits
  the optional event.
- `test_benchmark_handshake.py` covers both a received `ready` message and the
  missing-message compatibility path.

## Purpose

Replace ad-hoc timing comparisons with a repeatable end-to-end harness while
preserving all voice choices and the accepted serial TTS/render scheduling.

## Added

- `benchmark_avatar.py` with two commands:
  - `prepare-preset` generates a deterministic Breeze WAV, exact transcript and
    SHA-256 manifest;
  - `run` negotiates the real WebRTC endpoint, consumes media, sends the same
    data-channel request as the UI, and captures every server event.
- `benchmark.sh` orchestrates Breeze, optional fixture creation, avatar
  recreation, health checks, per-mode warm-ups and measured runs.
- JSON, phrase CSV and Markdown output under `results/benchmarks/`, plus a
  timestamped cross-scenario comparison CSV and Markdown table.
- Median first-ready/first-byte timing, TTS RTF, render RTF, weighted neural
  FPS, media duration, gap and wall-time comparisons.
- Configurable repeat, warm-up, voice-mode, render-stride and prefetch matrices.
- A Compose `benchmark-fixture` profile with write access only to the fixture
  and result volumes.
- Namespaced Compose overrides for stride, prefetch and phrase thresholds.

## Reproducibility controls

- Fixed preset text, instruction, CFG and seed.
- Fixed benchmark text and directions.
- Identical test text for all modes.
- One warm-up per mode is recorded but excluded from medians.
- Measured modes rotate within each repeat rather than being tested in isolated
  batches.
- Complete health/config snapshots and raw data-channel events are retained.
- The canonical fixture uses dedicated `benchmark-voice-preset.*` files, so the
  normal `voice-preset.*` pair is not touched.
- `BENCHMARK_PRESET_BASENAME=voice-preset` can deliberately test the normal
  preset without changing it.

## Deliberately not changed

- Voice generation, JoyVASA or FasterLivePortrait algorithms.
- Direct-memory rendering and a base stride of two.
- TTS prefetch remains disabled by default.
- Breeze still re-encodes reference audio for every phrase.
- Voice identity/naturalness and visual lip sync still need human assessment.
- No phrase-boundary blend or incremental PCM/motion/frame delivery yet.

## First run

```bash
docker compose build webrtc-avatar benchmark-fixture
./benchmark.sh
```

Use `REGENERATE_PRESET=1 ./benchmark.sh` only when intentionally regenerating
the selected benchmark preset basename.
