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
