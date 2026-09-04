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
