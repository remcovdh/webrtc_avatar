# neural-avatar-v2e1-benchmark-handshake-fix

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
preserving all v2D voice choices and the accepted serial renderer defaults.

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
- Direct-memory stride-two production defaults.
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
