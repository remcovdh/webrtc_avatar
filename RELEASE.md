# neural-avatar-v2a-warmup-metrics

## Purpose

Establish a warm, repeatable latency baseline before changing the renderer.

## Changed

- Added an optional startup request that warms Breeze TTS, HuBERT, JoyVASA and
  FasterLivePortrait before the avatar service reports ready.
- Added `STARTUP_WARMUP` and `WARMUP_TEXT` Compose settings.
- Added a unique `server_build` marker and warm-up results to `/health`.
- Split TTS timing into response headers, first byte, body transfer and WAV
  finalization.
- Split rendering timing into FasterLivePortrait pipeline and MP4 decode time.
- Added these measurements to the browser table and server logs.
- Added a benchmark procedure, acceptance criteria, rationale and rollback to
  `README.md`.

## Not changed

- Phrase splitting and playback behavior.
- FasterLivePortrait model settings or output quality.
- MP4/FFmpeg generation.
- Render frame rate or frame skipping.
- TTS/render concurrency.
- Docker base images and dependency versions.

## Rollback

Set `STARTUP_WARMUP: "false"` to restore the previous startup behavior while
retaining the additional timing measurements.

