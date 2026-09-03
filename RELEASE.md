# neural-avatar-v2c-b-tts-prefetch

## Purpose

Measure whether a bounded overlap between next-phrase Breeze TTS and
current-phrase FasterLivePortrait rendering reduces pauses on the shared RTX
5080 without destabilizing or slowing either model.

## Changed

- Added `TTS_PREFETCH`, default `true`, to overlap TTS for phrase N+1 with
  rendering phrase N.
- Fixed prefetch depth at one: there is never more than one Breeze request and
  at most one future WAV.
- Added cancellation/cleanup for unfinished prefetch work on request failure.
- Added total TTS, blocking TTS wait, hidden TTS overlap and prefetched status to
  server logs and WebRTC metrics.
- Added TTS wait/overlap columns and prefetch runtime status to the browser.
- Added prefetch status/depth and a unique v2C-B build marker to `/health`.
- Recorded the supplied v2C-A benchmark and its exact gap equation.
- Added v2C-B predictions, acceptance criteria, GPU-contention checks, decision
  record and runtime rollback to `README.md`.

## Deliberately not changed

- Phrase splitting thresholds, phrase order and one-at-a-time playback.
- Breeze, JoyVASA or FasterLivePortrait model configuration.
- WebRTC's 30 FPS video and 48 kHz audio output.
- Phrase-level batching: playback still waits for all frames in a phrase.
- Paste-back remains disabled because of the earlier cuSOLVER failure.
- Render stride remains two; no interpolation or incremental frame append is
  included.

## Expected result

- `/health` reports build `neural-avatar-v2c-b-tts-prefetch`, prefetch true and
  depth one.
- Phrase one reports no overlap; later phrases report nonzero overlap.
- With no GPU contention, the measured four-phrase workload predicts phrase-two
  gap near 4.87 seconds and later gaps near zero.
- Acceptance depends on total gap, first-ready, total TTS, Neural FPS, GPU
  memory and absence of CUDA errors—not overlap alone.

## Rollback

Set `TTS_PREFETCH: "false"` under `webrtc-avatar.environment` and run:

```bash
docker compose up -d --force-recreate webrtc-avatar
```

No rebuild is required. This restores exact v2C-A serial scheduling while
retaining direct memory and stride two.
