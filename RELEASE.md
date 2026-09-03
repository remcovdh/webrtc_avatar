# neural-avatar-v2c-a-render-stride

## Purpose

Measure the latency and visual-quality effect of rendering alternating JoyVASA
motion frames after v2B proved that neural frame inference is the sustained
bottleneck.

## Changed

- Added `RENDER_STRIDE`, default `2`, to direct-memory rendering.
- Rendered motion indices 0, 2, 4 and so on while assigning the resulting frame
  list half the source FPS, preserving phrase duration and lip-sync timestamps.
- Added configured/effective stride, rendered/original frame counts and playback
  FPS to `/health`, logs, WebRTC metrics and the browser table.
- Preserved exact v2B behavior when `RENDER_STRIDE=1`.
- Preserved automatic stride one for the legacy MP4/paste-back path.
- Recorded the complete supplied v2B timings and log conclusions.
- Added a v2C-A benchmark procedure, predicted ranges, visual acceptance check,
  decision record and rollback to `README.md`.

## Deliberately not changed

- Phrase splitting thresholds and sequential phrase scheduling.
- Breeze, JoyVASA or FasterLivePortrait model configuration.
- WebRTC's 30 FPS video and 48 kHz audio output.
- Phrase-level batching: playback still waits for all frames in a phrase.
- Paste-back remains disabled because of the earlier cuSOLVER failure.
- No TTS prefetch, interpolation or incremental frame append is included.

## Expected result

- `/health` reports build `neural-avatar-v2c-a-render-stride`, backend
  `direct-memory` and stride `2`.
- A 16/76/66-frame test renders 8/38/33 neural frames at 12.5 playback FPS while
  retaining 0.64/3.04/2.64-second media durations.
- Frame-loop time should fall roughly 40–50%; visual smoothness must be judged
  on the RTX 5080 before the change is accepted.

## Rollback

Set `RENDER_STRIDE: "1"` under `webrtc-avatar.environment` and run:

```bash
docker compose up -d --force-recreate webrtc-avatar
```

No rebuild is required. This restores v2B behavior while retaining direct
memory. `DIRECT_MEMORY_RENDER=false` remains the deeper legacy-MP4 rollback.
