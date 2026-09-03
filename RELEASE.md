# neural-avatar-v2b-direct-memory

## Purpose

Measure and remove the file/codec transport overhead between JoyVASA,
FasterLivePortrait and the existing WebRTC queue without changing neural model
settings or output-frame selection.

## Changed

- Added a direct renderer that calls JoyVASA `gen_motion_sequence` and
  FasterLivePortrait `run_with_pkl` for each phrase in memory.
- Bypassed the motion pickle, crop/original MP4 encodes, two FFmpeg audio muxes
  and final OpenCV MP4 decode when the direct backend is active.
- Retained the complete v2A.1 renderer as `legacy-mp4`.
- Added `DIRECT_MEMORY_RENDER=true` to Compose as a reversible runtime switch.
- Direct mode automatically falls back to the legacy path when paste-back is
  enabled.
- Added render backend, JoyVASA motion time, FasterLivePortrait frame-loop time
  and effective frame FPS to logs, WebRTC metrics and the browser table.
- Added renderer selection to `/health` and changed `server_build` to the unique
  v2B identifier.
- Added Docker build assertions for the upstream APIs used by direct mode.
- Recorded the supplied v2A.1 benchmark, interpretation, v2B test procedure,
  acceptance criteria, decision and rollback in `README.md`.

## Deliberately not changed

- Phrase splitting thresholds and sequential phrase scheduling.
- Breeze, JoyVASA or FasterLivePortrait model configuration.
- The number of inferred source frames or their quality.
- WebRTC's 30 FPS video and 48 kHz audio output.
- Phrase-level batching: playback still waits for all frames in a phrase.
- Paste-back remains disabled because of the earlier cuSOLVER failure.

## Expected result

- `/health` reports `render_backend: direct-memory`.
- UI rows report backend `direct-memory` and Decode `0 ms`.
- No FFmpeg banner or phrase MP4 appears during an interactive request.
- Pipeline time may decrease modestly; the approximately 11.2 FPS neural frame
  loop remains the likely dominant bottleneck.

## Rollback

Set `DIRECT_MEMORY_RENDER: "false"` under `webrtc-avatar.environment` and run:

```bash
docker compose up -d --force-recreate webrtc-avatar
```

No rebuild is required for this runtime rollback.
