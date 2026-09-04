# neural-avatar-v2d-voice-modes

## Purpose

Compare voice consistency and Breeze TTS cost without changing the accepted
serial, direct-memory, stride-two avatar renderer.

## Changed

- Added browser-selectable `design`, `preset-clone` and `preset-direction`
  modes.
- A preset can be an authorized WAV or one Breeze-designed sentence generated
  once; both use the same fixed WAV plus exact-transcript configuration.
- Preset files are server-controlled. The browser selects a mode but cannot
  supply a path.
- Designed voice defaults to CFG 4, preset clone to CFG 1, and preset direction
  to CFG 4.
- Preset options remain visibly disabled until both `voice-preset.wav` and
  `voice-preset.txt` are valid.
- Added Voice and CFG to per-phrase UI metrics and server logs.
- Added voice configuration and preset readiness to `/health` and
  `/client-config`.
- Changed `TTS_PREFETCH` default to `false` after v2C-B shared-GPU contention
  approximately halved portrait inference speed.
- Documented preset generation, external reference setup, exact rollback,
  v2C-B rejection, the recovered serial baseline, and the v2D A/B procedure.

## Deliberately not changed

- Breeze itself is unmodified and re-encodes uploaded reference audio per
  phrase; reference-token caching is a possible later experiment.
- Phrase splitting, phrase-level batching, playback order and WebRTC formats.
- JoyVASA and FasterLivePortrait model configuration.
- Direct-memory render and stride two.
- Paste-back remains disabled.
- No phrase-boundary blend or incremental PCM/motion/frame delivery yet.

## Expected result

- `/health` reports build `neural-avatar-v2d-voice-modes`, prefetch false and
  all three mode IDs.
- Without preset files, only Designed voice is selectable.
- With a nonempty preset WAV and exact transcript, clone and direction modes
  become selectable after recreating `webrtc-avatar`.
- Every phrase timing row identifies its voice mode and CFG scale.
- Preset modes should stabilize voice identity; whether clone CFG 1 also lowers
  TTS time is intentionally left for measurement.

## Rollback

Select Designed voice in the browser or set:

```yaml
BREEZE_DEFAULT_VOICE_MODE: "design"
TTS_PREFETCH: "false"
```

Then recreate `webrtc-avatar`. No rebuild is required for a runtime rollback.
