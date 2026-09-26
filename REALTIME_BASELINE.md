# Real-time avatar baseline (v2Q)

This is the first build where the avatar starts speaking within about a second,
talks without pauses between phrases, keeps its mouth in sync with the voice and
looks natural. This document explains what was wrong and what changed to get
there, starting from `neural-avatar-v2n2-1-build-fixture-fix`.

Hardware: RTX 5080 Laptop GPU (16 GB), passed through to a QEMU/KVM Ubuntu VM.

## Result

Measured on the same three-phrase test text. v2O already had the
synchronisation and facial fixes but still rendered with the CUDA provider:

| | v2O (CUDA) | v2Q (baseline) |
|---|---|---|
| Face render time per frame | 97 ms | 27 ms |
| Rendered frames per second | ~9 | 23-35 |
| Motion frames shown per second | 12.5 | 25 |
| Speech starts after sending | ~3.3 s | ~0.9 s |
| Pauses between phrases | 0.5-1.4 s | none |
| Whole test text | 15.4 s | 10.2 s |

Before v2O (v2N.2), speech started sooner but the mouth fell up to ~0.8 s per
phrase behind the voice.

## The pipeline

```
text ─► phrases ─► Chatterbox TTS ─► JoyVASA (audio → face motion)
                                         │
                   FasterLivePortrait: warping_spade renders each frame
                                         │
             PlaybackBuffer (audio clock) ─► WebRTC audio + video ─► browser
```

Each phrase is synthesised, turned into motion by JoyVASA, and rendered frame by
frame by FasterLivePortrait (FLP). Frames are published in windows of 8, so the
first phrase can play before it is fully rendered.

## 1. TensorRT FP16: the bottleneck (v2P)

**Problem.** Profiling showed that one network, FLP's `warping_spade` (the
generator that draws the face), took 97 ms per frame through ONNX Runtime's CUDA
provider. Everything else per frame (stitching, cropping) was negligible. At
~10 frames/s the renderer could never keep up with 25 frames/s of motion, which
caused every other symptom: the mouth falling behind, pauses between phrases,
choppy motion.

**What we measured.**

| Backend | ms/frame | Output difference vs. CUDA |
|---|---|---|
| ONNX Runtime CUDA provider (FP32) | 96.8 | reference |
| ONNX Runtime TensorRT provider, FP32 | 86.9 | max 0.002 |
| **ONNX Runtime TensorRT provider, FP16** | **27.0** | mean 0.00125, max 0.018 (0-1 scale) |

FP16 is what makes the difference: about 3.5x faster, with no visible change in
side-by-side close-ups.

**How it works.**

- We did *not* use FasterLivePortrait's own TensorRT route (TensorRT 8, PyCUDA,
  a prebuilt `grid_sample_3d` plugin), which does not support the RTX 50-series
  (Blackwell, sm_120).
- Instead `server.py` loads FLP normally and then replaces the `warping_spade`
  ONNX Runtime session with one that uses the **TensorRT execution provider**
  with `trt_fp16_enable` (`_enable_tensorrt_warping`). Inputs and outputs are
  identical, so FLP does not notice.
- TensorRT cannot import the network's 5-D `GridSample`
  (`addGridSample ... nbDims == 4`). ONNX Runtime handles this automatically:
  it runs that one node on the CUDA provider and the rest of the graph in
  TensorRT. The red `ModelImporter` errors in the log at startup are this and
  are expected.
- Building the TensorRT engines takes ~30 s. They are cached in
  `./models/trt-cache` (mounted into the container), so only the first start
  after an image or driver change pays for it.
- If anything fails, the server logs it and keeps the CUDA session.
  `/health` reports `warping_backend: tensorrt-fp16` or `cuda`.
- The image installs `tensorrt-cu12-libs==10.16.1.11`. That package is 6.2 GB
  because it contains engine-builder resources for every GPU generation plus
  Windows copies; the Dockerfile keeps only the sm_120 resources (1.1 GB).

**Settings.** `AVATAR_TENSORRT=true` (default). `false` returns to the CUDA
provider.

## 2. Audio is the clock (v2O)

**Problem.** Each phrase's audio was queued at once and played in real time,
while video frames were shown as they arrived. When rendering was slower than
real time, the video queue ran dry, the last frame froze, and the video never
caught up: the mouth ended up to ~0.8 s behind the voice and kept moving after
the phrase's audio had ended.

**Fix.** Every video frame gets a timestamp on the audio timeline, and the video
track shows the frame that belongs to the audio already played. Frames that
arrive too late are skipped instead of shown late (`PlaybackBuffer`).

## 3. Speech start gate (v2O)

Skipping late frames keeps sync but makes motion choppy when rendering is slow.
So each phrase's speech waits until enough video is rendered that the rest will
arrive in time:

```
lead = (W + (duration - W) * (1 - render_speed)) * 1.15
```

`W` is one 8-frame window and `render_speed` is the measured render rate
relative to real time (seeded from the startup warm-up). With TensorRT the
render speed is above 1, so the gate normally waits 0 ms. It stays as a safety
net when the GPU is shared (e.g. TTS running at the same time).

## 4. Facial motion corrections (v2O)

All three are applied to JoyVASA's motion before FLP renders it
(`_adjust_driving_motion`), and were found by recording the avatar and looking
at close-ups of the mouth and eyes.

- **Mouth: absolute instead of relative.** FLP applies motion as
  "portrait + (current frame − first frame)". For mouth keypoints, this
  squeezed this portrait's closed-smile lips shut, so the mouth hardly opened.
  `AVATAR_LIP_MOTION_MODE=absolute` uses JoyVASA's mouth shapes directly.
- **Eyes damped to 30%.** JoyVASA's eye motion made the avatar stare and wink.
  `AVATAR_EYE_MOTION_SCALE=0.3`.
- **Head damped to 30%.** JoyVASA tilted the head 1-5 degrees up and rolled it
  up to 5 degrees, so the avatar seemed to look above the camera.
  `AVATAR_HEAD_MOTION_SCALE=0.3`.

## 5. Browser lip-sync (v2Q)

- **Shared WebRTC stream.** aiortc gives every track its own random stream id,
  and browsers only synchronise audio and video (using RTCP sender reports) for
  tracks in the same stream. The server now puts both tracks in one stream.
- **Lip-sync offset.** Measured on dumped motion, JoyVASA's mouth leads the
  loudness by 40-160 ms (about 80 ms typical). `AVATAR_LIP_SYNC_OFFSET_MS=80`
  shows the mouth 80 ms later. It felt right in the browser, and the browser's
  audio and video buffers became nearly equal (53 and 66 ms, against 34 and
  290 ms at 0 ms).
- The page shows the browser's live buffers under the status box:
  `Browser buffers: audio … ms · video … ms`.

## 6. Smoother motion (v2Q)

With TensorRT there is enough speed for **stride 1**: all 25 JoyVASA motion
frames per second are rendered, instead of every second one. When the buffer
runs low the server falls back to stride 2 (was 3), so smoothness changes less.

## Settings overview

| Setting | Default | Purpose |
|---|---|---|
| `AVATAR_TENSORRT` | `true` | TensorRT FP16 face rendering |
| `AVATAR_RENDER_STRIDE` | `1` | Render every motion frame (25 fps) |
| `AVATAR_CATCHUP_RENDER_STRIDE` | `2` | Fallback when the buffer runs low |
| `AVATAR_SPEECH_START_GATE` | `true` | Hold speech until enough video exists |
| `AVATAR_LIP_MOTION_MODE` | `absolute` | JoyVASA mouth shapes used directly |
| `AVATAR_EYE_MOTION_SCALE` | `0.3` | Damp eye motion |
| `AVATAR_HEAD_MOTION_SCALE` | `0.3` | Damp head rotation/translation |
| `AVATAR_LIP_SYNC_OFFSET_MS` | `80` | Show the mouth later (ms) |
| `AVATAR_DEBUG_DUMP_DIR` | empty | Save each phrase's WAV + motion |

## Tools used to get here

- `./review_recording.sh "text"` records one phrase exactly as a browser
  receives it and writes `recording.mp4`, frame sheets and mouth/eye close-ups to
  `results/claude-review-images/<timestamp>/`. Judge mouth shape from the
  close-ups, with the text you actually care about.
- `AVATAR_DEBUG_DUMP_DIR=/workspace/results/motion-dumps` plus
  `python3 lip_sync_analysis.py results/motion-dumps` measures, per phrase, how
  far JoyVASA's mouth leads or lags the voice.
- Metrics per phrase in the UI and benchmark JSON now include
  `video_dropped_ms`, `speech_hold_ms` and `speech_lead_ms`.

## Known limitations

- JoyVASA's mouth follows individual syllables only loosely (correlation with
  loudness 0.2-0.5), which limits how precise the lip-sync can look.
- Stride 1 is close to the render budget (23-35 fps against 25 needed), so a
  phrase occasionally falls back to stride 2.
- The output frame has a small fixed tilt with black corner wedges, from how FLP
  crops and aligns the source photo.
- Under QEMU passthrough the GPU can lock up ("GPU requires reset" in
  `nvidia-smi -q`): running processes continue but new CUDA processes fail.
  Recovery required restarting and killing the stuck GPU process.
