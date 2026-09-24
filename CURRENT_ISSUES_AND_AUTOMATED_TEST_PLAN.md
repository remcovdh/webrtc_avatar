# Neural Avatar: Current Issues and Automated Test Plan

**Status date:** 2026-09-24  
**Current build:** `neural-avatar-v2n2-1-build-fixture-fix`  
**Target hardware:** NVIDIA GeForce RTX 5080 Laptop GPU  
**Purpose:** preserve the facts learned so far and replace manual, subjective iteration with repeatable automated evidence.

## 1. Original goal

Build a locally deployable streaming avatar that combines:

- Chatterbox or Breeze TTS 2;
- JoyVASA audio-to-motion generation;
- FasterLivePortrait facial rendering on the GPU;
- FastAPI, aiortc, and an HTML5 WebRTC client;
- progressive phrase playback with low initial delay;
- credible lip synchronization, mouth articulation, eye direction, and transitions;
- a single RTX 5080 laptop deployment where practical.

The current application is a functional phrase-streaming prototype. It is not yet a truly continuous real-time avatar.

## 2. Current architecture

```text
Browser text
    -> Chatterbox/Breeze creates phrase audio
    -> JoyVASA creates the complete phrase motion sequence
    -> FasterLivePortrait renders selected motion frames
    -> PlaybackBuffer schedules audio and video
    -> aiortc sends WebRTC audio/video to the browser
```

Important architectural limitation: both TTS and JoyVASA currently operate on complete phrases. FasterLivePortrait can publish rendered frame windows progressively, but it does not receive continuously generated motion.

## 3. What currently works

- Docker Compose starts the complete system.
- PyTorch, CUDA, and ONNX Runtime CUDA work on the RTX 5080 laptop.
- Chatterbox Turbo and Breeze TTS 2 are selectable providers.
- Preset voice cloning works with a reference WAV and exact transcript.
- WebRTC connection, data-channel control, browser audio, and browser video work.
- Direct-memory rendering avoids temporary pickle, MP4, FFmpeg, and decode overhead.
- Phrase splitting and progressive phrase scheduling work.
- TTS prefetch, adaptive rendering, persistent phrase motion, and neural idle-frame support exist.
- The v2N.1 gate prevents audio from starting before the first incremental video window.
- Repeatable benchmark scripts can record WebRTC sessions and collect timing data.
- Visual settings are reported by `/health` and phrase metrics.
- The latest container build-time tests pass.

## 4. Confirmed current configuration

The current controlled visual test is correctly running with:

```text
animation_region: all
driving_multiplier: 1.0
configured_render_stride: 1
adaptive_render_stride: false
incremental_frame_windows: false
tts_prefetch: false
idle_frame_source: original-avatar
normalize_lip: true
eye_retargeting: false
lip_retargeting: false
```

This means the present facial-quality problem is **not explained by adaptive stride, incremental delivery, TTS prefetch, browser caching, or an incorrect Compose environment**.

The `.env` file currently contains `AVATAR_USE_NEURAL_IDLE_FRAME` twice. The last value wins. Remove the earlier duplicate for clarity, but this is not the cause of the animated facial problem.

## 5. Main unresolved issues

### 5.1 Unnatural eye direction

With `AVATAR_ANIMATION_REGION=all`, some generated sequences drive both eyes upward, making the avatar appear to look at the ceiling. The source portrait itself is sufficiently frontal and camera-facing.

`AVATAR_EYE_RETARGETING=false` does not lock the eyes. It only disables FasterLivePortrait's explicit eye-retargeting feature; JoyVASA-provided eye/expression motion may still move them.

Using `AVATAR_ANIMATION_REGION=lip` suppresses the unwanted eye motion, but it also freezes almost all non-mouth behavior and produces an unnaturally static face.

### 5.2 Unstable or implausible mouth movement

Lip timing has sometimes appeared approximately correct, but mouth geometry remains inconsistent. Observed symptoms include:

- insufficient opening for audible vowels;
- excessive or unstable deformation;
- lip-only animation moving the mouth unnaturally;
- visible motion that does not look capable of producing the current sound;
- results varying between tests that appeared to use similar settings.

The `lip` region with multiplier `1.15` was a diagnostic experiment, not the previously accepted baseline. It freezes the eyes as expected but is not currently suitable as the production configuration for this portrait.

### 5.3 Lip synchronization versus mouth articulation

These must be measured separately:

- **Synchronization:** whether an audio phoneme and the corresponding visual mouth event occur at the same time.
- **Articulation:** whether the generated mouth shape and opening are visually plausible.

The v2N.1 playback gate improved synchronization at playback start. It did not improve JoyVASA motion quality or FLP mouth shapes.

### 5.4 Full-face motion versus camera-facing stability

The current controls are too coarse:

- `all` provides a more alive face but permits bad gaze motion;
- `lip` keeps the eyes camera-facing but removes natural facial behavior and may destabilize mouth appearance;
- eye and mouth strength cannot currently be adjusted independently.

### 5.5 Renderer throughput

Measured FasterLivePortrait throughput is usually about 11-12 newly rendered neural frames per second. The motion source is approximately 25 FPS.

- Stride 1 requests 25 unique neural frames per second and cannot stream in real time on the current ONNX path.
- Stride 2 plays at approximately 12.5 unique FPS and is the closest stable 5080-only compromise.
- Stride 3 helps the buffer catch up but reduces mouth-motion temporal detail to approximately 8.33 FPS.
- A stride-1 visual profile can be synchronized only by rendering the complete phrase before playback.

Therefore stride-1 quality recordings and real-time streaming recordings are different experiments and must never be compared as though they use the same playback policy.

### 5.6 Phrase batching is still visible

Progressive phrase playback reduces initial delay, but TTS and motion are still generated by phrase. Transitions are improved by persistent relative motion, yet the result is not fluid continuous speech animation.

## 6. Important conclusions and discarded assumptions

1. Rebuilding Docker repeatedly does not fix runtime visual behavior when the image and code are unchanged.
2. `.env` changes require container recreation, not an image rebuild.
3. The browser does not create malformed eye or mouth geometry; those pixels are already present in rendered frames.
4. Disabling eye retargeting does not disable JoyVASA eye motion.
5. `animation_region=lip` is useful for diagnosis but is not automatically a production solution.
6. Increasing the driving multiplier is not a general lip-sync fix; it can amplify artifacts.
7. The previous black video was associated with lip retargeting. Keep `AVATAR_LIP_RETARGETING=false` unless that path is deliberately repaired and retested.
8. A/V start gating and mouth articulation are independent problems.
9. Timing tables alone cannot prove visual quality.
10. Comparing runs with different generated audio, text, source images, render profiles, or model files is invalid.

## 7. Most important next diagnostic

Run the current visual baseline through the automated recorder:

```bash
QUALITY_BENCHMARK_PROFILE=visual \
QUALITY_BENCHMARK_SCENARIOS='baseline,all,1.0,true,false,false' \
QUALITY_BENCHMARK_WARMUPS=1 \
QUALITY_BENCHMARK_REPEATS=1 \
QUALITY_BENCHMARK_BUILD=0 \
./quality_benchmark.sh
```

Retain at least:

- the measured MP4;
- `health.json`;
- `benchmark.json`;
- `phrases.csv`;
- `container.log`;
- resolved Compose configuration.

Interpretation:

| Recorded MP4 | Browser result | Conclusion |
|---|---|---|
| Bad | Bad | Motion generation or FLP rendering problem |
| Good | Bad | Browser/WebRTC presentation or timestamp problem |
| Different across identical runs | Either | Nondeterministic input or motion generation |

## 8. Required automated test architecture

The next development version should turn every experiment into a reproducible test bundle.

### 8.1 Immutable run manifest

Every run must record:

- application build/version;
- Git commit when available;
- Docker image IDs;
- GPU name and driver;
- PyTorch, CUDA, ONNX Runtime, Chatterbox, JoyVASA, and FLP versions;
- complete resolved Compose environment;
- SHA-256 hashes of the source portrait, reference WAV, transcript, checkpoints, and configuration files;
- random seeds and deterministic-mode flags;
- exact text and voice mode;
- timestamps and run ID.

### 8.2 Fixed inputs

The test harness must support two distinct modes:

1. **End-to-end mode:** generate speech through the selected TTS provider.
2. **Fixed-driving-WAV mode:** bypass TTS and animate from one stored target-speech WAV.

Fixed-driving-WAV mode is essential. It separates TTS variability from JoyVASA and FLP behavior and makes before/after visual comparisons valid.

The generated phrase WAV from every end-to-end run should also be saved and hashed.

### 8.3 Stage artifacts

Save output from every pipeline boundary:

```text
request.json
tts-output.wav
tts-output.sha256
joyvasa-motion.npz
joyvasa-motion-summary.json
rendered-frames-or-contact-sheet
rendered-video.mp4
webrtc-recording.mp4
events.json
health.json
compose-resolved.yaml
container.log
metrics.json
report.md
```

The JoyVASA dump should include motion, eye ratios, lip ratios, output FPS, shapes, ranges, and checksums.

### 8.4 Objective measurements

Add automated metrics for:

- audio start versus first meaningful mouth movement;
- audio-energy envelope versus mouth-opening curve;
- estimated best A/V offset and confidence;
- mouth-opening minimum, maximum, range, and jitter;
- percentage of voiced frames with an implausibly closed mouth;
- eye/gaze displacement from the source portrait;
- blink rate and long abnormal eye holds;
- head-pose range and sudden jumps;
- unique neural FPS, repeated-frame ratio, audio gaps, and video holds;
- phrase-boundary position and expression discontinuity.

These metrics do not replace human review, but they prevent obviously worse versions from being accepted.

### 8.5 Controlled scenario matrix

Use one fixed WAV and source image while changing one variable at a time:

| Scenario | Region | Multiplier | Normalize lip | Eye scale | Render profile |
|---|---|---:|:---:|---:|---|
| Baseline visual | all | 1.00 | true | 1.00 | stride-1 batch |
| Baseline runtime | all | 1.00 | true | 1.00 | stride-2 incremental |
| Eye stabilized | all | 1.00 | true | 0.25 | stride-1 batch |
| Eyes locked | all | 1.00 | true | 0.00 | stride-1 batch |
| Lip normalization off | all | 1.00 | false | chosen winner | stride-1 batch |
| Mild mouth scaling | all | 1.10 | chosen winner | chosen winner | stride-1 batch |

Do not combine multiple unproven changes in one scenario.

### 8.6 Automatic comparison report

Generate one report containing:

- exact settings and hashes;
- timing and quality metrics;
- synchronized side-by-side or tiled videos;
- mouth-region and eye-region crops;
- contact sheets at fixed audio timestamps;
- plots of audio energy, mouth opening, and gaze displacement;
- pass/fail thresholds;
- direct links to every artifact.

## 9. Proposed next code version

The next version should focus on observability and controlled eye behavior rather than more ad-hoc `.env` tuning.

Recommended changes:

1. Add fixed-driving-WAV benchmark input.
2. Save each generated target-speech WAV.
3. Save and hash JoyVASA motion arrays.
4. Seed Python, NumPy, PyTorch CPU, and CUDA immediately before TTS and motion generation.
5. Add `AVATAR_EYE_MOTION_SCALE` while retaining `animation_region=all`:
   - `1.0`: original eye motion;
   - `0.2-0.4`: restrained eye motion;
   - `0.0`: source-facing eye lock.
6. Keep mouth/face expression motion independent from eye scaling.
7. Add strict Boolean parsing so values such as `fakse` fail startup with a clear error.
8. Add a browser control or API option to record the exact current request automatically.
9. Add regression tests for run manifests, fixed WAVs, motion capture, and configuration validation.

Eye scaling must be implemented and evaluated against captured JoyVASA eye data; it should not be approximated by switching the whole animation region to `lip`.

## 10. Two valid operating profiles

### Visual-quality profile

Use this to judge facial correctness:

```dotenv
AVATAR_RENDER_STRIDE=1
AVATAR_ADAPTIVE_RENDER_STRIDE=false
AVATAR_INCREMENTAL_FRAME_WINDOWS=false
AVATAR_TTS_PREFETCH=false
```

This profile is intentionally not real time. It renders a complete phrase before synchronized playback.

### Closest current RTX 5080 streaming profile

Use this only after the facial-motion winner is selected:

```dotenv
AVATAR_RENDER_STRIDE=2
AVATAR_ADAPTIVE_RENDER_STRIDE=false
AVATAR_INCREMENTAL_FRAME_WINDOWS=true
AVATAR_RENDER_WINDOW_FRAMES=8
AVATAR_TTS_PREFETCH=true
AVATAR_TTS_PREFETCH_POLICY=adaptive
```

Fixed stride 2 is preferred during quality validation because automatic stride 3 discards additional mouth-motion detail. Adaptive catch-up can be reconsidered after objective quality thresholds exist.

## 11. Longer-term real-time direction

Further phrase-size tuning cannot remove the main architectural limits. For truly continuous sub-second interaction, compare the optimized FLP path against an online renderer such as Ditto using the same fixed audio, portrait, recording method, and metrics.

Possible optimization sequence:

1. Make the current pipeline fully reproducible.
2. Stabilize eye and mouth behavior with objective regression tests.
3. Profile FLP TensorRT/FP16 as a time-boxed experiment.
4. Test TTS on the RTX 4080 laptop and rendering on the RTX 5080 laptop, measuring network overhead.
5. Build a Ditto online proof of concept.
6. Select the renderer using identical end-to-end benchmarks rather than claims or subjective memory.

## 12. Definition of done

A candidate version should not be called improved unless:

- the complete run can be reproduced from its manifest;
- fixed input produces stable hashes or documented bounded variation;
- audio and mouth movement start within the chosen synchronization threshold;
- voiced sections do not remain implausibly closed;
- eye direction stays within the accepted camera-facing range;
- no black frames, severe mouth warping, or large phrase-boundary jumps occur;
- real-time profile audio gaps and video holds stay within defined limits;
- both automated metrics and human review beat the stored baseline;
- all output artifacts are retained under one versioned run directory.

## 13. Working rule from now on

Do not accept conclusions from memory, isolated screenshots, or differently configured runs. Every change should follow:

```text
fixed inputs -> one controlled change -> automatic capture -> objective comparison -> human review -> documented decision
```

This is the path out of repeated configuration loops and toward a measurable real-time avatar.
