# Neural avatar WebRTC test

This test connects four stages:

1. **Breeze TTS 2** produces 24 kHz mono PCM from text.
2. **JoyVASA** turns the completed speech into neural facial and head motion.
3. **FasterLivePortrait** renders that motion onto `inputs/avatar.jpg`.
4. **aiortc** sends synchronized audio and 512×512 video to the browser.

The WebRTC connection stays live, but this milestone is **prepare, then play**:
JoyVASA processes the complete utterance before playback starts. It is a useful
end-to-end test, not yet a sub-second, chunk-by-chunk animation pipeline.

## Prerequisites

- Linux host with Docker Engine, Compose v2, NVIDIA Container Toolkit, and a
  recent NVIDIA driver suitable for CUDA 12.8 / RTX 5080.
- Enough disk for FasterLivePortrait, JoyVASA, and Breeze TTS 2 checkpoints.
- Research/non-commercial use unless you have separately obtained appropriate
  model rights. Breeze TTS 2 open weights and self-hosted output are licensed
  for research and non-commercial use.

## Start

Put a clear, front-facing image at:

```text
inputs/avatar.jpg
```

Then run:

```bash
mkdir -p inputs checkpoints models results
docker compose build
docker compose up
```

The first start downloads all model checkpoints and can take a long time. Open
`http://localhost:8000`, click **Enable audio**, then generate a short sentence.

The Compose file uses host networking because aiortc opens dynamic UDP sockets;
publishing only TCP port 8000 is not enough for WebRTC media from a Docker bridge.
Host networking is intended for this Linux test setup. For internet deployment,
put the HTTP endpoint behind TLS and configure a TURN server through
`ICE_SERVERS_JSON`.

## Verify and diagnose

Check the service health payload:

```bash
curl http://127.0.0.1:8000/health
```

Healthy output should name the RTX 5080 and include `CUDAExecutionProvider`.
Useful logs:

```bash
docker compose logs -f model-init
docker compose logs -f breeze-tts
docker compose logs -f webrtc-avatar
nvidia-smi
```

If the avatar service reports no CUDA execution provider, the ONNX models may
fall back to CPU. FasterLivePortrait's upstream ONNX path has historically had
special handling for five-dimensional `grid_sample`; this is the most likely
GPU compatibility point to investigate on a new Blackwell card.

If the GPU runs out of memory, keep Breeze's `--fast-all` option disabled (the
provided configuration does) and test with one browser/client. The two services
intentionally use separate containers because Breeze requires NumPy 2.x while
the portrait stack is pinned to NumPy 1.26.

FlashAttention 2 is deliberately not installed in the Breeze image: its
documented CUDA support covers Ampere, Ada, and Hopper, while the RTX 5080 is
Blackwell (`sm_120`). This test therefore uses Breeze's supported eager path on
PyTorch 2.9.1 / CUDA 12.8.

## What to improve next

- Replace JoyVASA's whole-utterance motion generation with a stateful sliding
  window and feed rendered frames directly into the WebRTC buffer.
- Add bounded backpressure instead of storing an entire rendered clip in RAM.
- Add TURN/TLS for clients outside the Docker host or local network.
- Benchmark ONNX Runtime against a Blackwell-compatible TensorRT engine once the
  FasterLivePortrait plugin path supports the installed TensorRT generation.
