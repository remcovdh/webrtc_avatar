# Sana on an RTX 5080 Laptop (16 GB) — Docker

This container is tuned for **NVIDIA consumer Blackwell / sm_120**, using the current Sana dependency baseline: **Python 3.11, PyTorch 2.9.1, CUDA 12.8**.

## What will fit on a 5080 Laptop

| Sana family | 16 GB RTX 5080 Laptop | Suggested mode |
|---|---|---|
| Sana 0.6B image | Yes | BF16 on GPU |
| Sana / Sana 1.5 1.6B image | Yes | BF16 on GPU; use `--offload model` if another app is using VRAM |
| 4-bit Sana image | Yes | Use the repo's quantized path if desired |
| SANA-Video 480p 2B | Try with offload | Start with `video.py` + `--offload model`; reduce frames if needed |
| SANA-Video 720p / LTX2 refiner | Not guaranteed in 16 GB | Strong offload / smaller workloads; expect slower execution |
| SANA-WM | No, not fully in VRAM | Repo documents ~25–29.4 GB even in its tight Blackwell FP4 profiles |

Docker isolates software; it cannot increase physical VRAM.

## Host setup

### Windows 11

Use a current NVIDIA Windows driver plus Docker Desktop with the WSL2 backend and GPU support enabled. Do **not** install a Linux NVIDIA display driver inside WSL; the Windows driver is exposed into WSL.

### Linux

Install a current NVIDIA driver, Docker, and NVIDIA Container Toolkit. Verify the host can run:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

## Build

```bash
docker compose build
```

The build intentionally compiles FlashAttention from source with `sm_120` as the target architecture. This is more reliable for RTX 50-series than relying on older prebuilt FlashAttention wheels.

If you only want Sana image inference and want a faster/lighter build:

```bash
docker compose build --build-arg INSTALL_FLASH_ATTN=0
```

To additionally attempt Transformer Engine (useful for SANA-WM fp8/fp4, but SANA-WM still does not fit 16 GB):

```bash
docker compose build --build-arg INSTALL_TRANSFORMER_ENGINE=1
```

## Verify GPU + CUDA

```bash
docker compose run --rm sana python /opt/launchers/verify_gpu.py
```

You want to see:

- GPU name: RTX 5080 Laptop GPU
- compute capability: `sm_120`
- PyTorch CUDA: `12.8`
- `sm_120` present in `torch.cuda.get_arch_list()`
- BF16 matmul: OK

## Generate a 1024×1024 image

```bash
docker compose run --rm sana python /opt/launchers/image.py \
  --prompt "cinematic photo of a futuristic Amsterdam canal at blue hour, highly detailed"
```

Output: `./outputs/sana.png`

If you hit an OOM because the desktop/browser/other apps are consuming VRAM:

```bash
docker compose run --rm sana python /opt/launchers/image.py \
  --offload model \
  --prompt "cinematic photo of a futuristic Amsterdam canal at blue hour, highly detailed"
```

For maximum memory savings (much slower), use `--offload sequential`.

## Generate SANA-Video at a conservative 480p profile

```bash
docker compose run --rm sana python /opt/launchers/video.py \
  --prompt "A small robot walks through a rainy neon alley, cinematic camera movement"
```

The launcher starts at 49 frames and model CPU offload. If stable, try the repo's 81-frame example:

```bash
docker compose run --rm sana python /opt/launchers/video.py \
  --frames 81 \
  --prompt "A small robot walks through a rainy neon alley, cinematic camera movement"
```

If it still OOMs, use `--offload sequential`, reduce `--frames`, or reduce width/height. CPU offload also needs adequate system RAM and is slower than keeping everything on the GPU.

## Run the Sana repo directly

The full Sana checkout is at `/opt/Sana` inside the image:

```bash
docker compose run --rm sana bash
cd /opt/Sana
```

Then you can run the upstream scripts/configs directly. Hugging Face downloads persist under `./cache/huggingface` and outputs under `./outputs`.

## Pin Sana to a commit/tag

For reproducibility, edit `SANA_REF` in `compose.yaml`, or build with:

```bash
docker build --build-arg SANA_REF=<git-tag-or-commit> -t sana-5080:cu128 .
```

## Notes

- The image is large because it uses the PyTorch CUDA **devel** base so CUDA extensions can compile for `sm_120`.
- The default `INSTALL_TRANSFORMER_ENGINE=0` is intentional: the 5080 Laptop has 16 GB VRAM and SANA-WM's documented low-precision configurations still exceed it.
- "Perfect" execution depends on host driver version, laptop GPU power limit/thermals, system RAM, and the exact Sana checkpoint. The included smoke test verifies the CUDA/PyTorch/Blackwell path before downloading large models.
