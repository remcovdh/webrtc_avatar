# syntax=docker/dockerfile:1.7

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel AS common

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg git libgl1 libglib2.0-0 \
      libsndfile1 libsm6 libxext6 ninja-build sox \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja


# Breeze is a separate runtime target because it requires NumPy 2.x while the
# FasterLivePortrait / InsightFace stack is safer on NumPy 1.26.
FROM common AS breeze

ARG BREEZE_REF=main
ARG BREEZE_INSTALL_FLASH_ATTN=0
ARG FLASH_ATTN_CUDA_ARCHS=120
WORKDIR /workspace

RUN mkdir -p /workspace/breeze-tts \
    && curl --fail --location --retry 5 --retry-all-errors \
      "https://github.com/breezeblue-ai/breeze-tts/archive/refs/heads/${BREEZE_REF}.tar.gz" \
      | tar -xz --strip-components=1 -C /workspace/breeze-tts
WORKDIR /workspace/breeze-tts
# Breeze's eager runtime works without FlashAttention and PyTorch 2.9.1 with
# CUDA 12.8 supports sm_120. Keep FlashAttention experimental below.
RUN python -m pip install -r requirements.txt
# Experimental on RTX 50-series/Blackwell: upstream FlashAttention 2 does not
# currently list Blackwell as supported. Keep this opt-in.
RUN if [ "${BREEZE_INSTALL_FLASH_ATTN}" = "1" ]; then \
      MAX_JOBS=4 FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" \
        python -m pip install --no-build-isolation --no-deps "flash-attn==2.8.3"; \
    fi

COPY entrypoint.sh breeze_benchmark.py /workspace/
RUN chmod +x /workspace/entrypoint.sh
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["tts"]


# Chatterbox is isolated from the avatar and Breeze dependency graphs. The
# official package pins Torch 2.6, which predates the RTX 5080 stack used by
# this project, so retain the CUDA 12.8 / Torch 2.9 runtime and install the
# Chatterbox package without allowing pip to replace Torch or torchaudio.
FROM common AS chatterbox

ARG CHATTERBOX_VERSION=0.1.7
WORKDIR /workspace
RUN python -m pip install "numpy==1.26.4" \
      "librosa==0.11.0" s3tokenizer "transformers==5.2.0" \
      "diffusers==0.29.0" "safetensors==0.5.3" "conformer==0.3.2" \
      "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master" \
      spacy-pkuseg "pykakasi==2.3.0" pyloudnorm omegaconf \
      fastapi uvicorn python-multipart \
    && python -m pip install --no-deps "chatterbox-tts==${CHATTERBOX_VERSION}" \
    && python -c "import torch, torchaudio; from chatterbox.tts_turbo import ChatterboxTurboTTS; assert torch.__version__.startswith('2.9.1'); print('Chatterbox import smoke test passed')"

COPY entrypoint.sh chatterbox_api.py /workspace/
RUN chmod +x /workspace/entrypoint.sh \
    && python -m py_compile /workspace/chatterbox_api.py
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["chatterbox"]


FROM common AS avatar

ARG FASTER_LIVE_PORTRAIT_REF=master
WORKDIR /workspace
RUN mkdir -p /workspace/FasterLivePortrait \
    && curl --fail --location --retry 5 --retry-all-errors \
      "https://github.com/warmshao/FasterLivePortrait/archive/refs/heads/${FASTER_LIVE_PORTRAIT_REF}.tar.gz" \
      | tar -xz --strip-components=1 -C /workspace/FasterLivePortrait

WORKDIR /workspace/FasterLivePortrait
# The v2B server calls these upstream APIs directly to avoid video files.
# Fail the build with a clear message if the moving master branch changes the
# contract, rather than failing on the first request at runtime.
RUN grep -Fq 'def gen_motion_sequence(' src/pipelines/joyvasa_audio_to_motion_pipeline.py \
    && grep -Fq 'def run_with_pkl(' src/pipelines/faster_live_portrait_pipeline.py \
    && grep -Fq 'self.src_imgs[0], self.src_infos[0]' src/pipelines/gradio_live_portrait_pipeline.py \
    && grep -Fq '"output_fps"' src/pipelines/gradio_live_portrait_pipeline.py \
    && grep -Fq '"motion"' src/pipelines/gradio_live_portrait_pipeline.py

# FasterLivePortrait's JoyVASA loader predates PyTorch 2.6's restricted
# checkpoint loading default. Keep restricted loading enabled and narrowly
# allow its two non-tensor metadata classes only while the official motion
# checkpoint is read.
RUN joyvasa_pipeline="src/pipelines/joyvasa_audio_to_motion_pipeline.py" \
    && sed -i 's/^import math$/import argparse\nimport math/' "${joyvasa_pipeline}" \
    && sed -i 's@^        model_data = torch.load(motion_model_path, map_location="cpu")$@        with torch.serialization.safe_globals([argparse.Namespace, pathlib.PosixPath]):\n            model_data = torch.load(motion_model_path, map_location="cpu")@' "${joyvasa_pipeline}" \
    && grep -Fq 'safe_globals([argparse.Namespace, pathlib.PosixPath])' "${joyvasa_pipeline}"

# JoyVASA's custom HuBERT implementation requests attention tensors. Recent
# Transformers releases select SDPA by default, but SDPA cannot return those
# tensors, so force the supported eager backend for JoyVASA's audio encoders.
# Assert the upstream call count before editing to avoid a silent partial patch.
RUN joyvasa_model="src/models/JoyVASA/dit_talking_head.py" \
    && test "$(grep -Fc 'from_pretrained(audio_encoder_path)' "${joyvasa_model}")" -eq 5 \
    && sed -i 's/from_pretrained(audio_encoder_path)/from_pretrained(audio_encoder_path, attn_implementation="eager")/g' "${joyvasa_model}" \
    && test "$(grep -Fc 'from_pretrained(audio_encoder_path, attn_implementation="eager")' "${joyvasa_model}")" -eq 5 \
    && python -m py_compile "${joyvasa_model}"

# PyCUDA is only required by the TensorRT route. This first RTX 5080 test uses
# ONNX Runtime, so avoiding it keeps the image independent of TensorRT 8.
RUN sed '/^pycuda$/d' requirements.txt > /tmp/flp-requirements.txt \
    && python -m pip install -r /tmp/flp-requirements.txt
RUN python -m pip uninstall -y opencv-python || true

RUN python -m pip install \
      "numpy==1.26.4" \
      "opencv-python-headless>=4.10,<5" \
      "onnx==1.19.1" \
      "onnxruntime-gpu>=1.23,<1.27" \
      "torchaudio==2.9.1" \
      "torchcodec==0.9.1" \
      "aiortc>=1.13,<2" \
      "av>=14,<17" \
      "fastapi>=0.115,<1" \
      "httpx>=0.27,<1" \
      "uvicorn[standard]>=0.30,<1"

# Torchaudio 2.9 delegates audio decoding to TorchCodec. Catch missing FFmpeg
# libraries or a Torch/TorchCodec ABI mismatch while building, not on the first
# browser request.
RUN ffmpeg -v error -f lavfi -i "sine=frequency=440:duration=0.05" \
      -ar 16000 -ac 1 /tmp/torchcodec-smoke.wav \
    && python -c "import torchaudio, torchcodec; audio, rate = torchaudio.load('/tmp/torchcodec-smoke.wav'); assert rate == 16000 and audio.shape[0] == 1; print('TorchCodec WAV smoke test passed')" \
    && rm /tmp/torchcodec-smoke.wav

# Confirm that the standard ONNX Runtime package can execute the opset-20 5-D
# GridSample required by LivePortrait before accepting the image build.
COPY patch_warping_onnx.py /workspace/patch_warping_onnx.py
RUN python -m py_compile /workspace/patch_warping_onnx.py \
    && python /workspace/patch_warping_onnx.py --self-test

COPY server.py index.html benchmark_avatar.py chatterbox_api.py test_benchmark_handshake.py test_motion_continuity.py test_neural_idle_frame.py test_frame_windows.py test_tts_provider.py /workspace/FasterLivePortrait/
# test_tts_provider.py validates the Compose-level provider switch as well as
# the Python adapter. Include the manifest in the build-test fixture so the
# same test works both from the source tree and inside the image build.
COPY docker-compose.yml /workspace/FasterLivePortrait/docker-compose.yml
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh
RUN cd /workspace/FasterLivePortrait \
    && python -m unittest -v test_motion_continuity.py test_neural_idle_frame.py test_frame_windows.py test_tts_provider.py

EXPOSE 8000
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["serve"]
