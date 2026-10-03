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


# Chatterbox is isolated from the avatar dependency graph. The
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

# Every image keeps the project code under /workspace/app in the repo's own
# layout (src/<package>, tests/<package>), with src on the Python path.
ENV PYTHONPATH=/workspace/app/src
COPY docker/entrypoint.sh /workspace/entrypoint.sh
COPY src/tts /workspace/app/src/tts
RUN chmod +x /workspace/entrypoint.sh \
    && python -m py_compile /workspace/app/src/tts/chatterbox_api.py
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["chatterbox"]


# Listening worker: VAD + streaming ASR + diarization. Isolated from the avatar
# stage (NumPy 1.26, FLP) because the Nemotron speech models need a
# Transformers build newer than any release (streaming RNNT and Sortformer
# support landed on main in 2026); pinned to a commit for reproducibility.
FROM common AS listener

ARG TRANSFORMERS_COMMIT=27166ea03f12c940f23176a904ab1d2ff1a3dcbb
WORKDIR /workspace
RUN python -m pip install \
      "transformers @ git+https://github.com/huggingface/transformers@${TRANSFORMERS_COMMIT}" \
      "silero-vad>=6,<7" librosa soundfile
ENV PYTHONPATH=/workspace/app/src
COPY src/shared /workspace/app/src/shared
COPY src/listener /workspace/app/src/listener
COPY tests/listener /workspace/app/tests/listener
RUN python -m unittest discover -v -s /workspace/app/tests/listener
CMD ["python", "-m", "listener.worker"]


# Conductor: owns the conversation (turn-taking, replies, logging). Its own
# image so System 1 (Laya, M3) and System 2 (M4) can bring their own
# dependencies without touching the avatar image.
FROM common AS conductor

WORKDIR /workspace
# System 1: classical CPU topic extraction with small spaCy models. The JevK5
# decision model itself runs on llama.cpp (installed below); jevk5 only
# provides its prompt format and option letters.
RUN python -m pip install "spacy==3.8.16" "yake==0.7.3" "rapidfuzz==3.14.6" \
      "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
      "nl_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-3.8.0/nl_core_news_sm-3.8.0-py3-none-any.whl"
# System 2 (M4): llama.cpp in-process for a small 4-bit LLM, compiled for the
# RTX 50-series only (sm_120) to keep the build short; sentence-transformers
# for CPU retrieval embeddings over the knowledge folder.
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" \
      python -m pip install "llama-cpp-python==0.3.35" "sentence-transformers==6.1.0"
RUN python -m pip install --no-deps "jevk5 @ git+https://github.com/allebee/jevk5@v0.3.0"
ENV PYTHONPATH=/workspace/app/src
COPY src/shared /workspace/app/src/shared
COPY src/conductor /workspace/app/src/conductor
COPY tests/conductor /workspace/app/tests/conductor
# The image's own copy for the tests and as the default; Compose mounts the
# live ./config over /workspace/config.
COPY config/system1.json /workspace/app/config/system1.json
COPY config/system1.json /workspace/config/system1.json
RUN python -m unittest discover -v -s /workspace/app/tests/conductor
CMD ["python", "-m", "conductor.app"]


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

# TensorRT runtime libraries for ONNX Runtime's TensorRT execution provider.
# FP16 TensorRT renders warping_spade in ~27 ms instead of ~97 ms with the
# CUDA provider on the RTX 5080. Its 5-D GridSample is unsupported by TensorRT
# and stays on the CUDA provider; ORT partitions the graph automatically.
# This does not use FasterLivePortrait's TensorRT 8 / PyCUDA route.
# The package ships engine-builder resources for every GPU generation plus
# Windows cross-compilation copies (6.2 GB); keep only what an sm_120
# (RTX 50-series) engine build needs (~1.1 GB).
RUN python -m pip install "tensorrt-cu12-libs==10.16.1.11" \
    && cd /opt/conda/lib/python3.11/site-packages/tensorrt_libs \
    && find . -name 'libnvinfer_builder_resource_*' \
         ! -name 'libnvinfer_builder_resource_sm120.so.*' -delete \
    && test -f libnvinfer.so.10 && ls libnvinfer_builder_resource_sm120.so.*
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}

# Torchaudio 2.9 delegates audio decoding to TorchCodec. Catch missing FFmpeg
# libraries or a Torch/TorchCodec ABI mismatch while building, not on the first
# browser request.
RUN ffmpeg -v error -f lavfi -i "sine=frequency=440:duration=0.05" \
      -ar 16000 -ac 1 /tmp/torchcodec-smoke.wav \
    && python -c "import torchaudio, torchcodec; audio, rate = torchaudio.load('/tmp/torchcodec-smoke.wav'); assert rate == 16000 and audio.shape[0] == 1; print('TorchCodec WAV smoke test passed')" \
    && rm /tmp/torchcodec-smoke.wav

# Confirm that the standard ONNX Runtime package can execute the opset-20 5-D
# GridSample required by LivePortrait before accepting the image build.
ENV PYTHONPATH=/workspace/app/src
COPY src/avatar/__init__.py src/avatar/patch_warping_onnx.py /workspace/app/src/avatar/
RUN python -m avatar.patch_warping_onnx --self-test

COPY src/shared /workspace/app/src/shared
COPY src/avatar /workspace/app/src/avatar
# Not run here; test_tts_provider.py checks the adapter's source.
COPY src/tts /workspace/app/src/tts
COPY scripts /workspace/app/scripts
COPY tests/avatar /workspace/app/tests/avatar
COPY tests/scripts /workspace/app/tests/scripts
# Some tests check docker-compose.yml, .env.example and the host scripts
# against the code, so the image carries them at the same relative paths.
COPY docker-compose.yml .env.example /workspace/app/
COPY docker/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh
# The working directory stays FasterLivePortrait's: its own `src` package and
# relative model paths are resolved from there.
WORKDIR /workspace/FasterLivePortrait
RUN python -m unittest discover -v -s /workspace/app/tests/avatar \
    && python -m unittest discover -v -s /workspace/app/tests/scripts

EXPOSE 8000
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["serve"]
