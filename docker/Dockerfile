# syntax=docker/dockerfile:1.7

# Reproducible builds: the base image is pinned by digest, source checkouts by
# commit, and every Python package by docker/locks/<service>.txt, which pip
# gets as a constraints file and docker/check_lock.py verifies at the end of
# each stage. Changing a dependency on purpose: see docker/locks/README.md.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel@sha256:39236c0ad9c66baecf01bb2e4f5562543c5b336c0c785887798775d6d6fdbf9a AS common

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

RUN python -m pip install "pip==26.2.1" "setuptools==84.0.0" "wheel==0.48.0" \
      "packaging==26.3" "ninja==1.13.2"
COPY docker/check_lock.py /opt/locks/check_lock.py
# Set to 0 only while changing dependencies (docker/locks/README.md).
ARG VERIFY_LOCK=1


# Chatterbox is isolated from the avatar dependency graph. The
# official package pins Torch 2.6, which predates the RTX 5080 stack used by
# this project, so retain the CUDA 12.8 / Torch 2.9 runtime and install the
# Chatterbox package without allowing pip to replace Torch or torchaudio.
FROM common AS chatterbox

ARG CHATTERBOX_VERSION=0.1.7
ARG VERIFY_LOCK
WORKDIR /workspace
COPY docker/locks/tts.txt /opt/locks/tts.txt
RUN python -m pip install -c /opt/locks/tts.txt "numpy==1.26.4" \
      "librosa==0.11.0" s3tokenizer "transformers==5.2.0" \
      "diffusers==0.29.0" "safetensors==0.5.3" "conformer==0.3.2" \
      "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@ff1c8ac55a976971245cdd53c18d6131ca00d993" \
      spacy-pkuseg "pykakasi==2.3.0" pyloudnorm omegaconf \
      fastapi uvicorn python-multipart \
    && python -m pip install --no-deps "chatterbox-tts==${CHATTERBOX_VERSION}" \
    && if [ "${VERIFY_LOCK}" = "1" ]; then python /opt/locks/check_lock.py /opt/locks/tts.txt; fi \
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
ARG VERIFY_LOCK
WORKDIR /workspace
COPY docker/locks/listener.txt /opt/locks/listener.txt
RUN python -m pip install -c /opt/locks/listener.txt \
      "transformers @ git+https://github.com/huggingface/transformers@${TRANSFORMERS_COMMIT}" \
      "silero-vad>=6,<7" librosa soundfile \
    && if [ "${VERIFY_LOCK}" = "1" ]; then python /opt/locks/check_lock.py /opt/locks/listener.txt; fi
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
ARG VERIFY_LOCK
COPY docker/locks/conductor.txt /opt/locks/conductor.txt
RUN python -m pip install -c /opt/locks/conductor.txt "spacy==3.8.16" "yake==0.7.3" "rapidfuzz==3.14.6" \
      "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
      "nl_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-3.8.0/nl_core_news_sm-3.8.0-py3-none-any.whl"
# System 2 (M4): llama.cpp in-process for a small 4-bit LLM, compiled for the
# RTX 50-series only (sm_120) to keep the build short; sentence-transformers
# for CPU retrieval embeddings over the knowledge folder.
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" \
      python -m pip install -c /opt/locks/conductor.txt \
      "llama-cpp-python==0.3.35" "sentence-transformers==6.1.0"
# jevk5 v0.3.0, by commit: a tag can be moved.
RUN python -m pip install --no-deps \
      "jevk5 @ git+https://github.com/allebee/jevk5@6c6522fe5462a05fdb82bceeb0e8c624c11f1517" \
    && if [ "${VERIFY_LOCK}" = "1" ]; then python /opt/locks/check_lock.py /opt/locks/conductor.txt; fi
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

# The commit this code was written against (master of 2025-06-29).
ARG FASTER_LIVE_PORTRAIT_REF=8aad3602177547aaa5e4beec0c3ef5b7944e7a1f
ARG VERIFY_LOCK
WORKDIR /workspace
# MediaPipe (the face detector, see src/avatar/face_model.py) needs EGL/GLES
# libraries present even when it runs on the CPU.
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/locks/avatar.txt /opt/locks/avatar.txt
RUN mkdir -p /workspace/FasterLivePortrait \
    && curl --fail --location --retry 8 --retry-all-errors --silent --show-error \
      -o /tmp/flp.tar.gz \
      "https://github.com/warmshao/FasterLivePortrait/archive/${FASTER_LIVE_PORTRAIT_REF}.tar.gz" \
    && tar -xzf /tmp/flp.tar.gz --strip-components=1 -C /workspace/FasterLivePortrait \
    && rm /tmp/flp.tar.gz

WORKDIR /workspace/FasterLivePortrait
# The renderer calls these upstream APIs directly. Fail the build with a clear
# message if a newer pinned commit changes the contract, rather than failing
# on the first request at runtime.
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

# PyCUDA is only required by FasterLivePortrait's own TensorRT 8 route, which
# is not used. opencv-python is replaced by the headless build below; leaving
# it here makes pip build an old release from source to fit the pinned NumPy.
# NumPy is left out of the constraints here: MediaPipe's OpenCV build asks for
# NumPy 2, so pip cannot resolve this step with the 1.26 that the next step
# installs on purpose (FasterLivePortrait and InsightFace need 1.x).
RUN sed '/^pycuda$/d; /^opencv-python$/d' requirements.txt > /tmp/flp-requirements.txt \
    && grep -v '^numpy==' /opt/locks/avatar.txt > /tmp/flp-constraints.txt \
    && python -m pip install -c /tmp/flp-constraints.txt -r /tmp/flp-requirements.txt
# A dependency still pulls opencv-python in; the headless build replaces it.
RUN python -m pip uninstall -y opencv-python

RUN python -m pip install -c /opt/locks/avatar.txt \
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
# The wheel is fetched from NVIDIA's index by its checksum and unpacked
# directly, leaving the unneeded resources out: installing it with pip needs
# more than 10 GB of free disk for a moment, this needs about 5.5 GB.
ARG TENSORRT_WHEEL=tensorrt_cu12_libs-10.16.1.11-py3-none-manylinux_2_28_x86_64.whl
ARG TENSORRT_SHA256=8e45036efeb964d323231544442a73619201136ccc84392560254cc8f0d516e4
RUN curl --fail --location --retry 5 --retry-all-errors --silent --show-error \
      -o "/tmp/${TENSORRT_WHEEL}" "https://pypi.nvidia.com/tensorrt-cu12-libs/${TENSORRT_WHEEL}" \
    && echo "${TENSORRT_SHA256}  /tmp/${TENSORRT_WHEEL}" | sha256sum --check --status \
    && python -c "import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); keep = [n for n in z.namelist() if 'libnvinfer_builder_resource_' not in n or 'libnvinfer_builder_resource_sm120.so' in n]; z.extractall(sys.argv[2], keep); print(len(keep), 'of', len(z.namelist()), 'files unpacked')" \
         "/tmp/${TENSORRT_WHEEL}" /opt/conda/lib/python3.11/site-packages \
    && rm "/tmp/${TENSORRT_WHEEL}" \
    && cd /opt/conda/lib/python3.11/site-packages/tensorrt_libs \
    && test -f libnvinfer.so.10 && ls libnvinfer_builder_resource_sm120.so.* \
    && if [ "${VERIFY_LOCK}" = "1" ]; then python /opt/locks/check_lock.py /opt/locks/avatar.txt; fi
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
