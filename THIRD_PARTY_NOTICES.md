# Third-party notices

This repository contains only its own code, documentation and a few input
files. It does **not** contain any model weights or third-party source code:
the Docker build installs the software listed here, and the first start
downloads the models. Whoever builds, runs or redistributes the result is bound
by the licences below.

This is an overview made with care from each project's own licence statement
(checked on 3 October 2026); it is not legal advice. The exact versions are in
`docker/locks/` and `docker/Dockerfile`.

## What to know before using this beyond private experiments

- **TensorRT is proprietary.** The avatar image contains NVIDIA TensorRT
  libraries under the NVIDIA Software License Agreement, and the base image
  contains CUDA under NVIDIA's container licence. Build the images yourself;
  do not publish them.
- **A GPL library is installed in the avatar image.** FasterLivePortrait's
  requirements pull in `phonemizer` (GPL-3.0-or-later) for a text-to-speech
  route this project does not use. That matters if you distribute the image.
- **`yake` is LGPL/GPL.** The conductor uses `yake` for keyword extraction;
  its metadata names both LGPL-3.0 and GPL-3.0. It is used as an unmodified
  library.
- **The Dutch spaCy model is share-alike.** `nl_core_news_sm` is CC BY-SA 4.0.
- **InsightFace weights are not used.** FasterLivePortrait's checkpoint
  repository includes two InsightFace models (`retinaface_det_static`,
  `face_2dpose_106_static`), which are for non-commercial research only. This
  project does not download or load them; it finds the face with MediaPipe
  (`src/avatar/face_model.py`). Setting `AVATAR_FACE_DETECTOR=insightface`
  brings them back and makes that restriction apply to you. The `insightface`
  Python package (MIT) is still installed because FasterLivePortrait imports a
  helper class from it.
- **Generated speech carries a watermark.** Chatterbox marks its audio with
  Resemble AI's Perth watermark.
- **Voices and faces are personal.** Use a portrait and a reference voice you
  have the right to use, and do not imitate a real person without consent.

## Models (downloaded at first start)

| Model | Used for | Licence |
|---|---|---|
| [warmshao/FasterLivePortrait](https://huggingface.co/warmshao/FasterLivePortrait) (LivePortrait ONNX weights) | Rendering the face | MIT (see the note on InsightFace above) |
| [jdh-algo/JoyVASA](https://huggingface.co/jdh-algo/JoyVASA) | Face motion from speech | MIT |
| [TencentGameMate/chinese-hubert-base](https://huggingface.co/TencentGameMate/chinese-hubert-base) | Audio features for JoyVASA | MIT |
| [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | Finding the face in the portrait | Apache-2.0 |
| [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) | Text to speech | MIT |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | Speech recognition | [OpenMDW 1.1](https://openmdw.ai/license/1-1/) |
| [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) | Speaker labels | OpenMDW 1.1 |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Voice activity detection | MIT |
| [alibiserikbay/JevK5-GGUF](https://huggingface.co/alibiserikbay/JevK5-GGUF) (based on Qwen3.5-4B) | System 1 decisions | Apache-2.0 |
| [ibm-granite/granite-4.0-1b-GGUF](https://huggingface.co/ibm-granite/granite-4.0-1b-GGUF) | System 2 answers | Apache-2.0 |
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | Searching the knowledge folder | MIT |
| [unsloth/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) | Conversation review (run by hand) | Apache-2.0 |
| spaCy `en_core_web_sm` / `nl_core_news_sm` | Topic extraction | MIT / CC BY-SA 4.0 |

OpenMDW 1.1 permits use, modification and distribution, including commercial
use, provided the licence text and the notices of origin are kept; it ends for
anyone who starts patent or copyright litigation over the model.

Llama 3.2 (Llama 3.2 Community License) was used for System 2 up to v2Z and is
no longer downloaded.

## Software (installed by the Docker build)

| Project | Used for | Licence |
|---|---|---|
| [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) / [LivePortrait](https://github.com/KwaiVGI/LivePortrait) | Face animation | MIT |
| [JoyVASA](https://github.com/jdh-algo/JoyVASA) | Audio-driven motion | MIT |
| [Chatterbox](https://github.com/resemble-ai/chatterbox), [Perth](https://github.com/resemble-ai/Perth) | Text to speech, watermark | MIT |
| [PyTorch](https://pytorch.org), torchaudio | All models | BSD-3-Clause |
| [ONNX Runtime](https://onnxruntime.ai) | Rendering | MIT |
| NVIDIA TensorRT, CUDA, cuDNN | GPU acceleration | NVIDIA proprietary licences |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Face landmarks | Apache-2.0 |
| [aiortc](https://github.com/aiortc/aiortc), PyAV | WebRTC | BSD-3-Clause |
| [FastAPI](https://fastapi.tiangolo.com), Uvicorn, HTTPX | Web server and client | MIT / BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers), huggingface_hub, sentence-transformers | Speech and embedding models | Apache-2.0 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) via llama-cpp-python | Language models | MIT |
| [jevk5](https://github.com/allebee/jevk5) | System 1 prompt format | Apache-2.0 |
| [spaCy](https://spacy.io), RapidFuzz | Topic extraction | MIT |
| [YAKE](https://github.com/LIAAD/yake) | Keyword extraction | LGPL-3.0 / GPL-3.0 (see above) |
| OpenCV, NumPy | Image and array handling | Apache-2.0 / BSD-3-Clause |
| FFmpeg (Ubuntu package) | Audio and video handling | LGPL/GPL |

The complete package lists are `docker/locks/*.txt`.

## Input files in this repository

- `inputs/avatar.jpg`: the portrait in use, generated with Midjourney; it
  shows no real person. The other `inputs/avatar*.jpg` are earlier test
  portraits. They are test material for this project and are not covered by
  the code licence.
- `inputs/*voice-preset*`: reference recordings for the cloned voice, and
  their transcripts. Test material, not covered by the code licence; replace
  them with a voice you have the right to use.
- `tools/audio/*.wav`: short test questions spoken by the project's own TTS.
- `knowledge/*.md`: example notes about this project.
