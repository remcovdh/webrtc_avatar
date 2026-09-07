#!/usr/bin/env bash
set -Eeuo pipefail

mode="${1:-serve}"
flp_root="/workspace/FasterLivePortrait"
flp_checkpoints="${FLP_CHECKPOINT_DIR:-${flp_root}/checkpoints}"
breeze_model="${BREEZE_MODEL_DIR:-/workspace/models/Breeze-TTS-2}"

download_models() {
  mkdir -p "${flp_checkpoints}"

  if [[ ! -f "${flp_checkpoints}/.download-complete" ]]; then
    echo "[models] Downloading FasterLivePortrait checkpoints"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='warmshao/FasterLivePortrait', local_dir='${flp_checkpoints}')"
    touch "${flp_checkpoints}/.download-complete"
  fi

  # The published warping model declares GridSample from opset 16, which only
  # accepts 4-D input. LivePortrait uses volumetric 5-D sampling, standardized
  # in opset 20. Conversion is atomic and skipped once its marker is current.
  python /workspace/patch_warping_onnx.py \
    "${flp_checkpoints}/liveportrait_onnx/warping_spade.onnx"

  if [[ ! -f "${flp_checkpoints}/JoyVASA/.download-complete" ]]; then
    echo "[models] Downloading JoyVASA checkpoints"
    mkdir -p "${flp_checkpoints}/JoyVASA"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jdh-algo/JoyVASA', local_dir='${flp_checkpoints}/JoyVASA')"
    touch "${flp_checkpoints}/JoyVASA/.download-complete"
  fi

  if [[ ! -f "${flp_checkpoints}/chinese-hubert-base/.download-complete" ]]; then
    echo "[models] Downloading JoyVASA audio encoder"
    mkdir -p "${flp_checkpoints}/chinese-hubert-base"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TencentGameMate/chinese-hubert-base', local_dir='${flp_checkpoints}/chinese-hubert-base')"
    touch "${flp_checkpoints}/chinese-hubert-base/.download-complete"
  fi

}

download_breeze_model() {
  mkdir -p "${breeze_model}"
  if [[ ! -f "${breeze_model}/.download-complete" ]]; then
    echo "[models] Downloading Breeze TTS 2"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BreezeBlue/Breeze-TTS-2', local_dir='${breeze_model}')"
    touch "${breeze_model}/.download-complete"
  fi
}

case "${mode}" in
  download)
    download_models
    ;;
  serve)
    cd "${flp_root}"
    exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8000}"
    ;;
  tts)
    download_breeze_model
    cd /workspace/breeze-tts
    breeze_fast_args=()
    if [[ -n "${BREEZE_FAST_ARGS:-}" ]]; then
      read -r -a breeze_fast_args <<< "${BREEZE_FAST_ARGS}"
    fi
    echo "[breeze] Fast-path arguments: ${BREEZE_FAST_ARGS:-<eager baseline>}"
    exec python -m breeze_infer.api "${breeze_model}" \
      --host 0.0.0.0 --port "${BREEZE_PORT:-7860}" \
      "${breeze_fast_args[@]}"
    ;;
  breeze)
    "$0" tts
    ;;
  chatterbox)
    cd /workspace
    exec uvicorn chatterbox_api:app --host 0.0.0.0 --port "${CHATTERBOX_PORT:-7860}"
    ;;
  *)
    exec "$@"
    ;;
esac
