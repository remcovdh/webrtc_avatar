#!/usr/bin/env bash
set -Eeuo pipefail

mode="${1:-serve}"
flp_root="/workspace/FasterLivePortrait"
flp_checkpoints="${FLP_CHECKPOINT_DIR:-${flp_root}/checkpoints}"

download_models() {
  mkdir -p "${flp_checkpoints}"

  if [[ ! -f "${flp_checkpoints}/.download-complete" ]]; then
    echo "[models] Downloading FasterLivePortrait checkpoints"
    # Pinned to the revision this code was tested with. The two InsightFace
    # models in that repository are skipped: their weights are for
    # non-commercial research only and the avatar finds the face with MediaPipe.
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='warmshao/FasterLivePortrait', revision='eb937f4bec7186598df2d1f68e1ddbb488ae1de5', local_dir='${flp_checkpoints}', ignore_patterns=['*retinaface*', '*face_2dpose*'])"
    touch "${flp_checkpoints}/.download-complete"
  fi

  # The published warping model declares GridSample from opset 16, which only
  # accepts 4-D input. LivePortrait uses volumetric 5-D sampling, standardized
  # in opset 20. Conversion is atomic and skipped once its marker is current.
  python -m avatar.patch_warping_onnx \
    "${flp_checkpoints}/liveportrait_onnx/warping_spade.onnx"

  if [[ ! -f "${flp_checkpoints}/JoyVASA/.download-complete" ]]; then
    echo "[models] Downloading JoyVASA checkpoints"
    mkdir -p "${flp_checkpoints}/JoyVASA"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jdh-algo/JoyVASA', revision='b8f13fe9c23679c56f21b1baafb92ed00dc087c3', local_dir='${flp_checkpoints}/JoyVASA')"
    touch "${flp_checkpoints}/JoyVASA/.download-complete"
  fi

  if [[ ! -f "${flp_checkpoints}/chinese-hubert-base/.download-complete" ]]; then
    echo "[models] Downloading JoyVASA audio encoder"
    mkdir -p "${flp_checkpoints}/chinese-hubert-base"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TencentGameMate/chinese-hubert-base', revision='fce0375452b1dd6c080ac3248d423d4d037bc831', local_dir='${flp_checkpoints}/chinese-hubert-base')"
    touch "${flp_checkpoints}/chinese-hubert-base/.download-complete"
  fi

  # MediaPipe Face Landmarker (Apache-2.0): finds the face in the portrait.
  # The URL names a fixed version and the checksum guards the content.
  local landmarker="${flp_checkpoints}/mediapipe/face_landmarker.task"
  local landmarker_sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
  if ! echo "${landmarker_sha256}  ${landmarker}" | sha256sum --check --status 2>/dev/null; then
    echo "[models] Downloading the MediaPipe face landmarker"
    mkdir -p "$(dirname "${landmarker}")"
    curl --fail --location --retry 5 --retry-all-errors --silent --show-error \
      -o "${landmarker}.tmp" \
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    echo "${landmarker_sha256}  ${landmarker}.tmp" | sha256sum --check --status
    mv "${landmarker}.tmp" "${landmarker}"
  fi

}

# Self-signed certificate for HTTPS (microphone access from other machines).
# It covers the host's IP addresses plus localhost and is regenerated when that
# list changes, e.g. after the VM gets a new address.
ensure_https_certificate() {
  local cert_dir hosts
  cert_dir="$(python -m avatar.config get https_cert_dir)"
  hosts="$(python -m avatar.config get https_cert_hosts)"
  hosts="${hosts:-$(hostname -I 2>/dev/null) 127.0.0.1 localhost}"
  local san="" host
  for host in ${hosts}; do
    if [[ "${host}" =~ ^[0-9.]+$ || "${host}" == *:* ]]; then
      san+="IP:${host},"
    else
      san+="DNS:${host},"
    fi
  done
  san="${san%,}"
  mkdir -p "${cert_dir}"
  export HTTPS_CERT_FILE="${cert_dir}/avatar-cert.pem"
  export HTTPS_KEY_FILE="${cert_dir}/avatar-key.pem"
  if [[ ! -f "${HTTPS_CERT_FILE}" || "$(cat "${cert_dir}/san.txt" 2>/dev/null)" != "${san}" ]]; then
    echo "[https] Creating a self-signed certificate for ${san}"
    openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
      -keyout "${HTTPS_KEY_FILE}" -out "${HTTPS_CERT_FILE}" \
      -subj "/CN=neural-avatar" -addext "subjectAltName=${san}" 2>/dev/null
    echo "${san}" > "${cert_dir}/san.txt"
  fi
}

case "${mode}" in
  download)
    download_models
    ;;
  serve)
    cd "${flp_root}"
    # Also stops here, with the list of problems, on an invalid configuration.
    https_port="$(python -m avatar.config get https_port)"
    if [[ -n "${https_port}" ]]; then
      ensure_https_certificate
    fi
    exec python -m avatar.serve
    ;;
  chatterbox)
    cd /workspace
    exec uvicorn tts.chatterbox_api:app --host 0.0.0.0 --port "${CHATTERBOX_PORT:-7860}"
    ;;
  *)
    exec "$@"
    ;;
esac
