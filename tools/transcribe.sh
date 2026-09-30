#!/usr/bin/env bash
# Transcribe WAV files with the listener's ASR model (English prompt), to check
# what the avatar really said. Paths are relative to the repo root.
#   tools/transcribe.sh results/listener-test/first-word/seg1.wav ...
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
args=()
for file in "$@"; do args+=("/repo/${file}"); done
docker run --rm --gpus all -e HF_HUB_OFFLINE=1 \
  -v "$PWD/models/hf-listener:/root/.cache/huggingface" -v "$PWD:/repo:ro" \
  neural-avatar-listener:latest python /repo/tools/transcribe.py "${args[@]}" 2>&1 | grep 'speech starts'
