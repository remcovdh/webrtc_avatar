#!/usr/bin/env bash
# Review the conversations since the previous review (M5; see review.py).
# The review model (Qwen3 4B) does not fit on the GPU next to the running
# avatar, so this pauses the avatar and TTS, runs the review in the conductor
# image, and starts them again. Talking to the avatar is not possible meanwhile.
#
#   ./review.sh            review new conversations
#   ./review.sh --all      review every logged conversation
#   ./review.sh --no-model facts only (no GPU needed, avatar keeps running)
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_root}"

paused=()
if [[ " $* " != *" --no-model "* ]]; then
  for service in webrtc-avatar tts; do
    if [[ -n "$(docker compose ps --status running -q "${service}")" ]]; then
      paused+=("${service}")
    fi
  done
  if ((${#paused[@]})); then
    echo "[review] Pausing ${paused[*]} to free GPU memory for the review model"
    docker compose stop "${paused[@]}"
  fi
fi

restart() {
  if ((${#paused[@]})); then
    echo "[review] Starting ${paused[*]} again"
    docker compose start "${paused[@]}" >/dev/null
  fi
}
trap restart EXIT

report="$(docker compose run --rm --no-deps -T conductor python /workspace/review.py "$@" | tail -1)"
echo "[review] Report: results/review/$(basename "${report}")"
