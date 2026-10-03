#!/usr/bin/env bash
# Rebuild and restart the avatar service from the working tree, then check it
# end to end: configuration against the reference, typed speech, a spoken
# question through listener and conductor, and errors in the avatar log.
#   tools/smoke.sh            # rebuild, restart, check
#   tools/smoke.sh --no-build # only the checks, on the running stack
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" != "--no-build" ]]; then
  docker compose build webrtc-avatar 2>&1 | grep -E 'Ran [0-9]+ tests|^#[0-9]+ [0-9.]+ (OK|FAILED)|ERROR|Built' || true
  docker compose up -d webrtc-avatar >/dev/null 2>&1
fi
state=""
for _ in $(seq 1 150); do
  state="$(docker inspect -f '{{.State.Status}}/{{.State.Health.Status}}' avatar_webrtc_studio 2>&1 || true)"
  case "${state}" in running/healthy|exited*|dead*|*unhealthy) break ;; esac
  sleep 2
done
echo "[smoke] avatar: ${state}"
[[ "${state}" == running/healthy ]] || { docker logs --tail 40 avatar_webrtc_studio; exit 1; }

echo "[smoke] configuration:"
tools/config_snapshot.py check | grep -v '^[-+~] compose-env' || true
echo "[smoke] typed speech:"
docker compose exec -T webrtc-avatar python - "Great, then we agree." "Okay, I'll stop." \
  < tools/multi_say.py 2>&1 | tail -1
echo "[smoke] spoken question:"
docker compose exec -T webrtc-avatar python - en-US /workspace/tools/audio/question.wav \
  < tools/mic_turn.py 2>&1 | tail -1 | cut -c1-200
echo "[smoke] errors in the avatar log:"
docker logs avatar_webrtc_studio 2>&1 | grep -E 'Traceback|ERROR avatar|Exception' | grep -v onnxruntime | head -5 || true
echo "[smoke] done"
