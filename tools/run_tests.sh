#!/usr/bin/env bash
# Run the unit tests against the working tree, without rebuilding any image:
# the repo is mounted read-only into a throwaway container of the service's
# image. No GPU needed.
#   tools/run_tests.sh                          # avatar, conductor and listener
#   tools/run_tests.sh avatar                   # one service
#   tools/run_tests.sh avatar test_av_sync.py   # one file of a service
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

run() {  # service image working-dir [test-file]
  local service="$1" image="$2" workdir="$3" pattern="${4:-test_*.py}"
  echo "== ${service} (${image})"
  docker run --rm -v "$PWD:/repo:ro" -w "${workdir}" \
    -e PYTHONPATH=/repo/src -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint python "${image}" \
    -m unittest discover -s "/repo/tests/${service}" -p "${pattern}"
}

services=("${1:-avatar}")
[[ $# -eq 0 ]] && services=(avatar scripts conductor listener)
for service in "${services[@]}"; do
  case "${service}" in
    # The avatar's tests import FasterLivePortrait's `src` package from there.
    avatar|scripts) run "${service}" neural-avatar:latest /workspace/FasterLivePortrait "${2:-}" ;;
    conductor) run conductor neural-avatar-conductor:latest /tmp "${2:-}" ;;
    listener) run listener neural-avatar-listener:latest /tmp "${2:-}" ;;
    *) echo "unknown service: ${service} (avatar, scripts, conductor, listener)" >&2; exit 2 ;;
  esac
done
