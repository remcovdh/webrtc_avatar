#!/usr/bin/env bash
# Run the avatar image's unit tests against the working tree, without
# rebuilding the image: the sources are copied over the baked-in ones inside a
# throwaway container. No GPU needed.
#   tools/run_tests.sh                 # the build-time test set
#   tools/run_tests.sh test_av_sync.py # selected files
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
tests=("$@")
if [[ ${#tests[@]} -eq 0 ]]; then
  tests=(test_motion_continuity.py test_neural_idle_frame.py test_frame_windows.py
    test_tts_provider.py test_progressive_scheduling.py test_visual_quality.py
    test_av_sync.py test_listener_client.py test_conductor_client.py
    test_avatar_config.py)
fi
docker run --rm -v "$PWD:/repo:ro" --entrypoint bash neural-avatar:latest -c '
  cd /workspace/FasterLivePortrait
  cp /repo/*.py /repo/docker-compose.yml /repo/quality_benchmark.sh /repo/index.html \
    /repo/.env.example .
  python -m unittest "$@"' _ "${tests[@]}"
