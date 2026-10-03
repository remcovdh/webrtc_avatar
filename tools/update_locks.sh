#!/usr/bin/env bash
# Write docker/locks/<service>.txt from the images as they are built now.
# Use after changing a dependency on purpose (see docker/locks/README.md).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for pair in avatar:neural-avatar tts:neural-avatar-tts-chatterbox \
            listener:neural-avatar-listener conductor:neural-avatar-conductor; do
  service="${pair%%:*}"
  docker run --rm --entrypoint python "${pair##*:}:latest" -m pip list --format=freeze \
    2>/dev/null | sort -f > "docker/locks/${service}.txt"
  echo "${service}: $(wc -l < "docker/locks/${service}.txt") packages"
done
