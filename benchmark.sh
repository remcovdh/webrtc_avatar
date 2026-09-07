#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_root}"

benchmark_repeats="${BENCHMARK_REPEATS:-3}"
benchmark_warmups="${BENCHMARK_WARMUPS:-1}"
benchmark_modes="${BENCHMARK_MODES:-design,preset-clone,preset-direction}"
benchmark_render_strides="${BENCHMARK_RENDER_STRIDES:-2}"
benchmark_adaptive_values="${BENCHMARK_ADAPTIVE_VALUES:-true}"
benchmark_catchup_stride="${BENCHMARK_CATCHUP_STRIDE:-3}"
benchmark_catchup_buffer_seconds="${BENCHMARK_CATCHUP_BUFFER_SECONDS:-0.75}"
benchmark_prefetch_values="${BENCHMARK_PREFETCH_VALUES:-false}"
regenerate_preset="${REGENERATE_PRESET:-0}"
build_images="${BENCHMARK_BUILD:-0}"
preset_basename="${BENCHMARK_PRESET_BASENAME:-benchmark-voice-preset}"

if [[ ! "${preset_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "[benchmark] Invalid BENCHMARK_PRESET_BASENAME: ${preset_basename}" >&2
  exit 2
fi
preset_wav="/workspace/inputs/${preset_basename}.wav"
preset_transcript="/workspace/inputs/${preset_basename}.txt"
preset_manifest="/workspace/inputs/${preset_basename}.manifest.json"

if [[ ! "${benchmark_catchup_stride}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[benchmark] Invalid BENCHMARK_CATCHUP_STRIDE: ${benchmark_catchup_stride}" >&2
  exit 2
fi
if [[ ! "${benchmark_catchup_buffer_seconds}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[benchmark] Invalid BENCHMARK_CATCHUP_BUFFER_SECONDS: ${benchmark_catchup_buffer_seconds}" >&2
  exit 2
fi

if [[ "${build_images}" == "1" ]]; then
  docker compose build webrtc-avatar benchmark-fixture
fi

mkdir -p inputs results/benchmarks
docker compose up -d tts

force_args=()
if [[ "${regenerate_preset}" == "1" ]]; then
  force_args+=(--force)
fi

docker compose --profile benchmark run --rm benchmark-fixture \
  python /workspace/FasterLivePortrait/benchmark_avatar.py \
  prepare-preset \
  --wav "${preset_wav}" \
  --transcript "${preset_transcript}" \
  --manifest "${preset_manifest}" \
  "${force_args[@]}"

for render_stride in ${benchmark_render_strides}; do
  if [[ ! "${render_stride}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[benchmark] Invalid render stride: ${render_stride}" >&2
    exit 2
  fi
  for adaptive_stride in ${benchmark_adaptive_values}; do
    if [[ "${adaptive_stride}" != "true" && "${adaptive_stride}" != "false" ]]; then
      echo "[benchmark] Invalid adaptive value: ${adaptive_stride}" >&2
      exit 2
    fi
    for tts_prefetch in ${benchmark_prefetch_values}; do
      if [[ "${tts_prefetch}" != "true" && "${tts_prefetch}" != "false" ]]; then
        echo "[benchmark] Invalid prefetch value: ${tts_prefetch}" >&2
        exit 2
      fi
      scenario="stride-${render_stride}_adaptive-${adaptive_stride}_catchup-${benchmark_catchup_stride}_prefetch-${tts_prefetch}"
      echo "[benchmark] Starting ${scenario}"
      AVATAR_RENDER_STRIDE="${render_stride}" \
      AVATAR_ADAPTIVE_RENDER_STRIDE="${adaptive_stride}" \
      AVATAR_CATCHUP_RENDER_STRIDE="${benchmark_catchup_stride}" \
      AVATAR_CATCHUP_BUFFER_SECONDS="${benchmark_catchup_buffer_seconds}" \
      AVATAR_TTS_PREFETCH="${tts_prefetch}" \
      AVATAR_PRESET_AUDIO_PATH="${preset_wav}" \
      AVATAR_PRESET_TRANSCRIPT_FILE="${preset_transcript}" \
        docker compose up -d --force-recreate webrtc-avatar

      healthy=0
      for _attempt in $(seq 1 180); do
        if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
          healthy=1
          break
        fi
        sleep 2
      done
      if [[ "${healthy}" != "1" ]]; then
        echo "[benchmark] Avatar health check did not become ready" >&2
        docker compose logs --tail=100 webrtc-avatar >&2
        exit 1
      fi

      docker compose exec -T webrtc-avatar \
        python /workspace/FasterLivePortrait/benchmark_avatar.py run \
        --output-root "/workspace/results/benchmarks/${scenario}" \
        --repeats "${benchmark_repeats}" \
        --warmups "${benchmark_warmups}" \
        --modes "${benchmark_modes}" \
        --fixture-manifest "${preset_manifest}"
    done
  done
done

docker compose exec -T webrtc-avatar \
  python /workspace/FasterLivePortrait/benchmark_avatar.py compare

echo "[benchmark] Reports and the cross-scenario comparison are under results/benchmarks/"
