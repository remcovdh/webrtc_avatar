#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_root}"

quality_repeats="${QUALITY_BENCHMARK_REPEATS:-1}"
quality_warmups="${QUALITY_BENCHMARK_WARMUPS:-1}"
quality_mode="${QUALITY_BENCHMARK_MODE:-preset-clone}"
quality_text="${QUALITY_BENCHMARK_TEXT:-Baby, please meet me by the bright blue moon. Open your mouth and say: amazing, wonderful, absolutely beautiful.}"
quality_scenarios="${QUALITY_BENCHMARK_SCENARIOS:-baseline,all,1.00,true,false,false;lip-100,lip,1.00,true,false,false;lip-115,lip,1.15,true,false,false;lip-130,lip,1.30,true,false,false;exp-115,exp,1.15,true,false,false}"
quality_profile="${QUALITY_BENCHMARK_PROFILE:-runtime}"
build_images="${QUALITY_BENCHMARK_BUILD:-0}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

case "${quality_profile}" in
  runtime)
    render_stride=2
    adaptive_stride=true
    incremental_windows=true
    ;;
  visual)
    render_stride=1
    adaptive_stride=false
    incremental_windows=false
    ;;
  *)
    echo "[quality] QUALITY_BENCHMARK_PROFILE must be runtime or visual" >&2
    exit 2
    ;;
esac

host_output="results/quality-benchmarks/${timestamp}-${quality_profile}"
container_output="/workspace/results/quality-benchmarks/${timestamp}-${quality_profile}"

if [[ ! "${quality_repeats}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[quality] QUALITY_BENCHMARK_REPEATS must be at least 1" >&2
  exit 2
fi
if [[ ! "${quality_warmups}" =~ ^[0-9]+$ ]]; then
  echo "[quality] QUALITY_BENCHMARK_WARMUPS must be zero or greater" >&2
  exit 2
fi
if [[ "${build_images}" == "1" ]]; then
  docker compose build webrtc-avatar
fi

mkdir -p "${host_output}"
docker compose up -d tts
echo "[quality] Profile=${quality_profile} stride=${render_stride} adaptive=${adaptive_stride} incremental=${incremental_windows}"

IFS=';' read -r -a scenarios <<< "${quality_scenarios}"
for raw_scenario in "${scenarios[@]}"; do
  IFS=',' read -r name region multiplier normalize_lip eye_retarget lip_retarget extra <<< "${raw_scenario}"
  if [[ -n "${extra:-}" || -z "${name:-}" || -z "${region:-}" || -z "${multiplier:-}" || -z "${normalize_lip:-}" || -z "${eye_retarget:-}" || -z "${lip_retarget:-}" ]]; then
    echo "[quality] Invalid scenario: ${raw_scenario}" >&2
    echo "[quality] Expected name,region,multiplier,normalize_lip,eye_retarget,lip_retarget" >&2
    exit 2
  fi
  if [[ ! "${name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "[quality] Unsafe scenario name: ${name}" >&2
    exit 2
  fi
  if [[ ! "${region}" =~ ^(all|exp|pose|lip|eyes)$ ]]; then
    echo "[quality] Invalid animation region in ${name}: ${region}" >&2
    exit 2
  fi
  for flag in "${normalize_lip}" "${eye_retarget}" "${lip_retarget}"; do
    if [[ "${flag}" != "true" && "${flag}" != "false" ]]; then
      echo "[quality] Invalid boolean in ${name}: ${flag}" >&2
      exit 2
    fi
  done

  echo "[quality] ${name}: region=${region} multiplier=${multiplier} normalize_lip=${normalize_lip} eye_retarget=${eye_retarget} lip_retarget=${lip_retarget}"
  scenario_host="${host_output}/${name}"
  scenario_container="${container_output}/${name}"
  mkdir -p "${scenario_host}"

  AVATAR_ANIMATION_REGION="${region}" \
  AVATAR_DRIVING_MULTIPLIER="${multiplier}" \
  AVATAR_NORMALIZE_LIP="${normalize_lip}" \
  AVATAR_EYE_RETARGETING="${eye_retarget}" \
  AVATAR_LIP_RETARGETING="${lip_retarget}" \
  AVATAR_RENDER_STRIDE="${render_stride}" \
  AVATAR_ADAPTIVE_RENDER_STRIDE="${adaptive_stride}" \
  AVATAR_INCREMENTAL_FRAME_WINDOWS="${incremental_windows}" \
  AVATAR_TTS_PREFETCH=false \
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
    docker compose logs --tail=200 webrtc-avatar > "${scenario_host}/startup-error.log"
    echo "[quality] Avatar did not become healthy for ${name}" >&2
    exit 1
  fi

  AVATAR_ANIMATION_REGION="${region}" \
  AVATAR_DRIVING_MULTIPLIER="${multiplier}" \
  AVATAR_NORMALIZE_LIP="${normalize_lip}" \
  AVATAR_EYE_RETARGETING="${eye_retarget}" \
  AVATAR_LIP_RETARGETING="${lip_retarget}" \
  AVATAR_RENDER_STRIDE="${render_stride}" \
  AVATAR_ADAPTIVE_RENDER_STRIDE="${adaptive_stride}" \
  AVATAR_INCREMENTAL_FRAME_WINDOWS="${incremental_windows}" \
  AVATAR_TTS_PREFETCH=false \
    docker compose config > "${scenario_host}/compose-resolved.yaml"
  curl --fail --silent http://127.0.0.1:8000/health > "${scenario_host}/health.json"

  docker compose exec -T webrtc-avatar \
    python /workspace/FasterLivePortrait/benchmark_avatar.py run \
    --output-root "${scenario_container}" \
    --repeats "${quality_repeats}" \
    --warmups "${quality_warmups}" \
    --modes "${quality_mode}" \
    --text "${quality_text}" \
    --record-media

  docker compose logs --no-color webrtc-avatar > "${scenario_host}/container.log"
done

docker compose exec -T webrtc-avatar \
  python /workspace/FasterLivePortrait/benchmark_avatar.py compare \
  --input-root "${container_output}" \
  --output-dir "${container_output}"

echo "[quality] Videos, exact settings, raw metrics and comparison: ${host_output}/"
