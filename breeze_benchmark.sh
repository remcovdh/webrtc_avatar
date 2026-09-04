#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_root}"

warmups="${BREEZE_BENCHMARK_WARMUPS:-2}"
repeats="${BREEZE_BENCHMARK_REPEATS:-3}"
profiles="${BREEZE_BENCHMARK_PROFILES:-eager backbone-decode depth-decoder codec backbone-prefill text-encoder}"
include_fast_all="${BREEZE_BENCHMARK_INCLUDE_FAST_ALL:-0}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="results/breeze-benchmarks/${timestamp}"
mkdir -p "${output_root}/audio"

declare -A flags=(
  [eager]=""
  [backbone-decode]="--fast-backbone-decode"
  [depth-decoder]="--fast-depth-decoder"
  [codec]="--fast-codec"
  [backbone-prefill]="--fast-backbone-prefill"
  [text-encoder]="--fast-text-encoder"
  [depth-codec]="--fast-depth-decoder --fast-codec"
  [depth-codec-backbone]="--fast-depth-decoder --fast-codec --fast-backbone-decode"
  # Compatibility alias retained from v2G.
  [decode-depth-codec]="--fast-backbone-decode --fast-depth-decoder --fast-codec"
  [fast-all]="--fast-all"
)

if [[ "${include_fast_all}" == "1" ]]; then
  profiles="${profiles} fast-all"
fi

if [[ "${BREEZE_BENCHMARK_BUILD:-0}" == "1" ]]; then
  docker compose build breeze-tts
fi

# Isolate Breeze and free the portrait model's VRAM. This intentionally leaves
# the avatar stopped after the benchmark; `docker compose up -d` restores it.
docker compose stop webrtc-avatar >/dev/null 2>&1 || true
docker compose up -d model-init

for profile in ${profiles}; do
  if [[ ! -v "flags[${profile}]" ]]; then
    echo "[breeze-benchmark] Unknown profile: ${profile}" >&2
    exit 2
  fi
  echo "[breeze-benchmark] ${profile}: ${flags[${profile}]:-eager baseline}"
  export BREEZE_FAST_ARGS="${flags[${profile}]}"
  docker compose up -d --force-recreate breeze-tts

  healthy=0
  for _attempt in $(seq 1 240); do
    if curl --fail --silent http://127.0.0.1:7860/docs >/dev/null; then
      healthy=1
      break
    fi
    sleep 2
  done
  if [[ "${healthy}" != "1" ]]; then
    echo "${profile}" >> "${output_root}/failed.txt"
    docker compose logs --tail=120 breeze-tts > "${output_root}/${profile}.log" 2>&1 || true
    echo "[breeze-benchmark] ${profile} failed to become healthy; continuing" >&2
    continue
  fi

  if ! docker compose exec -T breeze-tts python /workspace/breeze_benchmark.py \
      --scenario "${profile}" \
      --fast-args="${flags[${profile}]}" \
      --warmups "${warmups}" \
      --repeats "${repeats}" \
      --capture-wav "/workspace/results/${profile}.wav" \
      --output "/workspace/results/${profile}.json"; then
    echo "${profile}" >> "${output_root}/failed.txt"
    docker compose logs --tail=120 breeze-tts > "${output_root}/${profile}.log" 2>&1 || true
    echo "[breeze-benchmark] ${profile} failed during synthesis; continuing" >&2
    continue
  fi
  docker compose cp "breeze-tts:/workspace/results/${profile}.json" "${output_root}/${profile}.json"
  docker compose cp "breeze-tts:/workspace/results/${profile}.wav" "${output_root}/audio/${profile}.wav"
done

python - "${output_root}" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows=[]
for path in sorted(root.glob("*.json")):
    data=json.loads(path.read_text())
    row={"scenario":data["scenario"], "fast_args":data["fast_args"], "flash_attention":data.get("flash_attention"), "capture_wav":data.get("capture_wav"), "capture_sha256":data.get("capture_sha256"), **data["median"]}
    rows.append(row)
fields=["scenario","fast_args","flash_attention","capture_wav","capture_sha256","first_byte_ms","total_ms","media_seconds","rtf","gpu_peak_mib","gpu_min_mib","headers_ms"]
with (root/"comparison.csv").open("w", newline="") as f:
    writer=csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
lines=["# Breeze fast-path benchmark", "", "Medians after warm-up.", "",
       "| Profile | Fast arguments | Audio capture | First byte | Total | Audio | RTF | Peak GPU |",
       "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
for r in rows:
    lines.append(f"| {r['scenario']} | `{r['fast_args'] or 'eager'}` | [listen](audio/{r['scenario']}.wav) | {r['first_byte_ms']:.0f} ms | {r['total_ms']:.0f} ms | {r['media_seconds']:.2f} s | {r['rtf']:.3f} | {r['gpu_peak_mib'] or 'n/a'} MiB |")
failed=root/"failed.txt"
if failed.exists():
    lines += ["", "## Failed profiles", "", *[f"- {x}" for x in failed.read_text().splitlines()]]
(root/"report.md").write_text("\n".join(lines)+"\n")
print(root/"report.md")
PY

unset BREEZE_FAST_ARGS
echo "[breeze-benchmark] Results: ${output_root}/report.md"
