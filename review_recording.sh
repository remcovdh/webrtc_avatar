#!/usr/bin/env bash
# Record one phrase from the running avatar over WebRTC and turn it into
# images a reviewer (human or Claude) can inspect without playing the video:
#   sheet_NN.png     timestamped frame grids (REVIEW_FPS frames per second)
#   closeup_NN.png   timestamped mouth-and-eyes close-ups, 4 per second
#   mouth_audio.png  a mouth-region strip over the audio waveform, for lip sync
# Usage: ./review_recording.sh ["text to speak"]   (stack must already be up)
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_root}"

text="${1:-Baby, please meet me by the bright blue moon. Open your mouth and say: amazing, wonderful, absolutely beautiful.}"
mode="${REVIEW_MODE:-preset-clone}"
fps="${REVIEW_FPS:-4}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
# The container writes the MP4 as root under results/claude-review/; the
# review images go to a user-owned sibling folder.
host_output="results/claude-review-images/${timestamp}"

mkdir -p "${host_output}"
curl --fail --silent http://127.0.0.1:8000/health >/dev/null \
  || { echo "[review] avatar is not healthy on :8000; run docker compose up -d first" >&2; exit 1; }

docker compose exec -T webrtc-avatar \
  python /workspace/FasterLivePortrait/benchmark_avatar.py run \
  --output-root "/workspace/results/claude-review/${timestamp}" \
  --repeats 1 --warmups 0 --modes "${mode}" --text "${text}" --record-media >/dev/null

video="$(find "results/claude-review/${timestamp}" -name '*.mp4' | head -1)"
[[ -n "${video}" ]] || { echo "[review] no recording produced" >&2; exit 1; }
cp "${video}" "${host_output}/recording.mp4"

ffmpeg -v error -y -i "${video}" \
  -vf "fps=${fps},scale=320:-1,drawtext=text='%{pts\:hms}':x=5:y=5:fontcolor=yellow:fontsize=18,tile=6x4" \
  "${host_output}/sheet_%02d.png"

# Mouth-and-eyes close-ups over the whole recording at 4 fps: the crops that
# showed the pressed-lips problem the full-face sheets hid.
ffmpeg -v error -y -i "${video}" \
  -vf "fps=4,crop=iw*0.5:ih*0.34:iw*0.25:ih*0.45,scale=200:-1,drawtext=text='%{pts\:hms}':x=4:y=4:fontcolor=yellow:fontsize=12,tile=10x7" \
  "${host_output}/closeup_%02d.png"

# Mouth crops sampled at 10 fps above the waveform of the same span, folded
# into 3-second rows: open mouths should line up with loud syllables.
duration="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${video}")"
rows="$(awk -v d="${duration}" 'BEGIN { printf "%d", int(d / 3) + 1 }')"
ffmpeg -v error -y -i "${video}" -filter_complex "\
[0:v]fps=10,crop=iw*0.36:ih*0.2:iw*0.32:ih*0.62,scale=64:-2,tile=$((rows * 30))x1[m];\
[0:a]apad=whole_dur=$((rows * 3)),showwavespic=s=$((rows * 1920))x120:colors=yellow,format=rgba,colorkey=black:0.01,split[wv][wa];[wa]drawbox=c=black:t=fill[bg];[bg][wv]overlay,format=rgb24[w];\
[m][w]vstack,untile=${rows}x1,tile=1x${rows}:padding=6" -frames:v 1 "${host_output}/mouth_audio.png"

echo "${host_output}"
