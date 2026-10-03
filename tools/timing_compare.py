#!/usr/bin/env python3
"""Compare timing runs of the fixed three-phrase benchmark text.

Make a run (stack up; three measured repeats after one warm-up):

    docker compose exec -T webrtc-avatar python /workspace/app/scripts/benchmark_avatar.py \\
        run --output-root /workspace/results/benchmarks/<name> --repeats 3 --warmups 1 \\
        --modes preset-clone

Then compare the latest run under each name, on the host:

    tools/timing_compare.py <name> [<name> ...]

Per phrase it shows the medians over the repeats of what makes speech uneven:
audio gaps (underrun), video that was late (video underrun) or skipped
(dropped), speech held for the video, and frames shown twice to catch up.
"""

from __future__ import annotations

import csv
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results" / "benchmarks"
SUMMARY = ["median_first_ready_ms", "median_neural_fps", "median_gap_ms", "median_client_wall_ms"]
PHRASE = ["render_stride", "render_effective_fps", "underrun_ms", "video_underrun_ms",
          "video_dropped_ms", "speech_hold_ms", "render_held_frames", "render_frames"]


def latest(name: str) -> Path:
    runs = sorted(glob.glob(str(ROOT / name / "*" / "benchmark.json")))
    if not runs:
        raise SystemExit(f"no run under results/benchmarks/{name}")
    return Path(runs[-1]).parent


def main() -> int:
    names = sys.argv[1:]
    if not names:
        print(__doc__)
        return 2
    for name in names:
        folder = latest(name)
        summary = json.loads((folder / "benchmark.json").read_text())["summary"]["by_mode"]
        mode = next(iter(summary.values()))
        print(f"== {name} ({folder.name})")
        print("  " + "  ".join(f"{key.removeprefix('median_')}={mode.get(key)}" for key in SUMMARY))
        rows = [row for row in csv.DictReader((folder / "phrases.csv").open())
                if row.get("warmup") == "False"]
        for chunk in sorted({row["chunk"] for row in rows}, key=int):
            values = []
            for key in PHRASE:
                numbers = [float(row[key]) for row in rows
                           if row["chunk"] == chunk and row.get(key) not in (None, "")]
                values.append(f"{key.removeprefix('render_')}="
                              f"{statistics.median(numbers):g}" if numbers else f"{key}=-")
            print(f"  phrase {chunk}: " + "  ".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
