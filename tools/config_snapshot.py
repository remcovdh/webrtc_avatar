#!/usr/bin/env python3
"""Record or check the effective configuration of the running stack.

The snapshot is the environment Compose gives every service plus the settings
the avatar reports on /health (measurements such as timings are left out). It
is the safety net for configuration refactoring: a change that is meant to keep
behaviour must leave the snapshot unchanged, and a deliberate change shows up
as a small, reviewable difference.

    tools/config_snapshot.py save     # write config/reference/*.json
    tools/config_snapshot.py check    # compare the running stack with them

Run from the repo root on the host, with the stack up.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "config" / "reference"
HEALTH_URL = "http://127.0.0.1:8000/health"
# Measured values and machine facts: they differ between starts without any
# configuration change.
HEALTH_VOLATILE = {
    "startup_warmup_seconds",
    "startup_warmup_metrics",
    "tts_startup_wait_seconds",
    "render_fps_estimate",
    "gpu",
    "torch_version",
    "torch_cuda",
    "onnxruntime_version",
    "onnx_providers",
}


def compose_environment() -> dict[str, dict[str, str]]:
    raw = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    services = json.loads(raw)["services"]
    return {
        name: dict(sorted((service.get("environment") or {}).items()))
        for name, service in sorted(services.items())
        if service.get("environment")
    }


def health_settings() -> dict[str, object]:
    with urllib.request.urlopen(HEALTH_URL, timeout=10) as response:
        health = json.load(response)
    return {k: v for k, v in sorted(health.items()) if k not in HEALTH_VOLATILE}


def differences(name: str, expected: object, actual: object) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        lines: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{name}.{key}"
            if key not in actual:
                lines.append(f"- {path} = {expected[key]!r} (gone)")
            elif key not in expected:
                lines.append(f"+ {path} = {actual[key]!r} (new)")
            else:
                lines.extend(differences(path, expected[key], actual[key]))
        return lines
    if expected != actual:
        return [f"~ {name}: {expected!r} -> {actual!r}"]
    return []


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in {"save", "check"}:
        print(__doc__)
        return 2
    current = {
        "compose-env": compose_environment(),
        "health": health_settings(),
    }
    if mode == "save":
        REFERENCE.mkdir(parents=True, exist_ok=True)
        for name, value in current.items():
            path = REFERENCE / f"{name}.json"
            path.write_text(json.dumps(value, indent=2) + "\n")
            print(f"wrote {path.relative_to(ROOT)}")
        return 0
    lines: list[str] = []
    for name, value in current.items():
        expected = json.loads((REFERENCE / f"{name}.json").read_text())
        lines.extend(differences(name, expected, value))
    print("\n".join(lines) if lines else "configuration matches the reference")
    return 1 if lines else 0


if __name__ == "__main__":
    raise SystemExit(main())
