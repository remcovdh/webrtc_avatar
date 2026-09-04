#!/usr/bin/env python3
"""Repeatable Breeze-only streaming latency benchmark.

Runs inside the Breeze container so its Python dependencies and GPU namespace
match the service under test. Results are appended to one JSON file per
scenario; the shell orchestrator builds CSV and Markdown comparisons.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx

PCM_BYTES_PER_SECOND = 24_000 * 2  # mono, signed 16-bit PCM


def flash_attention_version() -> str | None:
    try:
        import flash_attn
        return getattr(flash_attn, "__version__", "installed")
    except ImportError:
        return None


def gpu_used_mib() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return sum(int(line.strip()) for line in output.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


async def sample_gpu(stop: asyncio.Event, samples: list[int]) -> None:
    while not stop.is_set():
        value = await asyncio.to_thread(gpu_used_mib)
        if value is not None:
            samples.append(value)
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.05)
        except TimeoutError:
            pass


async def synthesize(client: httpx.AsyncClient, url: str, text: str, instruction: str,
                     cfg_scale: float, seed: int) -> dict[str, float | int | None]:
    started = time.perf_counter()
    first_byte_ms = None
    size = 0
    samples: list[int] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_gpu(stop, samples))
    try:
        async with client.stream(
            "POST",
            f"{url.rstrip('/')}/v1/audio/speech",
            files={
                "text": (None, text),
                "instruction": (None, instruction),
                "cfg_scale": (None, str(cfg_scale)),
                "seed": (None, str(seed)),
            },
            timeout=None,
        ) as response:
            response.raise_for_status()
            headers_ms = (time.perf_counter() - started) * 1000
            async for chunk in response.aiter_bytes():
                if chunk and first_byte_ms is None:
                    first_byte_ms = (time.perf_counter() - started) * 1000
                size += len(chunk)
    finally:
        stop.set()
        await sampler
    total_ms = (time.perf_counter() - started) * 1000
    media_seconds = size / PCM_BYTES_PER_SECOND
    return {
        "headers_ms": round(headers_ms, 3),
        "first_byte_ms": round(first_byte_ms or total_ms, 3),
        "total_ms": round(total_ms, 3),
        "bytes": size,
        "media_seconds": round(media_seconds, 6),
        "rtf": round(total_ms / 1000 / media_seconds, 6) if media_seconds else None,
        "gpu_peak_mib": max(samples) if samples else None,
        "gpu_min_mib": min(samples) if samples else None,
    }


async def main_async(args: argparse.Namespace) -> int:
    async with httpx.AsyncClient() as client:
        for _ in range(args.warmups):
            await synthesize(client, args.url, args.text, args.instruction,
                             args.cfg_scale, args.seed)
        runs = []
        for index in range(1, args.repeats + 1):
            result = await synthesize(client, args.url, args.text, args.instruction,
                                      args.cfg_scale, args.seed)
            result["run"] = index
            runs.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    def median(key: str) -> float | int | None:
        values = [row[key] for row in runs if row.get(key) is not None]
        return round(statistics.median(values), 3) if values else None

    payload = {
        "schema_version": 1,
        "scenario": args.scenario,
        "fast_args": args.fast_args,
        "text": args.text,
        "instruction": args.instruction,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "flash_attention": flash_attention_version(),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "runs": runs,
        "median": {key: median(key) for key in (
            "headers_ms", "first_byte_ms", "total_ms", "media_seconds", "rtf",
            "gpu_peak_mib", "gpu_min_mib"
        )},
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--fast-args", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text", default=(
        "Hello! This is a repeatable Breeze streaming benchmark. "
        "It measures the time to first audio and the total generation speed."
    ))
    parser.add_argument("--instruction", default=(
        "A warm, clear, natural voice with a calm conversational delivery."
    ))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
