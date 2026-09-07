#!/usr/bin/env python3
"""Generate a canonical Breeze preset and benchmark the real WebRTC avatar path."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription


DEFAULT_PRESET_TEXT = (
    "Hello, I am your friendly virtual assistant, ready to help with a clear "
    "and natural conversational voice."
)
DEFAULT_BENCHMARK_TEXT = (
    "Hello! This is a neural streaming avatar test. "
    "I run repeatable tests because comparable results help this avatar improve. "
    "Let's make this a clear and enjoyable journey."
)
DEFAULT_DESIGN_INSTRUCTION = (
    "A warm, clear, natural voice with a natural, slightly brisk conversational "
    "pace and no long pauses."
)
DEFAULT_DIRECTION_INSTRUCTION = (
    "Speak clearly at a natural, slightly brisk conversational pace without "
    "long pauses."
)
DEFAULT_MODES = "design,preset-clone,preset-direction"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gpu_snapshot() -> dict[str, Any] | None:
    fields = [
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "utilization.gpu",
        "temperature.gpu",
        "pstate",
        "power.draw",
        "clocks.current.sm",
    ]
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
        return dict(zip(fields, values, strict=False))
    except (FileNotFoundError, IndexError, subprocess.SubprocessError):
        return None


async def prepare_preset(args: argparse.Namespace) -> int:
    wav_path = args.wav.resolve()
    transcript_path = args.transcript.resolve()
    manifest_path = args.manifest.resolve()
    existing = [wav_path.exists(), transcript_path.exists()]
    if all(existing) and not args.force:
        wav_hash = _sha256(wav_path)
        transcript_hash = _sha256(transcript_path)
        manifest_matches = False
        if manifest_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_matches = (
                    previous.get("wav_sha256") == wav_hash
                    and previous.get("transcript_sha256") == transcript_hash
                )
            except (OSError, json.JSONDecodeError):
                manifest_matches = False
        if not manifest_matches:
            with wave.open(str(wav_path), "rb") as source:
                sample_rate = source.getframerate()
                duration = source.getnframes() / max(sample_rate, 1)
            _write_json(
                manifest_path,
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "generator": "existing preset (not regenerated)",
                    "text": transcript_path.read_text(encoding="utf-8").strip(),
                    "sample_rate": sample_rate,
                    "duration_seconds": round(duration, 3),
                    "wav_path": str(wav_path),
                    "wav_sha256": wav_hash,
                    "transcript_path": str(transcript_path),
                    "transcript_sha256": transcript_hash,
                },
            )
        print(f"Preset already exists: {wav_path}")
        print("Use --force only when you intentionally want to replace it.")
        return 0
    if any(existing) and not args.force:
        raise RuntimeError(
            "Only one preset file exists. Refusing to overwrite it; complete the "
            "pair manually or rerun with --force."
        )

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pcm_tmp = wav_path.with_suffix(".pcm.tmp")
    wav_tmp = wav_path.with_suffix(".wav.tmp")
    transcript_tmp = transcript_path.with_suffix(".txt.tmp")
    started = time.perf_counter()
    first_byte_at: float | None = None
    received = 0
    form = {
        "text": (None, args.text),
        "instruction": (None, args.instruction),
        "cfg_scale": (None, str(args.cfg_scale)),
        "seed": (None, str(args.seed)),
    }
    timeout = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=15.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{args.tts_url.rstrip('/')}/v1/audio/speech",
                files=form,
            ) as response:
                if response.is_error:
                    detail = (await response.aread()).decode(
                        "utf-8", errors="replace"
                    )[:500]
                    raise RuntimeError(
                        f"Breeze preset generation failed ({response.status_code}): "
                        f"{detail}"
                    )
                sample_rate = int(response.headers.get("X-Sample-Rate", "24000"))
                with pcm_tmp.open("wb") as output:
                    async for chunk in response.aiter_bytes():
                        if first_byte_at is None:
                            first_byte_at = time.perf_counter()
                        output.write(chunk)
                        received += len(chunk)
        raw = pcm_tmp.read_bytes()
        if len(raw) < 2:
            raise RuntimeError("Breeze returned no preset audio")
        if len(raw) % 2:
            raw = raw[:-1]
        with wave.open(str(wav_tmp), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(raw)
        transcript_tmp.write_text(args.text.strip() + "\n", encoding="utf-8")
        wav_tmp.replace(wav_path)
        transcript_tmp.replace(transcript_path)
    finally:
        pcm_tmp.unlink(missing_ok=True)
        wav_tmp.unlink(missing_ok=True)
        transcript_tmp.unlink(missing_ok=True)

    finished = time.perf_counter()
    first_byte_at = first_byte_at or finished
    frames = received // 2
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "Breeze TTS 2 streaming API",
        "tts_url": args.tts_url,
        "text": args.text,
        "instruction": args.instruction,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "sample_rate": sample_rate,
        "duration_seconds": round(frames / sample_rate, 3),
        "first_byte_ms": round((first_byte_at - started) * 1000),
        "total_ms": round((finished - started) * 1000),
        "wav_path": str(wav_path),
        "wav_sha256": _sha256(wav_path),
        "transcript_path": str(transcript_path),
        "transcript_sha256": _sha256(transcript_path),
    }
    _write_json(manifest_path, manifest)
    print(f"Generated preset: {wav_path}")
    print(f"Exact transcript: {transcript_path}")
    print(
        f"Duration {manifest['duration_seconds']:.3f}s; "
        f"first byte {manifest['first_byte_ms']}ms; total {manifest['total_ms']}ms"
    )
    return 0


async def _wait_for_ice(peer: RTCPeerConnection) -> None:
    if peer.iceGatheringState == "complete":
        return
    complete = asyncio.Event()

    @peer.on("icegatheringstatechange")
    def state_changed() -> None:
        if peer.iceGatheringState == "complete":
            complete.set()

    if peer.iceGatheringState == "complete":
        complete.set()
    await asyncio.wait_for(complete.wait(), timeout=30)


async def _drain_track(track: Any) -> None:
    try:
        while True:
            await track.recv()
    except Exception:
        return


class AvatarBenchmarkClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        ready_event_grace_seconds: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.ready_event_grace_seconds = ready_event_grace_seconds
        self.peer = RTCPeerConnection()
        self.channel = self.peer.createDataChannel("avatar-benchmark")
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.opened = asyncio.Event()
        self.track_tasks: set[asyncio.Task[Any]] = set()

        @self.channel.on("open")
        def opened() -> None:
            self.opened.set()

        @self.channel.on("message")
        def message(payload: Any) -> None:
            try:
                event = json.loads(payload) if isinstance(payload, str) else {}
            except json.JSONDecodeError:
                event = {"type": "invalid", "message": str(payload)}
            self.events.put_nowait(event)

        @self.peer.on("track")
        def track_received(track: Any) -> None:
            task = asyncio.create_task(_drain_track(track))
            self.track_tasks.add(task)
            task.add_done_callback(self.track_tasks.discard)

    async def connect(self, ice_servers: list[dict[str, Any]]) -> None:
        # The local benchmark normally uses no ICE servers. The argument is kept
        # in the report, while the same host candidates as the browser are used.
        _ = ice_servers
        self.peer.addTransceiver("video", direction="recvonly")
        self.peer.addTransceiver("audio", direction="recvonly")
        offer = await self.peer.createOffer()
        await self.peer.setLocalDescription(offer)
        await _wait_for_ice(self.peer)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/offer",
                json={
                    "sdp": self.peer.localDescription.sdp,
                    "type": self.peer.localDescription.type,
                },
            )
        response.raise_for_status()
        answer = response.json()
        await self.peer.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        await asyncio.wait_for(self.opened.wait(), timeout=30)
        # The open local data channel is sufficient to send the request. New
        # servers also emit `ready`, but tolerate old aiortc/server combinations
        # that missed the remote channel's open event instead of waiting 30s.
        try:
            while True:
                event = await asyncio.wait_for(
                    self.events.get(), timeout=self.ready_event_grace_seconds
                )
                if event.get("type") == "ready":
                    break
                if event.get("type") == "error":
                    raise RuntimeError(
                        event.get("message", "Avatar connection failed")
                    )
        except TimeoutError:
            print(
                "[benchmark] Data channel is open; continuing without the "
                "optional server ready event.",
                flush=True,
            )

    async def request(
        self,
        *,
        text: str,
        mode: str,
        instruction: str,
    ) -> dict[str, Any]:
        while not self.events.empty():
            self.events.get_nowait()
        started = time.perf_counter()
        self.channel.send(
            json.dumps(
                {
                    "text": text,
                    "voice_mode": mode,
                    "instruction": instruction,
                }
            )
        )
        collected: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        plan: dict[str, Any] | None = None
        summary: dict[str, Any] | None = None
        while True:
            event = await asyncio.wait_for(
                self.events.get(), timeout=self.timeout_seconds
            )
            collected.append(event)
            event_type = event.get("type")
            if event_type == "plan":
                plan = event
            elif event_type == "metrics":
                metrics.append(event)
            elif event_type == "summary":
                summary = event
            elif event_type == "error":
                raise RuntimeError(
                    f"{mode} benchmark failed: {event.get('message', 'unknown error')}"
                )
            elif event_type == "ready" and summary is not None:
                break
        return {
            "mode": mode,
            "instruction": instruction,
            "text": text,
            "client_wall_ms": round((time.perf_counter() - started) * 1000),
            "plan": plan,
            "metrics": metrics,
            "summary": summary,
            "events": collected,
        }

    async def close(self) -> None:
        await self.peer.close()
        for task in tuple(self.track_tasks):
            task.cancel()
        if self.track_tasks:
            await asyncio.gather(*self.track_tasks, return_exceptions=True)


def _aggregate_run(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    tts_ms = sum(float(item.get("tts_ms", 0)) for item in metrics)
    render_ms = sum(float(item.get("render_ms", 0)) for item in metrics)
    gap_ms = sum(float(item.get("underrun_ms", 0)) for item in metrics)
    video_hold_ms = sum(
        float(item.get("video_underrun_ms", 0)) for item in metrics
    )
    media_ms = sum(float(item.get("media_seconds", 0)) * 1000 for item in metrics)
    frame_loop_ms = sum(
        float(item.get("render_frame_loop_ms", 0)) for item in metrics
    )
    frames = sum(int(item.get("render_frames", 0)) for item in metrics)
    first_ready_values = [
        float(item["first_ready_ms"])
        for item in metrics
        if item.get("first_ready_ms") is not None
    ]
    first_window_values = [
        float(item["render_first_window_ms"])
        for item in metrics
        if item.get("render_first_window_ms") is not None
    ]
    first_byte_values = [
        float(item.get("tts_first_byte_ms", 0)) for item in metrics
    ]
    stride_counts: dict[str, int] = {}
    stride_reason_counts: dict[str, int] = {}
    for item in metrics:
        stride = str(item.get("render_stride", "unknown"))
        reason = str(item.get("render_stride_reason", "fixed"))
        stride_counts[stride] = stride_counts.get(stride, 0) + 1
        stride_reason_counts[reason] = stride_reason_counts.get(reason, 0) + 1
    return {
        "mode": result["mode"],
        "run_index": result["run_index"],
        "warmup": result["warmup"],
        "phrases": len(metrics),
        "client_wall_ms": result["client_wall_ms"],
        "tts_ms": round(tts_ms),
        "render_ms": round(render_ms),
        "gap_ms": round(gap_ms),
        "video_hold_ms": round(video_hold_ms),
        "media_ms": round(media_ms),
        "first_ready_ms": round(first_ready_values[0]) if first_ready_values else None,
        "first_window_median_ms": (
            round(statistics.median(first_window_values))
            if first_window_values
            else None
        ),
        "first_byte_median_ms": (
            round(statistics.median(first_byte_values)) if first_byte_values else None
        ),
        "tts_rtf": round(tts_ms / media_ms, 4) if media_ms else None,
        "render_rtf": round(render_ms / media_ms, 4) if media_ms else None,
        "neural_fps": (
            round(frames / (frame_loop_ms / 1000), 3) if frame_loop_ms else None
        ),
        "render_stride_counts": stride_counts,
        "render_stride_reason_counts": stride_reason_counts,
    }


def _median(values: list[float | int | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return round(statistics.median(available), 4) if available else None


def _summarize(runs: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    summaries = [_aggregate_run(run) for run in runs]
    measured = [item for item in summaries if not item["warmup"]]
    by_mode: dict[str, Any] = {}
    keys = [
        "client_wall_ms",
        "tts_ms",
        "render_ms",
        "gap_ms",
        "video_hold_ms",
        "media_ms",
        "first_ready_ms",
        "first_window_median_ms",
        "first_byte_median_ms",
        "tts_rtf",
        "render_rtf",
        "neural_fps",
    ]
    for mode in modes:
        selected = [item for item in measured if item["mode"] == mode]
        by_mode[mode] = {
            "measured_runs": len(selected),
            **{f"median_{key}": _median([item[key] for item in selected]) for key in keys},
        }
    design = by_mode.get("design", {})
    comparisons: dict[str, Any] = {}
    for mode in modes:
        if mode == "design":
            continue
        current = by_mode[mode]
        comparison: dict[str, Any] = {}
        for key in ["tts_rtf", "first_ready_ms", "gap_ms", "media_ms", "neural_fps"]:
            baseline = design.get(f"median_{key}")
            value = current.get(f"median_{key}")
            comparison[f"{key}_delta_percent"] = (
                round((value - baseline) / baseline * 100, 2)
                if baseline not in (None, 0) and value is not None
                else None
            )
        comparisons[mode] = comparison
    return {
        "runs": summaries,
        "by_mode": by_mode,
        "versus_design": comparisons,
    }


def _write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for metric in run["metrics"]:
            row = {
                "run_index": run["run_index"],
                "warmup": run["warmup"],
                "requested_mode": run["mode"],
                "client_wall_ms": run["client_wall_ms"],
                **{key: value for key, value in metric.items() if key != "type"},
            }
            rows.append(row)
    preferred = [
        "run_index",
        "warmup",
        "requested_mode",
        "chunk",
        "chunks",
        "phrase",
        "voice_mode",
        "voice_cfg_scale",
        "tts_ms",
        "tts_first_byte_ms",
        "render_ms",
        "render_stride",
        "render_adaptive_stride",
        "render_stride_reason",
        "render_buffer_before_seconds",
        "render_effective_fps",
        "render_first_window_ms",
        "render_window_count",
        "render_window_size",
        "media_seconds",
        "underrun_ms",
        "video_underrun_ms",
        "first_ready_ms",
        "client_wall_ms",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=preferred + remaining)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Neural avatar repeatable benchmark",
        "",
        f"Created: `{payload['created_utc']}`",
        "",
        f"Server build: `{payload['health'].get('server_build', 'unknown')}`",
        "",
        "## Fixed inputs",
        "",
        f"Benchmark text: {payload['benchmark']['text']}",
        "",
        f"Measured repeats per mode: `{payload['benchmark']['repeats']}`",
        "",
        "Render policy: base stride `{base}`, adaptive `{adaptive}`, catch-up "
        "stride `{catchup}` below `{buffer:.2f}` buffered seconds".format(
            base=payload["health"].get("render_stride", "unknown"),
            adaptive=payload["health"].get("adaptive_render_stride", False),
            catchup=payload["health"].get("catchup_render_stride", "unknown"),
            buffer=float(payload["health"].get("catchup_buffer_seconds", 0)),
        ),
        "",
        "Motion policy: relative `{relative}`, persistent across phrases "
        "`{persistent}`".format(
            relative=payload["health"].get("relative_motion", False),
            persistent=payload["health"].get("persistent_phrase_motion", False),
        ),
        "",
        "## Median results",
        "",
        "| Mode | First ready | FLP first window | First byte | TTS RTF | Render RTF | Neural FPS | Media | Audio gap | Video hold | Client wall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, stats in payload["summary"]["by_mode"].items():
        lines.append(
            "| {mode} | {first:.0f} ms | {window:.0f} ms | {byte:.0f} ms | {tts:.3f} | "
            "{render:.3f} | {fps:.2f} | {media:.0f} ms | {gap:.0f} ms | {hold:.0f} ms | "
            "{wall:.0f} ms |".format(
                mode=mode,
                first=stats.get("median_first_ready_ms") or 0,
                window=stats.get("median_first_window_median_ms") or 0,
                byte=stats.get("median_first_byte_median_ms") or 0,
                tts=stats.get("median_tts_rtf") or 0,
                render=stats.get("median_render_rtf") or 0,
                fps=stats.get("median_neural_fps") or 0,
                media=stats.get("median_media_ms") or 0,
                gap=stats.get("median_gap_ms") or 0,
                hold=stats.get("median_video_hold_ms") or 0,
                wall=stats.get("median_client_wall_ms") or 0,
            )
        )
    lines.extend(
        [
            "",
            "## Per-run totals",
            "",
            "| Run | Mode | Warm-up | Strides | TTS | Render | Media | Gap | TTS RTF | Neural FPS |",
            "| ---: | --- | :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in payload["summary"]["runs"]:
        lines.append(
            f"| {run['run_index']} | {run['mode']} | "
            f"{'yes' if run['warmup'] else 'no'} | "
            f"{', '.join(f'{stride}×{count}' for stride, count in run['render_stride_counts'].items())} | "
            f"{run['tts_ms']} ms | "
            f"{run['render_ms']} ms | {run['media_ms']} ms | {run['gap_ms']} ms | "
            f"{(run['tts_rtf'] or 0):.3f} | {(run['neural_fps'] or 0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Compare TTS RTF rather than TTS milliseconds when modes produce different speech durations.",
            "- Neural FPS should remain similar because TTS prefetch is disabled.",
            "- This harness measures performance and stability; voice identity and naturalness still require listening.",
            "- Keep JSON and CSV files when changing code or configuration so later versions remain comparable.",
            "",
        ]
    )
    return "\n".join(lines)


def compare_results(args: argparse.Namespace) -> int:
    input_root = args.input_root.resolve()
    report_paths = sorted(input_root.glob("*/20*/benchmark.json"))
    rows: list[dict[str, Any]] = []
    for report_path in report_paths:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable report {report_path}: {exc}", file=sys.stderr)
            continue
        health = payload.get("health", {})
        benchmark = payload.get("benchmark", {})
        fixture = payload.get("preset_fixture") or {}
        for mode, stats in payload.get("summary", {}).get("by_mode", {}).items():
            rows.append(
                {
                    "created_utc": payload.get("created_utc"),
                    "scenario": report_path.parent.parent.name,
                    "run_directory": report_path.parent.name,
                    "server_build": health.get("server_build"),
                    "render_stride": health.get("render_stride"),
                    "adaptive_render_stride": health.get("adaptive_render_stride"),
                    "catchup_render_stride": health.get("catchup_render_stride"),
                    "catchup_buffer_seconds": health.get("catchup_buffer_seconds"),
                    "tts_prefetch": health.get("tts_prefetch"),
                    "phrase_first_target_chars": health.get(
                        "phrase_first_target_chars"
                    ),
                    "phrase_target_chars": health.get("phrase_target_chars"),
                    "phrase_max_chars": health.get("phrase_max_chars"),
                    "mode": mode,
                    "measured_runs": stats.get("measured_runs"),
                    "median_first_ready_ms": stats.get("median_first_ready_ms"),
                    "median_first_window_ms": stats.get(
                        "median_first_window_median_ms"
                    ),
                    "median_first_byte_ms": stats.get(
                        "median_first_byte_median_ms"
                    ),
                    "median_tts_rtf": stats.get("median_tts_rtf"),
                    "median_render_rtf": stats.get("median_render_rtf"),
                    "median_neural_fps": stats.get("median_neural_fps"),
                    "median_media_ms": stats.get("median_media_ms"),
                    "median_gap_ms": stats.get("median_gap_ms"),
                    "median_video_hold_ms": stats.get("median_video_hold_ms"),
                    "median_client_wall_ms": stats.get("median_client_wall_ms"),
                    "benchmark_text_sha256": hashlib.sha256(
                        str(benchmark.get("text", "")).encode("utf-8")
                    ).hexdigest(),
                    "preset_wav_sha256": fixture.get("wav_sha256"),
                }
            )
    if not rows:
        raise RuntimeError(f"No benchmark reports found below {input_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    csv_path = args.output_dir / f"comparison-{stamp}.csv"
    md_path = args.output_dir / f"comparison-{stamp}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Neural avatar benchmark comparison",
        "",
        f"Created: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "All discovered runs are listed. Compare rows with the same benchmark-text "
        "and preset hashes.",
        "",
        "| Date | Scenario | Mode | Runs | First ready | FLP first window | First byte | TTS RTF | Render RTF | Neural FPS | Media | Audio gap | Video hold |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {date} | {scenario} | {mode} | {runs} | {ready:.0f} ms | {window:.0f} ms | "
            "{byte:.0f} ms | {tts:.3f} | {render:.3f} | {fps:.2f} | "
            "{media:.0f} ms | {gap:.0f} ms | {hold:.0f} ms |".format(
                date=str(row["created_utc"] or "")[:19],
                scenario=row["scenario"],
                mode=row["mode"],
                runs=row["measured_runs"] or 0,
                ready=row["median_first_ready_ms"] or 0,
                window=row.get("median_first_window_ms") or 0,
                byte=row["median_first_byte_ms"] or 0,
                tts=row["median_tts_rtf"] or 0,
                render=row["median_render_rtf"] or 0,
                fps=row["median_neural_fps"] or 0,
                media=row["median_media_ms"] or 0,
                gap=row["median_gap_ms"] or 0,
                hold=row.get("median_video_hold_ms") or 0,
            )
        )
    lines.extend(
        [
            "",
            "Use the CSV for filtering and charts. A changed text or preset hash "
            "means the rows are not a controlled like-for-like comparison.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison CSV: {csv_path}")
    print(f"Comparison report: {md_path}")
    return 0


async def run_benchmark(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=30.0) as client:
        health_response = await client.get(f"{base_url}/health")
        health_response.raise_for_status()
        health = health_response.json()
        config_response = await client.get(f"{base_url}/client-config")
        config_response.raise_for_status()
        client_config = config_response.json()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    available = {
        item["id"]: bool(item.get("available"))
        for item in client_config.get("voiceModes", [])
    }
    missing = [mode for mode in modes if not available.get(mode, False)]
    if missing:
        raise RuntimeError(
            "Requested voice modes are unavailable: "
            + ", ".join(missing)
            + ". Check /health preset_voice_status and the preset WAV/transcript."
        )

    output_dir = args.output_root.resolve() / _utc_stamp()
    output_dir.mkdir(parents=True, exist_ok=False)
    fixture_manifest_path = args.fixture_manifest
    fixture_manifest = None
    if fixture_manifest_path.is_file():
        fixture_manifest = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))

    client = AvatarBenchmarkClient(base_url, args.timeout_seconds)
    runs: list[dict[str, Any]] = []
    run_index = 0
    try:
        await client.connect(client_config.get("iceServers", []))
        for mode in modes:
            for warmup_index in range(args.warmups):
                run_index += 1
                instruction = (
                    args.direction_instruction
                    if mode == "preset-direction"
                    else args.design_instruction
                )
                print(f"Warm-up {warmup_index + 1}/{args.warmups}: {mode}", flush=True)
                result = await client.request(
                    text=args.warmup_text,
                    mode=mode,
                    instruction=instruction,
                )
                result.update(
                    run_index=run_index,
                    warmup=True,
                    gpu_after=_gpu_snapshot(),
                )
                runs.append(result)

        for repeat in range(1, args.repeats + 1):
            for mode in modes:
                run_index += 1
                instruction = (
                    args.direction_instruction
                    if mode == "preset-direction"
                    else args.design_instruction
                )
                print(f"Measured repeat {repeat}/{args.repeats}: {mode}", flush=True)
                gpu_before = _gpu_snapshot()
                result = await client.request(
                    text=args.text,
                    mode=mode,
                    instruction=instruction,
                )
                result.update(
                    run_index=run_index,
                    warmup=False,
                    repeat=repeat,
                    gpu_before=gpu_before,
                    gpu_after=_gpu_snapshot(),
                )
                runs.append(result)
    finally:
        await client.close()

    summary = _summarize(runs, modes)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": {
            "python": sys.version,
            "platform": platform.platform(),
            "gpu_initial": _gpu_snapshot(),
        },
        "health": health,
        "client_config": client_config,
        "preset_fixture": fixture_manifest,
        "benchmark": {
            "base_url": base_url,
            "modes": modes,
            "repeats": args.repeats,
            "warmups_per_mode": args.warmups,
            "warmup_text": args.warmup_text,
            "text": args.text,
            "design_instruction": args.design_instruction,
            "direction_instruction": args.direction_instruction,
        },
        "summary": summary,
        "raw_runs": runs,
    }
    _write_json(output_dir / "benchmark.json", payload)
    _write_csv(output_dir / "phrases.csv", runs)
    (output_dir / "report.md").write_text(
        _markdown_report(payload), encoding="utf-8"
    )
    print(f"Benchmark complete: {output_dir}")
    print((output_dir / "report.md").read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preset = commands.add_parser(
        "prepare-preset", description="Generate the fixed Breeze test voice"
    )
    preset.add_argument("--tts-url", default="http://127.0.0.1:7860")
    preset.add_argument(
        "--wav",
        type=Path,
        default=Path("/workspace/inputs/benchmark-voice-preset.wav"),
    )
    preset.add_argument(
        "--transcript",
        type=Path,
        default=Path("/workspace/inputs/benchmark-voice-preset.txt"),
    )
    preset.add_argument(
        "--manifest",
        type=Path,
        default=Path("/workspace/inputs/benchmark-voice-preset.manifest.json"),
    )
    preset.add_argument("--text", default=DEFAULT_PRESET_TEXT)
    preset.add_argument("--instruction", default=DEFAULT_DESIGN_INSTRUCTION)
    preset.add_argument("--cfg-scale", type=float, default=4.0)
    preset.add_argument("--seed", type=int, default=42)
    preset.add_argument("--force", action="store_true")

    run = commands.add_parser(
        "run", description="Run the same text through each WebRTC voice mode"
    )
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/results/benchmarks"),
    )
    run.add_argument("--modes", default=DEFAULT_MODES)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--warmup-text", default="Hello.")
    run.add_argument("--text", default=DEFAULT_BENCHMARK_TEXT)
    run.add_argument(
        "--fixture-manifest",
        type=Path,
        default=Path("/workspace/inputs/benchmark-voice-preset.manifest.json"),
    )
    run.add_argument("--design-instruction", default=DEFAULT_DESIGN_INSTRUCTION)
    run.add_argument(
        "--direction-instruction", default=DEFAULT_DIRECTION_INSTRUCTION
    )
    run.add_argument("--timeout-seconds", type=float, default=900.0)

    compare = commands.add_parser(
        "compare", description="Combine saved scenario results into one report"
    )
    compare.add_argument(
        "--input-root",
        type=Path,
        default=Path("/workspace/results/benchmarks"),
    )
    compare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/results/benchmarks"),
    )
    return parser


async def async_main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare-preset":
        return await prepare_preset(args)
    if args.command == "compare":
        return compare_results(args)
    if args.repeats < 1 or args.warmups < 0:
        raise ValueError("--repeats must be at least 1 and --warmups cannot be negative")
    return await run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
