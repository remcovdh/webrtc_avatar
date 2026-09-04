# Neural avatar repeatable benchmark

Created: `2026-09-04T06:52:19.322378+00:00`

Server build: `neural-avatar-v2e1-benchmark-handshake-fix`

## Fixed inputs

Benchmark text: Hello! This is a neural streaming avatar test. I run repeatable tests because comparable results help this avatar improve. Let's make this a clear and enjoyable journey.

Measured repeats per mode: `3`

## Median results

| Mode | First ready | First byte | TTS RTF | Render RTF | Neural FPS | Media | Gap | Client wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| design | 981 ms | 194 ms | 0.971 | 0.781 | 11.80 | 7920 ms | 7340 ms | 16285 ms |

## Per-run totals

| Run | Mode | Warm-up | TTS | Render | Media | Gap | TTS RTF | Neural FPS |
| ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | design | yes | 471 ms | 461 ms | 480 ms | 0 ms | 0.981 | 12.82 |
| 2 | design | no | 7687 ms | 6258 ms | 7920 ms | 7400 ms | 0.971 | 11.72 |
| 3 | design | no | 7701 ms | 6164 ms | 7920 ms | 7320 ms | 0.972 | 11.83 |
| 4 | design | no | 7680 ms | 6189 ms | 7920 ms | 7340 ms | 0.970 | 11.80 |

## Interpretation rules

- Compare TTS RTF rather than TTS milliseconds when modes produce different speech durations.
- Neural FPS should remain similar because TTS prefetch is disabled.
- This harness measures performance and stability; voice identity and naturalness still require listening.
- Keep JSON and CSV files when changing code or configuration so later versions remain comparable.
