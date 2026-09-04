# Neural avatar repeatable benchmark

Created: `2026-09-04T06:50:43.746284+00:00`

Server build: `neural-avatar-v2e1-benchmark-handshake-fix`

## Fixed inputs

Benchmark text: Hello! This is a neural streaming avatar test. I run repeatable tests because comparable results help this avatar improve. Let's make this a clear and enjoyable journey.

Measured repeats per mode: `3`

## Median results

| Mode | First ready | First byte | TTS RTF | Render RTF | Neural FPS | Media | Gap | Client wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| design | 1146 ms | 192 ms | 0.979 | 1.146 | 11.71 | 7840 ms | 9420 ms | 18440 ms |

## Per-run totals

| Run | Mode | Warm-up | TTS | Render | Media | Gap | TTS RTF | Neural FPS |
| ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | design | yes | 450 ms | 574 ms | 400 ms | 0 ms | 1.125 | 11.96 |
| 2 | design | no | 7669 ms | 8982 ms | 7840 ms | 9420 ms | 0.978 | 11.71 |
| 3 | design | no | 7677 ms | 8958 ms | 7840 ms | 9420 ms | 0.979 | 11.71 |
| 4 | design | no | 7689 ms | 8994 ms | 7840 ms | 9460 ms | 0.981 | 11.66 |

## Interpretation rules

- Compare TTS RTF rather than TTS milliseconds when modes produce different speech durations.
- Neural FPS should remain similar because TTS prefetch is disabled.
- This harness measures performance and stability; voice identity and naturalness still require listening.
- Keep JSON and CSV files when changing code or configuration so later versions remain comparable.
