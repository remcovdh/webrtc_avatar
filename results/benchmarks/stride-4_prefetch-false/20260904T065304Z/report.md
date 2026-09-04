# Neural avatar repeatable benchmark

Created: `2026-09-04T06:53:51.373250+00:00`

Server build: `neural-avatar-v2e1-benchmark-handshake-fix`

## Fixed inputs

Benchmark text: Hello! This is a neural streaming avatar test. I run repeatable tests because comparable results help this avatar improve. Let's make this a clear and enjoyable journey.

Measured repeats per mode: `3`

## Median results

| Mode | First ready | First byte | TTS RTF | Render RTF | Neural FPS | Media | Gap | Client wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| design | 900 ms | 194 ms | 0.981 | 0.605 | 11.77 | 7840 ms | 6320 ms | 15128 ms |

## Per-run totals

| Run | Mode | Warm-up | TTS | Render | Media | Gap | TTS RTF | Neural FPS |
| ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | design | yes | 440 ms | 362 ms | 480 ms | 0 ms | 0.917 | 12.88 |
| 2 | design | no | 7746 ms | 4776 ms | 7840 ms | 6320 ms | 0.988 | 11.59 |
| 3 | design | no | 7688 ms | 4743 ms | 7840 ms | 6360 ms | 0.981 | 11.79 |
| 4 | design | no | 7692 ms | 4727 ms | 7840 ms | 6320 ms | 0.981 | 11.77 |

## Interpretation rules

- Compare TTS RTF rather than TTS milliseconds when modes produce different speech durations.
- Neural FPS should remain similar because TTS prefetch is disabled.
- This harness measures performance and stability; voice identity and naturalness still require listening.
- Keep JSON and CSV files when changing code or configuration so later versions remain comparable.
