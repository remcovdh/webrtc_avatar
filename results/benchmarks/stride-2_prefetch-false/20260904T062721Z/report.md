# Neural avatar repeatable benchmark

Created: `2026-09-04T06:28:22.463027+00:00`

Server build: `neural-avatar-v2e1-benchmark-handshake-fix`

## Fixed inputs

Benchmark text: Hello! This is a neural streaming avatar test. I run repeatable tests because comparable results help this avatar improve. Let's make this a clear and enjoyable journey.

Measured repeats per mode: `1`

## Median results

| Mode | First ready | First byte | TTS RTF | Render RTF | Neural FPS | Media | Gap | Client wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| design | 1122 ms | 193 ms | 0.986 | 1.121 | 12.04 | 7840 ms | 9320 ms | 18327 ms |
| preset-clone | 1113 ms | 192 ms | 0.924 | 1.117 | 11.98 | 8480 ms | 9460 ms | 19067 ms |
| preset-direction | 1040 ms | 246 ms | 1.010 | 1.125 | 11.96 | 7760 ms | 9700 ms | 18532 ms |

## Per-run totals

| Run | Mode | Warm-up | TTS | Render | Media | Gap | TTS RTF | Neural FPS |
| ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | design | yes | 445 ms | 576 ms | 400 ms | 0 ms | 1.113 | 12.14 |
| 2 | preset-clone | yes | 608 ms | 727 ms | 560 ms | 0 ms | 1.086 | 12.35 |
| 3 | preset-direction | yes | 604 ms | 651 ms | 480 ms | 0 ms | 1.258 | 11.54 |
| 4 | design | no | 7731 ms | 8789 ms | 7840 ms | 9320 ms | 0.986 | 12.04 |
| 5 | preset-clone | no | 7835 ms | 9473 ms | 8480 ms | 9460 ms | 0.924 | 11.98 |
| 6 | preset-direction | no | 7840 ms | 8733 ms | 7760 ms | 9700 ms | 1.010 | 11.96 |

## Interpretation rules

- Compare TTS RTF rather than TTS milliseconds when modes produce different speech durations.
- Neural FPS should remain similar because TTS prefetch is disabled.
- This harness measures performance and stability; voice identity and naturalness still require listening.
- Keep JSON and CSV files when changing code or configuration so later versions remain comparable.
