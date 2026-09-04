# Breeze fast-path benchmark

Medians after warm-up.

| Profile | Fast arguments | FlashAttention | First byte | Total | Audio | RTF | Peak GPU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| backbone-decode | `--fast-backbone-decode` | no | 176 ms | 7536 ms | 8.40 s | 0.897 | 8964 MiB |
| codec | `--fast-codec` | no | 104 ms | 7686 ms | 8.40 s | 0.915 | 8896 MiB |
| depth-decoder | `--fast-depth-decoder` | no | 116 ms | 5397 ms | 9.84 s | 0.548 | 8968 MiB |
| eager | `eager` | no | 188 ms | 8196 ms | 8.40 s | 0.976 | 8526 MiB |
