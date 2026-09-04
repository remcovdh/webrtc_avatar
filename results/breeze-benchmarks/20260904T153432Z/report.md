# Breeze fast-path benchmark

Medians after warm-up.

| Profile | Fast arguments | FlashAttention | First byte | Total | Audio | RTF | Peak GPU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| backbone-decode | `--fast-backbone-decode` | no | 181 ms | 7836 ms | 8.40 s | 0.933 | 8964 MiB |
| eager | `eager` | no | 184 ms | 8191 ms | 8.40 s | 0.975 | 8526 MiB |
