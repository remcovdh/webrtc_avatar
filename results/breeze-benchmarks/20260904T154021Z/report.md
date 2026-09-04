# Breeze fast-path benchmark

Medians after warm-up.

| Profile | Fast arguments | FlashAttention | First byte | Total | Audio | RTF | Peak GPU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| backbone-decode | `--fast-backbone-decode` | no | 177 ms | 7547 ms | 8.40 s | 0.898 | 8964 MiB |
| backbone-prefill | `--fast-backbone-prefill` | no | 179 ms | 7848 ms | 8.48 s | 0.925 | 14814 MiB |
| codec | `--fast-codec` | no | 104 ms | 7757 ms | 8.40 s | 0.923 | 8896 MiB |
| depth-decoder | `--fast-depth-decoder` | no | 112 ms | 5371 ms | 9.84 s | 0.546 | 8968 MiB |
| eager | `eager` | no | 185 ms | 8133 ms | 8.40 s | 0.968 | 8526 MiB |
| text-encoder | `--fast-text-encoder` | no | 162 ms | 10049 ms | 10.72 s | 0.937 | 10790 MiB |
