# Test and measurement tools

Scripts used to find and verify problems with the avatar. They live in the repo
(not in a scratch folder) so they survive VM restarts. Run them from the repo
root with the stack up (`docker compose up -d`). Scripts that run inside a
container are fed through `python -` so nothing needs to be copied in.

| Tool | What it answers | How to run |
|---|---|---|
| `../review_recording.sh` | What does a phrase look like (frames, mouth close-ups)? | `./review_recording.sh "text"` |
| `mic_turn.py` | What does System 1/2 do with this spoken input? | `docker compose exec -T webrtc-avatar python - en-US /workspace/tools/audio/question.wav < tools/mic_turn.py` |
| `multi_say.py` | Do several sentences in one session arrive complete? Records `results/multi_say.wav`. | `docker compose exec -T webrtc-avatar python - "Great, then we agree." "Okay, I'll stop." < tools/multi_say.py` |
| `transcribe.sh` | What did the avatar really say? (ASR as an "ear") | `tools/transcribe.sh results/some.wav ...` |
| `eval_system1.py` | How well does a JevK5 model classify 18 Dutch/English utterances? | pause the avatar, then `docker compose run --rm --no-deps -T conductor python - < tools/eval_system1.py` |
| `eval_system2.py` | Retrieval threshold and LLM answers/speed | pause the avatar, then `docker compose run --rm --no-deps -T conductor python - < tools/eval_system2.py` |
| `../lip_sync_analysis.py` | Does JoyVASA's mouth lead or lag the voice? | see its header |

"Pause the avatar" means `docker compose stop webrtc-avatar tts` (and
`docker compose start webrtc-avatar tts` afterwards): the evaluation scripts
load a second copy of a model, which does not fit next to the running avatar on
the 16 GB GPU.

Test audio lives in `tools/audio/` (committed): `question.wav` ("Why is the
avatar so fast?", answerable from `knowledge/`), `gap.wav` ("What is the capital
of France?", a knowledge gap) and `dutch.wav` (a Dutch question). The avatar
container sees them as `/workspace/tools/audio/` (mounted read-only). Make new
ones with the running TTS, e.g.:

```bash
curl -s -o /tmp/q.raw -F "text=Why is the avatar so fast?" http://127.0.0.1:7860/v1/audio/speech
ffmpeg -y -f s16le -ar 24000 -ac 1 -i /tmp/q.raw tools/audio/q.wav
```
