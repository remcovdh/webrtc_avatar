# Inputs

What the avatar looks and sounds like. Only this file and the example
portraits are in git; everything else in this folder stays on your machine.

| File | What it is |
|---|---|
| `avatar.jpg` | The portrait: clear, front-facing, one face. Set another file with `AVATAR_IMAGE_PATH`. |
| `voice-preset-nohello.wav` | The reference recording of the voice to clone (5-15 seconds of clean speech, mono WAV). Set another file with `AVATAR_PRESET_AUDIO_PATH`. |

Without a reference recording the avatar still works: it speaks with
Chatterbox's built-in voice, and the page offers only that voice mode.

## Voices need consent

A voice identifies a person, and a cloned voice can say things that person
never said. So:

- Use a voice only with the clear consent of the person it belongs to, for
  this use, and keep a note of that consent (who, when, for what) next to the
  recording, for example `voice-preset-nohello.consent.txt`.
- Remove the recording when that person asks.
- Never commit a recording or anything spoken with a cloned voice. `.gitignore`
  keeps this folder out of git; do not force-add files here.
- An AI-generated voice that imitates no real person needs no consent, but
  check the terms of the tool that made it.

The same goes for portraits of real people. The example portraits here were
generated with Midjourney and show no real person.

Recordings the scripts make (`benchmark-voice-preset.*`, written by
`scripts/benchmark.sh`) also land here and are ignored by git.
