# The avatar

## What the avatar is

The avatar is a real-time talking face that runs locally on one computer with an
NVIDIA RTX 5080 laptop GPU. It is streamed to the browser over WebRTC. Its face
is animated from a single portrait photo, and its voice comes from a
text-to-speech model with a cloned preset voice.

## How the avatar speaks

Text is split into short phrases. Chatterbox Turbo turns each phrase into speech.
JoyVASA turns the speech audio into face motion, and FasterLivePortrait renders
the portrait with that motion, 25 frames per second. The audio is the clock:
every video frame is timed against the audio so the mouth stays in sync with the
voice.

## Why the avatar is fast

The main bottleneck was FasterLivePortrait's warping_spade network, which took
97 milliseconds per frame. Running it with TensorRT in FP16 brought that down to
27 milliseconds per frame. Since then the avatar starts speaking about 0.9
seconds after it receives text, and phrases follow each other without pauses.

## How the avatar listens

The browser microphone is streamed to the avatar. Silero voice activity
detection finds speech, NVIDIA's Nemotron 3.5 streaming speech recognition
transcribes Dutch and English, and Nemotron-3-Diarization tells different
speakers apart. The avatar only answers the first speaker in a session.

## How the avatar decides what to say

A fast "System 1" based on the JevK5 decision model classifies each sentence:
is it a question, a remark, a greeting, a request to stop, and which emotion
does it carry. JevK5 replaced the Laya model, which recognised intents and
especially emotions far less reliably. For quick reactions the avatar picks a fitting short phrase. Questions go
to "System 2", which looks up facts in a personal knowledge folder of Markdown
pages and lets a small local language model turn them into a short spoken
answer. The avatar replies in English.

## How the avatar learns

When the avatar misjudges a sentence, the user can correct the class in the web
page. The correction applies immediately to similar sentences. A review script
reads the logged conversations and reports knowledge gaps, unsure decisions,
turns that went wrong, and suggestions for better classes and new knowledge
pages. Retraining System 1 on the corrections comes later.
