# The project

## Goal of the project

The project builds a local, real-time neural avatar that can listen, react and
answer, as a test bed for new speech and animation models. Every part (speech
recognition, System 1, knowledge, the language model, text-to-speech and the
renderer) sits behind a small interface so a better model can be swapped in
quickly.

## Where it runs

Everything runs in Docker containers on an Ubuntu virtual machine under QEMU,
with the RTX 5080 laptop GPU passed through to the virtual machine. The web page
is served over HTTPS on port 8443 so the microphone works from other computers.

## Privacy

Conversations are logged only on the local machine, as one line per turn.
Microphone audio is only saved when that is switched on explicitly, and a
forget script deletes logged conversations.
