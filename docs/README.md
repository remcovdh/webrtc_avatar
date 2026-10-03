# Documentation index

Start with the [README](../README.md). The settings are in
[CONFIG.md](../CONFIG.md) and the open ideas in [BACKLOG.md](../BACKLOG.md).

## Current

| Document | What it covers |
|---|---|
| [RELEASE.md](RELEASE.md) | Changelog, one entry per version, newest first |
| [architecture/realtime-rendering.md](architecture/realtime-rendering.md) | How the avatar speaks in real time: the pipeline, TensorRT, the audio clock, the speech start gate, lip-sync and motion corrections |
| [architecture/listening.md](architecture/listening.md) | How the avatar listens and answers: decisions, architecture (listener, conductor, System 1, System 2), and the milestones M1-M5 with what was learned in each |
| [../tools/README.md](../tools/README.md) | The diagnosis and measurement scripts |
| [../knowledge/README.md](../knowledge/README.md) | How to write pages the avatar can answer from |

The two architecture documents were written when their part was finished
(v2R and v2W). Their explanations still hold; file paths and setting names in
them follow the current layout.

## Historical

Kept for the record. They describe earlier states and mention files, settings
and features that no longer exist.

| Document | What it is |
|---|---|
| [history/readme-until-v2w.md](history/readme-until-v2w.md) | The README up to v2W: one section per version from v2A to v2W, old settings tables, recorded benchmarks, the decision log and troubleshooting notes |
| [history/state-2026-09-24.md](history/state-2026-09-24.md) | Project status on 24 September 2026, before real-time rendering and listening |
| [history/issues-and-test-plan-2026-09-24.md](history/issues-and-test-plan-2026-09-24.md) | Open issues and the automated test plan of the same date |
