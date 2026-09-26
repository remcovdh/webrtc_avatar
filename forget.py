#!/usr/bin/env python3
"""Delete logged conversations: one session, everything before a date, or all.

Conversation logs are local JSON lines in results/conversations/
(<YYYYMMDD>-<session>.jsonl); opt-in utterance audio sits in its audio/ folder
and is referenced from the log lines.

    python3 forget.py --session 3f2a9c1b7d4e     # one session
    python3 forget.py --before 2026-10-01        # everything older
    python3 forget.py --all
Add --yes to delete; without it the script only lists what it would delete.

The containers write these files as root, so run it inside the conductor:

    docker compose exec conductor python /workspace/forget.py --all --yes
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def audio_of(log: Path) -> list[Path]:
    paths = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for audio in record.get("audio", []):
            # Logged paths are container paths; map them into this folder.
            paths.append(log.parent / "audio" / Path(audio).name)
    return paths


def select(directory: Path, session: str | None, before: date | None, everything: bool) -> list[Path]:
    logs = sorted(directory.glob("*.jsonl"))
    if everything:
        return logs
    if session:
        return [log for log in logs if log.stem.split("-", 1)[-1] == session]
    if before:
        stamp = before.strftime("%Y%m%d")
        return [log for log in logs if log.stem.split("-", 1)[0] < stamp]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--session")
    which.add_argument("--before", type=date.fromisoformat)
    which.add_argument("--all", action="store_true")
    parser.add_argument("--dir", type=Path, default=Path("/workspace/results/conversations"))
    parser.add_argument("--yes", action="store_true", help="really delete")
    args = parser.parse_args()

    logs = select(args.dir, args.session, args.before, args.all)
    files = [path for log in logs for path in [log, *audio_of(log)] if path.exists()]
    if args.all:
        files += [p for p in (args.dir / "audio").glob("*.wav") if p not in files]
    for path in files:
        print(("deleting " if args.yes else "would delete ") + str(path))
        if args.yes:
            path.unlink()
    if not files:
        print("nothing to delete")
    elif not args.yes:
        print("(dry run; add --yes to delete)")


if __name__ == "__main__":
    main()
