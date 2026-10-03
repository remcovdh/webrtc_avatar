"""Fail the image build when the installed Python packages differ from the
lock file of the service (docker/locks/<service>.txt).

The lock file is the exact set of packages the service was tested with. It is
given to pip as a constraints file, and this check proves the result: a build
either reproduces that set or stops here and says what differs.

    python check_lock.py /opt/locks/avatar.txt

To change dependencies on purpose, see docker/locks/README.md.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    lock_path = Path(sys.argv[1])
    locked = {}
    for line in lock_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, version = line.partition("==")
            locked[normalise(name)] = version
    installed = {
        normalise(dist.metadata["Name"]): dist.version
        for dist in metadata.distributions()
        if dist.metadata["Name"]
    }
    problems = []
    for name in sorted(set(locked) | set(installed)):
        want, have = locked.get(name), installed.get(name)
        if want != have:
            problems.append(f"  {name}: locked {want or '-'}, installed {have or '-'}")
    if problems:
        print(f"Installed packages differ from {lock_path.name}:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"{len(installed)} packages match {lock_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
