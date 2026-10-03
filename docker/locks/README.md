# Lock files

One file per image with every Python package and its exact version, as
installed in the images this project was tested with. The Dockerfile gives the
file to pip as a constraints file and `docker/check_lock.py` compares the
result at the end of each stage, so a build either reproduces this set or
fails and says what differs.

Also pinned, in `docker/Dockerfile` and the code:

| What | How |
|---|---|
| Base image | by digest (`FROM ...@sha256:`) |
| FasterLivePortrait, Perth, Transformers, jevk5 | by commit |
| TensorRT libraries | by wheel checksum |
| Model weights | by Hugging Face revision (`repo@revision` in the model settings, see `CONFIG.md`) and, for the MediaPipe face model, by checksum |

Not pinned: the Ubuntu packages installed with `apt-get` (they follow the
pinned base image's release) and the build tools pip fetches while compiling a
package from source.

## Changing a dependency on purpose

1. Edit the requirement in `docker/Dockerfile` and delete the lines of the
   packages that may change from the service's lock file.
2. Build without the check: `docker compose build --build-arg VERIFY_LOCK=0 <service>`.
3. Run the tests and `tools/smoke.sh --no-build`.
4. Write the new lock files: `tools/update_locks.sh`.
5. Build again normally; the check must pass.

A rebuild of the dependency layers needs about 12 GB of free disk.
