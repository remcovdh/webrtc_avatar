"""Model names with a pinned revision.

Models are downloaded from the Hugging Face Hub at first start. A name may end
in `@<revision>` (a commit hash), so everyone gets exactly the weights this
code was tested with instead of whatever the repository holds today:

    nvidia/Nemotron-3-Diarization@f667ed73aee57d40cc39428eb768b4fd87a0a29e
    ibm-granite/granite-4.0-1b-GGUF@b27c2fe3...::granite-4.0-1b-Q4_K_M.gguf

Without a revision the latest version is used. Only the standard library, so
every service can import it.
"""

from __future__ import annotations


def split_revision(name: str) -> tuple[str, str | None]:
    """`repo@revision` -> (`repo`, `revision`); no `@` gives revision None."""
    repo, _, revision = name.partition("@")
    return repo, revision or None


def split_gguf(spec: str) -> tuple[str, str | None, str]:
    """`repo[@revision]::file.gguf` -> (`repo`, `revision`, `file.gguf`)."""
    name, separator, filename = spec.partition("::")
    if not separator or not filename:
        raise ValueError(f"Expected '<repo>[@<revision>]::<file>.gguf', got {spec!r}")
    repo, revision = split_revision(name)
    return repo, revision, filename


def download_gguf(spec: str) -> str:
    """Download (or find in the cache) the file of a GGUF spec; returns its path."""
    from huggingface_hub import hf_hub_download

    repo, revision, filename = split_gguf(spec)
    return hf_hub_download(repo, filename, revision=revision)
