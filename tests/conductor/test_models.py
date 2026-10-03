"""Tests for model names with a pinned revision."""

import re
import unittest
from pathlib import Path

from shared.models import split_gguf, split_revision

ROOT = Path(__file__).resolve().parents[2]


class ModelNameTests(unittest.TestCase):
    def test_revision_is_optional(self) -> None:
        self.assertEqual(split_revision("org/model@abc123"), ("org/model", "abc123"))
        self.assertEqual(split_revision("org/model"), ("org/model", None))

    def test_gguf_spec_has_repo_revision_and_file(self) -> None:
        self.assertEqual(
            split_gguf("org/model-GGUF@abc123::model-Q4_K_M.gguf"),
            ("org/model-GGUF", "abc123", "model-Q4_K_M.gguf"),
        )
        self.assertEqual(split_gguf("org/m::f.gguf"), ("org/m", None, "f.gguf"))
        for wrong in ("org/model", "org/model::"):
            with self.assertRaises(ValueError):
                split_gguf(wrong)

    def test_every_default_model_is_pinned_to_a_commit(self) -> None:
        """A default without a revision would follow whatever upstream publishes."""
        defaults = []
        for path in sorted((ROOT / "src").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for match in re.finditer(
                r'os\.getenv\(\s*"[A-Z0-9_]*MODEL",\s*((?:"[^"]*"\s*)+)', source
            ):
                defaults.append((path.name, "".join(re.findall(r'"([^"]*)"', match.group(1)))))
        # The conductor image holds four of them; the repo also has the listener's.
        self.assertGreaterEqual(len(defaults), 4)
        for name, default in defaults:
            with self.subTest(file=name, default=default):
                self.assertRegex(default, r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}(::[\w.-]+\.gguf)?$")


if __name__ == "__main__":
    unittest.main()
