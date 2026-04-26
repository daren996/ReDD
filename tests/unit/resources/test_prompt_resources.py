from __future__ import annotations

import unittest

from redd.config import resolve_repo_path
from redd.core.utils.prompt_utils import load_prompt_text, resolve_prompt_reference


class PromptResourceTests(unittest.TestCase):
    def test_repo_relative_prompt_resolution_prefers_repository_files(self) -> None:
        resolved = resolve_prompt_reference("prompts/schemagen_4_0.txt")
        self.assertEqual(resolved, resolve_repo_path("prompts/schemagen_4_0.txt"))

    def test_packaged_prompt_resolution_supports_short_names(self) -> None:
        resolved = resolve_prompt_reference("schemagen_4_0.txt")
        self.assertTrue(str(resolved).endswith("schemagen_4_0.txt"))

        prompt_text = load_prompt_text("schemagen_4_0.txt")
        self.assertTrue(prompt_text.strip())
        self.assertIn("schema", prompt_text.lower())

    def test_missing_prompt_raises_file_not_found(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "could not be resolved"):
            resolve_prompt_reference("missing_prompt.txt")


if __name__ == "__main__":
    unittest.main()
