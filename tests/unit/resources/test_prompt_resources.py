from __future__ import annotations

import importlib
import re
import unittest
from importlib import resources as importlib_resources

import yaml

from redd.core.utils.prompt_registry import get_prompt_spec, iter_prompt_specs
from redd.core.utils.prompt_utils import load_prompt_text, resolve_prompt_reference


class PromptResourceTests(unittest.TestCase):
    def test_registered_prompt_ids_resolve_to_packaged_resources(self) -> None:
        for spec in iter_prompt_specs():
            with self.subTest(prompt_id=spec.id):
                resolved = resolve_prompt_reference(spec.id)
                self.assertTrue(str(resolved).endswith(spec.filename))
                self.assertTrue(load_prompt_text(spec.id).strip())

    def test_legacy_prompt_references_still_resolve(self) -> None:
        resolved = resolve_prompt_reference("prompts/schema_tailor.txt")
        self.assertTrue(str(resolved).endswith("schema_tailor.txt"))

        resolved = resolve_prompt_reference("data_extraction_table_json.txt")
        self.assertTrue(str(resolved).endswith("data_extraction_table_json.txt"))

        prompt_text = load_prompt_text("data_extraction_table_json.txt")
        self.assertTrue(prompt_text.strip())
        self.assertIn("database expert", prompt_text.lower())

    def test_prompt_metadata_matches_registry(self) -> None:
        prompt_root = importlib_resources.files("redd").joinpath("resources").joinpath("prompts")
        for spec in iter_prompt_specs():
            with self.subTest(prompt_id=spec.id):
                metadata_path = prompt_root.joinpath(spec.metadata_filename)
                self.assertTrue(metadata_path.is_file())
                metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))

                self.assertEqual(metadata["id"], spec.id)
                self.assertEqual(metadata["purpose"], spec.purpose)
                self.assertEqual(tuple(metadata["input_fields"]), spec.input_fields)
                self.assertEqual(metadata["output_schema"], spec.output_schema)
                self.assertEqual(tuple(metadata["used_by"]), spec.used_by)
                self.assertEqual(metadata["owner"], spec.owner)

    def test_prompt_template_variables_are_declared(self) -> None:
        for spec in iter_prompt_specs():
            with self.subTest(prompt_id=spec.id):
                text = load_prompt_text(spec.id)
                metadata_path = (
                    importlib_resources.files("redd")
                    .joinpath("resources")
                    .joinpath("prompts")
                    .joinpath(spec.metadata_filename)
                )
                metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
                declared = set(metadata.get("template_variables") or [])
                found = set(re.findall(r"{{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*}}", text))
                self.assertLessEqual(found, declared)

    def test_prompt_output_schemas_are_importable(self) -> None:
        for spec in iter_prompt_specs():
            with self.subTest(prompt_id=spec.id):
                if not spec.output_schema.startswith("redd."):
                    continue
                module_name, object_name = spec.output_schema.rsplit(".", 1)
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, object_name))

    def test_prompt_aliases_resolve_to_registry_specs(self) -> None:
        self.assertEqual(get_prompt_spec("prompts/schemagen.txt").id, "schemagen")
        self.assertEqual(get_prompt_spec("data_extraction_table_json.txt").id, "data_extraction_table")
        self.assertEqual(get_prompt_spec("data_extraction_cmp_str").id, "data_extraction_cmp_str")

    def test_missing_prompt_raises_file_not_found(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "could not be resolved"):
            resolve_prompt_reference("missing_prompt.txt")


if __name__ == "__main__":
    unittest.main()
