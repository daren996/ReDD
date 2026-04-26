from __future__ import annotations

import os
import unittest
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

from redd.core.data_loader import create_data_loader, get_loader_profile_notes, get_loader_registry
from redd.core.data_population.factory import create_data_populator
from redd.core.data_population.strategies import ProxyRuntimeExtractionStrategy
from redd.core.schema_gen.factory import create_schema_generator
from redd.llm import get_api_key, normalize_provider_name
from redd.llm.hidden_states import HiddenStatesManager


class FactoryAndProviderTests(unittest.TestCase):
    def test_create_data_loader_rejects_unknown_loader_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown loader type"):
            create_data_loader("dataset/unknown", loader_type="mystery")

    def test_loader_registry_helpers_document_supported_loader_family(self) -> None:
        registry = get_loader_registry()
        notes = get_loader_profile_notes()

        self.assertEqual(set(registry), set(notes))
        self.assertEqual(set(registry), {"hf_manifest"})
        self.assertIn("manifest/parquet", notes["hf_manifest"].lower())

    def test_normalize_provider_name_uses_canonical_provider_names(self) -> None:
        self.assertEqual(normalize_provider_name("openai"), "openai")

    def test_get_api_key_reads_environment_for_cloud_providers(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            self.assertEqual(get_api_key(None, "openai"), "env-key")

    @patch("redd.core.data_population.factory.import_module")
    def test_data_populator_factory_uses_unified_orchestrator_for_basic_runs(
        self,
        import_module_mock,
    ) -> None:
        class UnifiedPopulator:
            def __init__(self, config, api_key=None):
                self.config = config
                self.api_key = api_key

        import_module_mock.return_value = SimpleNamespace(DataExtraction=UnifiedPopulator)

        populator = create_data_populator(
            {"mode": "openai"},
            api_key="secret-key",
        )

        self.assertIsInstance(populator, UnifiedPopulator)
        self.assertEqual(populator.config["mode"], "openai")
        self.assertEqual(populator.api_key, "secret-key")
        import_module_mock.assert_called_once_with(".data_extraction", "redd.core.data_population")

    @patch("redd.core.data_population.factory.import_module")
    def test_data_populator_factory_uses_unified_orchestrator_for_local_runs(
        self,
        import_module_mock,
    ) -> None:
        class FakePopulator:
            def __init__(self, config, api_key=None):
                self.config = config
                self.api_key = api_key

        import_module_mock.return_value = SimpleNamespace(DataExtraction=FakePopulator)

        populator = create_data_populator({"mode": "local"})

        self.assertIsInstance(populator, FakePopulator)
        self.assertEqual(populator.config["mode"], "local")
        self.assertIsNone(populator.api_key)
        import_module_mock.assert_called_once_with(".data_extraction", "redd.core.data_population")

    @patch("redd.core.schema_gen.factory.import_module")
    def test_schema_generator_factory_normalizes_provider_before_instantiation(
        self,
        import_module_mock,
    ) -> None:
        class FakeGenerator:
            def __init__(self, config, api_key=None):
                self.config = config
                self.api_key = api_key

        import_module_mock.return_value = SimpleNamespace(SchemaGen=FakeGenerator)

        generator = create_schema_generator({"mode": "openai"}, api_key="secret-key")

        self.assertIsInstance(generator, FakeGenerator)
        self.assertEqual(generator.config["mode"], "openai")
        self.assertEqual(generator.api_key, "secret-key")
        import_module_mock.assert_called_once_with(".schemagen", "redd.core.schema_gen")

    def test_schema_generator_factory_rejects_local_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Local schema generation is not implemented"):
            create_schema_generator({"mode": "local"})

    def test_internal_data_population_package_no_longer_exports_legacy_classes(self) -> None:
        data_population = import_module("redd.core.data_population")

        with self.assertRaises(AttributeError):
            getattr(data_population, "DataPopGPT")
        with self.assertRaises(AttributeError):
            getattr(data_population, "DataPopLocal")
        with self.assertRaises(AttributeError):
            getattr(data_population, "DataPop")

    def test_internal_schema_gen_package_no_longer_exports_provider_specific_aliases(self) -> None:
        schema_gen = import_module("redd.core.schema_gen")

        with self.assertRaises(AttributeError):
            getattr(schema_gen, "SchemaGenDeepSeek")
        with self.assertRaises(AttributeError):
            getattr(schema_gen, "SchemaGenTogether")
        with self.assertRaises(AttributeError):
            getattr(schema_gen, "LegacySchemaGeneratorBase")
        with self.assertRaises(AttributeError):
            getattr(schema_gen, "AdaptiveSamplingMixin")

    def test_proxy_runtime_strategy_has_dedicated_internal_strategy_module(self) -> None:
        self.assertEqual(
            ProxyRuntimeExtractionStrategy.__module__,
            "redd.core.data_population.strategies.proxy_runtime",
        )

    def test_hidden_states_support_lives_under_llm(self) -> None:
        self.assertEqual(HiddenStatesManager.__module__, "redd.llm.hidden_states")


if __name__ == "__main__":
    unittest.main()
