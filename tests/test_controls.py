"""Tests for md_llm's OpenAI-compatible per-endpoint registry (``md_llm.controls``).

The registry remembers models AND the API key PER endpoint URL, keyed by the
normalized endpoint, so switching endpoints restores the matching model list +
key. These cover the pure read helpers (no disk I/O) and the write helper (disk
I/O goes through the injected Core, stubbed here via a memory-backed Core).

Ported from transcriber_system's test_llm_openai.py (the OaiRegistry* classes),
retargeted at md_llm.controls.
"""

import os
import tempfile
import unittest

from md_llm import controls, core
from md_llm.core import Core


class _MemorySettingsCore(Core):
    """A Core whose settings live in an in-memory dict (no file I/O)."""

    def __init__(self, base_dir, markdown_dirs, chat_save_dir):
        super().__init__(
            base_dir=base_dir,
            markdown_dirs=markdown_dirs,
            chat_save_dir=chat_save_dir,
            settings_path=None,
        )

    # load_settings/save_settings already fall back to _memory_store when
    # settings_path is None — nothing else to override.


def _make_core(store=None):
    tmp = tempfile.mkdtemp()
    c = _MemorySettingsCore(base_dir=tmp, markdown_dirs=(tmp,), chat_save_dir=tmp)
    if store:
        c._memory_store = dict(store)
    return c


class OaiRegistryReadTests(unittest.TestCase):
    def setUp(self):
        core._reset_for_tests(_make_core())

    def tearDown(self):
        core._reset_for_tests(None)

    def test_entry_returns_defaults_for_unknown_endpoint(self):
        saved = {"oai_endpoints": {}}
        entry = controls._oai_registry_entry(saved, "https://x/v1")
        self.assertEqual(entry, {"models": [], "last_model": "", "api_key": ""})

    def test_entry_returns_stored_models_last_model_and_key(self):
        saved = {
            "oai_endpoints": {
                "https://api.groq.com/openai/v1": {
                    "models": ["qwen/qwen3-32b", "gpt-4o-mini"],
                    "last_model": "qwen/qwen3-32b",
                    "api_key": "gsk_abc",
                },
            }
        }
        entry = controls._oai_registry_entry(
            saved, "https://api.groq.com/openai/v1"
        )
        self.assertEqual(entry["models"], ["qwen/qwen3-32b", "gpt-4o-mini"])
        self.assertEqual(entry["last_model"], "qwen/qwen3-32b")
        self.assertEqual(entry["api_key"], "gsk_abc")

    def test_endpoint_normalization_strips_trailing_slash(self):
        saved = {
            "oai_endpoints": {
                "https://api.groq.com/openai/v1": {
                    "models": ["m"], "last_model": "m", "api_key": "k",
                },
            }
        }
        entry = controls._oai_registry_entry(
            saved, "https://api.groq.com/openai/v1/"
        )
        self.assertEqual(entry["models"], ["m"])
        self.assertEqual(entry["api_key"], "k")

    def test_known_endpoints_lists_all_registry_keys(self):
        saved = {
            "oai_endpoints": {
                "https://a/v1": {"models": []},
                "https://b/v1": {"models": []},
            }
        }
        self.assertEqual(
            sorted(controls._oai_known_endpoints(saved)[0]),
            ["https://a/v1", "https://b/v1"],
        )

    def test_registry_is_isolated_from_openrouter_history(self):
        saved = {"llm_or_models": ["openrouter-only"]}
        self.assertEqual(controls._oai_registry(saved), {})
        self.assertEqual(
            controls._oai_registry_entry(saved, "https://x/v1"),
            {"models": [], "last_model": "", "api_key": ""},
        )


class OaiRegistryWriteTests(unittest.TestCase):
    """_save_oai_registry_entry writes the per-endpoint entry through Core."""

    def setUp(self):
        self._store = {"llm": {}}
        core._reset_for_tests(_make_core(self._store))

    def tearDown(self):
        core._reset_for_tests(None)

    def _read_store(self):
        return core.get_core()._memory_store

    def test_remember_model_creates_entry_promoting_model_to_front(self):
        controls._save_oai_registry_entry(
            "https://api.groq.com/openai/v1", last_model="qwen3-32b",
        )
        entry = self._read_store()["llm"]["oai_endpoints"]["https://api.groq.com/openai/v1"]
        self.assertEqual(entry["models"], ["qwen3-32b"])
        self.assertEqual(entry["last_model"], "qwen3-32b")

    def test_remember_model_promotes_existing_to_front_dedup(self):
        self._read_store()["llm"] = {"oai_endpoints": {
            "https://api.groq.com/openai/v1": {
                "models": ["old", "qwen3-32b"], "last_model": "old",
            },
        }}
        controls._save_oai_registry_entry(
            "https://api.groq.com/openai/v1", last_model="qwen3-32b",
        )
        entry = self._read_store()["llm"]["oai_endpoints"]["https://api.groq.com/openai/v1"]
        self.assertEqual(entry["models"], ["qwen3-32b", "old"])
        self.assertEqual(entry["last_model"], "qwen3-32b")

    def test_remember_key_does_not_touch_models(self):
        self._read_store()["llm"] = {"oai_endpoints": {
            "https://x/v1": {"models": ["a", "b"], "last_model": "a"},
        }}
        controls._save_oai_registry_entry("https://x/v1", api_key="sk-new")
        entry = self._read_store()["llm"]["oai_endpoints"]["https://x/v1"]
        self.assertEqual(entry["api_key"], "sk-new")
        self.assertEqual(entry["models"], ["a", "b"])

    def test_remember_key_and_model_together(self):
        controls._save_oai_registry_entry(
            "https://x/v1", last_model="m1", api_key="k1",
        )
        entry = self._read_store()["llm"]["oai_endpoints"]["https://x/v1"]
        self.assertEqual(entry["models"], ["m1"])
        self.assertEqual(entry["last_model"], "m1")
        self.assertEqual(entry["api_key"], "k1")

    def test_endpoints_kept_separate(self):
        self._read_store()["llm"] = {"oai_endpoints": {
            "https://a/v1": {"models": ["a-model"], "last_model": "a-model"},
        }}
        controls._save_oai_registry_entry(
            "https://b/v1", last_model="b-model", api_key="kb",
        )
        reg = self._read_store()["llm"]["oai_endpoints"]
        self.assertEqual(reg["https://a/v1"]["models"], ["a-model"])
        self.assertNotIn("api_key", reg["https://a/v1"])
        self.assertEqual(reg["https://b/v1"]["models"], ["b-model"])
        self.assertEqual(reg["https://b/v1"]["api_key"], "kb")


if __name__ == "__main__":
    unittest.main()
