"""Tests for md_llm's OpenAI-compatible per-endpoint registry (``md_llm.controls``).

The registry remembers models AND the API key PER endpoint URL, keyed by the
normalized endpoint, so switching endpoints restores the matching model list +
key. These cover the pure read helpers (no disk I/O) and the write helper (disk
I/O goes through the injected Core, stubbed here via a memory-backed Core).

Ported from transcriber_system's test_llm_openai.py (the OaiRegistry* classes),
retargeted at md_llm.controls.
"""

import tempfile
import unittest
from unittest import mock

import streamlit as st

from md_llm import controls, core, llm
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


class OpenRouterMemoryTests(unittest.TestCase):
    """OpenRouter settings are memorized through Core.

    ``_remember_openrouter_model`` is the OpenRouter sibling of the OpenCode
    model memory: it promotes the model to the front of the ``llm_or_models``
    history and records it as the last selection; ``_remember_openrouter_endpoint``
    stores the endpoint URL. The chat panel calls both on every send, and
    ``_seed_openrouter_last_model`` replays the memorized model into the
    selectbox on a fresh mount.
    """

    def setUp(self):
        self._store = {"llm": {}}
        core._reset_for_tests(_make_core(self._store))

    def tearDown(self):
        core._reset_for_tests(None)
        st.session_state.pop("_pending_or_model_sel", None)

    def _read_llm_settings(self):
        return core.get_core()._memory_store.get("llm") or {}

    def test_remember_model_promotes_to_front_and_sets_selection(self):
        controls._remember_openrouter_model("m/a")
        controls._remember_openrouter_model("m/b")
        controls._remember_openrouter_model("m/a")
        s = self._read_llm_settings()
        self.assertEqual(s["llm_or_models"], ["m/a", "m/b"])
        self.assertEqual(s["llm_or_model_sel"], "m/a")

    def test_remember_model_ignores_empty(self):
        controls._remember_openrouter_model("   ")
        self.assertEqual(self._read_llm_settings(), {})

    def test_remember_model_stages_pending_selection(self):
        controls._remember_openrouter_model("m/a")
        self.assertEqual(st.session_state.get("_pending_or_model_sel"), "m/a")

    def test_remember_endpoint_strips_and_stores(self):
        controls._remember_openrouter_endpoint("  https://x.example/v1  ")
        self.assertEqual(
            self._read_llm_settings()["llm_or_endpoint"],
            "https://x.example/v1",
        )

    def test_remember_endpoint_ignores_empty(self):
        controls._remember_openrouter_endpoint("   ")
        self.assertEqual(self._read_llm_settings(), {})


class SeedOpenRouterLastModelTests(unittest.TestCase):
    """_seed_openrouter_last_model fills an ABSENT selectbox key from settings."""

    KEY = "chat_llm_or_model_sel"

    def setUp(self):
        st.session_state.pop(self.KEY, None)

    def tearDown(self):
        st.session_state.pop(self.KEY, None)

    def test_seeds_absent_key_and_returns_model(self):
        seeded = controls._seed_openrouter_last_model(
            {"llm_or_model_sel": "m/a"}, "chat_"
        )
        self.assertEqual(seeded, "m/a")
        self.assertEqual(st.session_state[self.KEY], "m/a")

    def test_existing_selection_wins(self):
        st.session_state[self.KEY] = "m/live"
        seeded = controls._seed_openrouter_last_model(
            {"llm_or_model_sel": "m/a"}, "chat_"
        )
        self.assertEqual(seeded, "")
        self.assertEqual(st.session_state[self.KEY], "m/live")

    def test_no_memorized_model_seeds_nothing(self):
        seeded = controls._seed_openrouter_last_model({}, "chat_")
        self.assertEqual(seeded, "")
        self.assertNotIn(self.KEY, st.session_state)

    def test_respects_prefix(self):
        controls._seed_openrouter_last_model({"llm_or_model_sel": "m/a"})
        self.assertEqual(st.session_state["llm_or_model_sel"], "m/a")
        self.assertNotIn("chat_llm_or_model_sel", st.session_state)


class OpenrouterDropdownOptionsTests(unittest.TestCase):
    """_openrouter_dropdown_options: default → history → free catalog → other."""

    def test_default_first_when_history_empty(self):
        options = controls._openrouter_dropdown_options(
            {}, ["a/b:free", "c/d:free"]
        )
        self.assertEqual(
            options,
            [llm.OPENROUTER_DEFAULT_MODEL, "a/b:free", "c/d:free",
             "(other — type below)"],
        )

    def test_default_not_duplicated_when_in_history(self):
        options = controls._openrouter_dropdown_options(
            {"llm_or_models": [llm.OPENROUTER_DEFAULT_MODEL, "m/a"]}, []
        )
        self.assertEqual(
            options, [llm.OPENROUTER_DEFAULT_MODEL, "m/a", "(other — type below)"]
        )

    def test_history_before_discovered_and_deduped(self):
        options = controls._openrouter_dropdown_options(
            {"llm_or_models": ["m/a"]}, ["m/a", "b/c:free"]
        )
        self.assertEqual(
            options,
            [llm.OPENROUTER_DEFAULT_MODEL, "m/a", "b/c:free",
             "(other — type below)"],
        )

    def test_bogus_entries_skipped(self):
        options = controls._openrouter_dropdown_options(
            {"llm_or_models": [None, ""]}, ["ok/free", None, 42]
        )
        self.assertEqual(
            options,
            [llm.OPENROUTER_DEFAULT_MODEL, "ok/free", "(other — type below)"],
        )


class OpenrouterCachedModelsTests(unittest.TestCase):
    """_openrouter_cached_models fetches once per session, then serves the cache."""

    def setUp(self):
        st.session_state.pop(controls._OPENROUTER_MODELS_CACHE_KEY, None)

    def tearDown(self):
        st.session_state.pop(controls._OPENROUTER_MODELS_CACHE_KEY, None)

    def test_fetches_once_then_serves_cache(self):
        with mock.patch.object(
            llm, "list_openrouter_models", return_value=["a/b:free"]
        ) as fetch:
            first = controls._openrouter_cached_models()
            second = controls._openrouter_cached_models()
        self.assertEqual(first, ["a/b:free"])
        self.assertEqual(second, ["a/b:free"])
        self.assertEqual(fetch.call_count, 1)

    def test_refresh_key_pop_forces_refetch(self):
        with mock.patch.object(
            llm, "list_openrouter_models", return_value=["a/b:free"]
        ) as fetch:
            controls._openrouter_cached_models()
            st.session_state.pop(controls._OPENROUTER_MODELS_CACHE_KEY, None)
            self.assertEqual(
                controls._openrouter_cached_models(), ["a/b:free"]
            )
        self.assertEqual(fetch.call_count, 2)

    def test_failed_fetch_is_cached_until_refresh(self):
        with mock.patch.object(
            llm, "list_openrouter_models", return_value=[]
        ) as fetch:
            self.assertEqual(controls._openrouter_cached_models(), [])
            self.assertEqual(controls._openrouter_cached_models(), [])
        self.assertEqual(fetch.call_count, 1)

    def test_forwarded_endpoint(self):
        with mock.patch.object(llm, "list_openrouter_models") as fetch:
            controls._openrouter_cached_models("https://proxy.example/v1")
        fetch.assert_called_once_with("https://proxy.example/v1")


class CurrentOpencodeVariantTests(unittest.TestCase):
    """_current_opencode_variant resolves the dropdown / custom-input value."""

    def setUp(self):
        for k in ("llm_opencode_variant_sel", "llm_opencode_variant",
                  "chat_llm_opencode_variant_sel", "chat_llm_opencode_variant"):
            st.session_state.pop(k, None)

    def tearDown(self):
        for k in ("llm_opencode_variant_sel", "llm_opencode_variant",
                  "chat_llm_opencode_variant_sel", "chat_llm_opencode_variant"):
            st.session_state.pop(k, None)

    def test_none_when_unset(self):
        self.assertIsNone(controls._current_opencode_variant())

    def test_none_for_explicit_none_option(self):
        st.session_state["llm_opencode_variant_sel"] = "(none)"
        self.assertIsNone(controls._current_opencode_variant())

    def test_returns_preset_selection(self):
        st.session_state["llm_opencode_variant_sel"] = "high"
        self.assertEqual(controls._current_opencode_variant(), "high")

    def test_returns_custom_value_when_other_selected(self):
        st.session_state["llm_opencode_variant_sel"] = "(other — type below)"
        st.session_state["llm_opencode_variant"] = "  turbo  "
        self.assertEqual(controls._current_opencode_variant(), "turbo")

    def test_empty_custom_value_resolves_to_none(self):
        st.session_state["llm_opencode_variant_sel"] = "(other — type below)"
        st.session_state["llm_opencode_variant"] = "   "
        self.assertIsNone(controls._current_opencode_variant())

    def test_respects_prefix(self):
        st.session_state["chat_llm_opencode_variant_sel"] = "max"
        # Default-prefix key stays unset so it doesn't leak across panels.
        self.assertIsNone(controls._current_opencode_variant())
        self.assertEqual(controls._current_opencode_variant("chat_"), "max")


class SelectedOpencodeModelTests(unittest.TestCase):
    """_selected_opencode_model reads the dropdown, or the custom-name input."""

    def setUp(self):
        for k in ("llm_opencode_model_sel", "llm_opencode_model",
                  "chat_llm_opencode_model_sel", "chat_llm_opencode_model"):
            st.session_state.pop(k, None)

    def tearDown(self):
        self.setUp()

    def test_empty_when_unset(self):
        self.assertEqual(controls._selected_opencode_model(""), "")

    def test_returns_dropdown_selection(self):
        st.session_state["llm_opencode_model_sel"] = "zhipuai-coding-plan/glm-5.3"
        self.assertEqual(
            controls._selected_opencode_model(""), "zhipuai-coding-plan/glm-5.3"
        )

    def test_custom_input_wins_when_other_selected(self):
        st.session_state["llm_opencode_model_sel"] = "(other — type below)"
        st.session_state["llm_opencode_model"] = "  prov/m  "
        self.assertEqual(controls._selected_opencode_model(""), "prov/m")

    def test_other_with_empty_input_is_empty(self):
        st.session_state["llm_opencode_model_sel"] = "(other — type below)"
        self.assertEqual(controls._selected_opencode_model(""), "")

    def test_respects_prefix(self):
        st.session_state["chat_llm_opencode_model_sel"] = "prov/chat"
        self.assertEqual(controls._selected_opencode_model(""), "")
        self.assertEqual(controls._selected_opencode_model("chat_"), "prov/chat")


class OpencodeVariantOptionsTests(unittest.TestCase):
    """_opencode_variant_options: model-aware options + highest-effort default."""

    GLM = {"low": {"reasoningEffort": "low"},
           "high": {"reasoningEffort": "high"},
           "max": {"reasoningEffort": "max"}}

    def test_discovered_variants_sorted_with_highest_default(self):
        options, default = controls._opencode_variant_options(
            {"prov/glm-5.3": {"variants": self.GLM}}, "prov/glm-5.3"
        )
        self.assertEqual(
            options, ["(none)", "low", "high", "max", "(other — type below)"]
        )
        self.assertEqual(default, "max")

    def test_model_without_variants_gets_escape_hatches_only(self):
        """Passing --variant to a variant-less model errors, so presets drop."""
        options, default = controls._opencode_variant_options(
            {"prov/m": {"variants": {}}}, "prov/m"
        )
        self.assertEqual(options, ["(none)", "(other — type below)"])
        self.assertEqual(default, "(none)")

    def test_unknown_model_in_known_catalog(self):
        options, default = controls._opencode_variant_options(
            {"prov/other": {"variants": self.GLM}}, "prov/unknown"
        )
        self.assertEqual(options, ["(none)", "(other — type below)"])
        self.assertEqual(default, "(none)")

    def test_unavailable_discovery_falls_back_to_static_presets(self):
        """Old opencode without --verbose: {} details → the static ladder."""
        options, default = controls._opencode_variant_options({}, "prov/m")
        self.assertEqual(
            options,
            ["(none)"] + list(llm.OPENCODE_VARIANTS) + ["(other — type below)"],
        )
        self.assertEqual(default, "(none)")
        self.assertIn("xhigh", options)


if __name__ == "__main__":
    unittest.main()
