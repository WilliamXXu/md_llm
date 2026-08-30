"""Tests for the chat panel's control-state snapshot/restore (``md_llm.chat``).

Streamlit prunes a widget's value from session_state when the widget isn't
rendered on a run. A host that mounts only the active view (the demo's sidebar
buttons + ``if/else``) would therefore wipe every ``chat_*`` control on a
Reader -> chat round-trip. The snapshot/restore helpers mirror the chat
panel's widget values into a non-widget key (which Streamlit does NOT prune)
and seed them back before the controls mount. These exercise that pure
session_state logic directly.
"""

import unittest

import streamlit as st

from md_llm import chat


def _clear_chat_keys():
    """Drop every chat-panel key so tests start from a clean slate."""
    for k in list(st.session_state.keys()):
        if not isinstance(k, str):
            continue
        if (
            k.startswith("chat_")
            or k.startswith("_chat_ssh_")
            or k.startswith("_chat_")
            or k == chat._PANEL_SNAPSHOT_KEY
        ):
            st.session_state.pop(k, None)


class ChatControlKeysTests(unittest.TestCase):
    def setUp(self):
        _clear_chat_keys()

    def tearDown(self):
        _clear_chat_keys()

    def test_includes_chat_llm_and_ssh_widget_keys(self):
        st.session_state["chat_llm_provider"] = "Ollama"
        st.session_state["chat_llm_model_sel"] = "llama3"
        st.session_state["_chat_ssh_local_port"] = 11434
        keys = set(chat._chat_control_keys())
        self.assertIn("chat_llm_provider", keys)
        self.assertIn("chat_llm_model_sel", keys)
        self.assertIn("_chat_ssh_local_port", keys)

    def test_excludes_internal_non_widget_keys(self):
        st.session_state["_chat_messages"] = [{"role": "user", "content": "hi"}]
        st.session_state["_chat_bg_task"] = {"done": True}
        st.session_state["_chat_autossh_proc"] = object()  # a tracked Popen
        st.session_state["_chat_autossh_start"] = False  # button key
        st.session_state["_chat_opencode_clear_sandbox"] = False  # button key
        keys = set(chat._chat_control_keys())
        self.assertNotIn("_chat_messages", keys)
        self.assertNotIn("_chat_bg_task", keys)
        self.assertNotIn("_chat_autossh_proc", keys)
        self.assertNotIn("_chat_autossh_start", keys)
        self.assertNotIn("_chat_opencode_clear_sandbox", keys)

    def test_button_widget_keys_must_live_outside_snapshot_prefixes(self):
        # Regression: the OpenCode clear-sandbox button was originally keyed
        # "chat_llm_opencode_clear_sandbox". That landed in every snapshot and
        # _restore_chat_controls re-injected it before mount — Streamlit then
        # raised StreamlitValueAssignmentNotAllowedError (button keys may never
        # be set via st.session_state). Guard: any key a BUTTON binds must
        # start with "_" so the chat_* snapshot never captures it.
        st.session_state["chat_llm_opencode_hardened"] = True  # checkbox: ok
        st.session_state["chat_llm_opencode_workdir"] = ""     # text input: ok
        from md_llm import controls
        self.assertTrue(controls.OPENCODE_CLEAR_SANDBOX_KEY.startswith("_"))
        keys = set(chat._chat_control_keys())
        self.assertNotIn(controls.OPENCODE_CLEAR_SANDBOX_KEY, keys)

    def test_cline_clear_sandbox_button_key_matches_snapshot_contract(self):
        # The Cline clear-sandbox button builds its key from the same
        # per-panel pattern; it must obey the same no-button-key-in-snapshot
        # rule as the OpenCode one (see the regression note above) — even
        # while the key is live in session_state.
        from md_llm import controls
        self.assertEqual(controls.CLINE_CLEAR_SANDBOX_KEY, "_cline_clear_sandboxchat")
        st.session_state[controls.CLINE_CLEAR_SANDBOX_KEY] = False
        keys = set(chat._chat_control_keys())
        self.assertNotIn(controls.CLINE_CLEAR_SANDBOX_KEY, keys)


class SnapshotChatControlsTests(unittest.TestCase):
    def setUp(self):
        _clear_chat_keys()

    def tearDown(self):
        _clear_chat_keys()

    def test_snapshot_copies_values_under_non_widget_key(self):
        st.session_state["chat_llm_provider"] = "OpenAI-compatible"
        st.session_state["chat_llm_oai_api_key"] = "sk-secret"
        st.session_state["_chat_ssh_identity"] = "~/.ssh/id"
        st.session_state["_chat_messages"] = [{"role": "user", "content": "x"}]

        chat._snapshot_chat_controls()

        snap = st.session_state[chat._PANEL_SNAPSHOT_KEY]
        self.assertEqual(snap["chat_llm_provider"], "OpenAI-compatible")
        self.assertEqual(snap["chat_llm_oai_api_key"], "sk-secret")
        self.assertEqual(snap["_chat_ssh_identity"], "~/.ssh/id")
        # Internal chat keys are never mirrored.
        self.assertNotIn("_chat_messages", snap)

    def test_snapshot_is_a_noop_when_no_control_keys_present(self):
        st.session_state["_chat_messages"] = [{"role": "user", "content": "x"}]
        chat._snapshot_chat_controls()
        self.assertNotIn(chat._PANEL_SNAPSHOT_KEY, st.session_state)


class RestoreChatControlsTests(unittest.TestCase):
    def setUp(self):
        _clear_chat_keys()

    def tearDown(self):
        _clear_chat_keys()

    def test_restore_seeds_missing_keys_from_snapshot(self):
        # The snapshot taken on a prior chat-view run.
        st.session_state[chat._PANEL_SNAPSHOT_KEY] = {
            "chat_llm_provider": "OpenRouter",
            "chat_llm_or_model_sel": "openai/gpt-4o-mini",
            "chat_llm_or_api_key": "or-key",
            "_chat_ssh_local_port": 11434,
        }
        # Simulate Streamlit pruning the widget keys on a Reader-view run: only
        # the non-widget snapshot survives.
        chat._restore_chat_controls()
        self.assertEqual(
            st.session_state["chat_llm_provider"], "OpenRouter"
        )
        self.assertEqual(
            st.session_state["chat_llm_or_model_sel"], "openai/gpt-4o-mini"
        )
        self.assertEqual(st.session_state["chat_llm_or_api_key"], "or-key")
        self.assertEqual(st.session_state["_chat_ssh_local_port"], 11434)

    def test_restore_does_not_overwrite_keys_already_set(self):
        st.session_state[chat._PANEL_SNAPSHOT_KEY] = {
            "chat_llm_provider": "OpenRouter",
            "chat_llm_or_model_sel": "stale",
        }
        # An on_change callback (or the pending-selection block) set a fresh
        # value this run — restore must leave it alone.
        st.session_state["chat_llm_provider"] = "Ollama"
        chat._restore_chat_controls()
        self.assertEqual(st.session_state["chat_llm_provider"], "Ollama")
        # A key absent this run is still seeded from the snapshot.
        self.assertEqual(st.session_state["chat_llm_or_model_sel"], "stale")

    def test_restore_is_a_noop_without_snapshot(self):
        st.session_state["chat_llm_provider"] = "Ollama"
        chat._restore_chat_controls()  # no snapshot present
        self.assertEqual(st.session_state["chat_llm_provider"], "Ollama")


class RoundTripTests(unittest.TestCase):
    """End-to-end: snapshot -> prune widgets -> restore brings values back."""

    def setUp(self):
        _clear_chat_keys()

    def tearDown(self):
        _clear_chat_keys()

    def test_round_trip_survives_simulated_tab_switch(self):
        # 1) Chat view: user has set controls; end of render snapshots them.
        st.session_state["chat_llm_provider"] = "OpenAI-compatible"
        st.session_state["chat_llm_oai_endpoint"] = "https://api.groq.com/openai/v1"
        st.session_state["chat_llm_oai_model_sel"] = "qwen/qwen3-32b"
        st.session_state["chat_llm_oai_api_key"] = "gsk_abc"
        chat._snapshot_chat_controls()

        # 2) Reader view: Streamlit prunes the unmounted widget keys. The
        #    non-widget snapshot (and other _chat_* keys) survive.
        for k in ("chat_llm_provider", "chat_llm_oai_endpoint",
                  "chat_llm_oai_model_sel", "chat_llm_oai_api_key"):
            st.session_state.pop(k, None)
        self.assertNotIn("chat_llm_provider", st.session_state)
        self.assertIn(chat._PANEL_SNAPSHOT_KEY, st.session_state)

        # 3) Back to chat view: restore seeds the pruned keys before mount.
        chat._restore_chat_controls()
        self.assertEqual(
            st.session_state["chat_llm_provider"], "OpenAI-compatible"
        )
        self.assertEqual(
            st.session_state["chat_llm_oai_endpoint"],
            "https://api.groq.com/openai/v1",
        )
        self.assertEqual(
            st.session_state["chat_llm_oai_model_sel"], "qwen/qwen3-32b"
        )
        self.assertEqual(st.session_state["chat_llm_oai_api_key"], "gsk_abc")


if __name__ == "__main__":
    unittest.main()
