"""Tests for saving an md_llm chat conversation to its save directory.

A saved chat is a plain ``<docstem>__chat_<UTC>.md`` file — no sidecar metadata,
no transcript linkage (md_llm has no notion of transcripts). The save helper's
pure-I/O part (``_write_chat_md``), the markdown renderer
(``_render_chat_as_markdown``), the memorized save-directory helpers
(``_remember_chat_save_dir`` / ``_chat_save_dir``), the directory validation
(``_save_dir_problem``) and the Streamlit-facing ``_save_conversation``
outcomes (return values; bare-mode st.* calls no-op) are exercised here.

Ported from transcriber_system's test_chat_save.py, adapted to md_llm's
plain-.md save behavior.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import streamlit as st

from md_llm import chat, core, llm
from md_llm.core import Core


def _make_core():
    tmp = tempfile.mkdtemp()
    c = Core(
        base_dir=tmp,
        markdown_dirs=(tmp,),
        chat_save_dir=tmp,
        settings_path=None,
    )
    core._reset_for_tests(c)
    os.makedirs(tmp, exist_ok=True)
    return tmp


def _non_root():
    """Permission checks are meaningless as root (os.access always allows)."""
    return hasattr(os, "geteuid") and os.geteuid() != 0


class ChatSaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_core()

    def tearDown(self):
        core._reset_for_tests(None)

    def test_saved_chat_named_from_doc_stem_with_timestamp(self):
        doc = os.path.join(self.tmp, "my-notes.md")
        with open(doc, "w") as f:
            f.write("# doc\nbody")
        out = chat._write_chat_md(doc, "# chat\nbody")
        base = os.path.basename(out)
        # <slug>__chat_<UTC>.md — slug derived from the doc stem.
        self.assertTrue(base.startswith("my-notes__chat_"))
        self.assertTrue(base.endswith(".md"))
        self.assertTrue(os.path.isfile(out))

    def test_two_saved_chats_get_distinct_names(self):
        doc = os.path.join(self.tmp, "doc.md")
        open(doc, "w").write("x")
        a = chat._write_chat_md(doc, "# chat\nbody")
        b = chat._write_chat_md(doc, "# chat\nbody2")
        self.assertNotEqual(os.path.basename(a), os.path.basename(b))

    def test_saved_chat_has_no_sidecar(self):
        # md_llm writes a plain .md — no .meta.json (it has no transcript concept).
        doc = os.path.join(self.tmp, "doc.md")
        open(doc, "w").write("x")
        out = chat._write_chat_md(doc, "# chat\nbody")
        sidecar = os.path.splitext(out)[0] + ".meta.json"
        self.assertTrue(os.path.isfile(out))
        self.assertFalse(os.path.exists(sidecar))

    def test_saved_chat_without_doc_uses_chat_stem(self):
        out = chat._write_chat_md(None, "# chat\nbody")
        base = os.path.basename(out)
        self.assertTrue(base.startswith("chat__chat_"))
        self.assertTrue(os.path.isfile(out))

    def test_render_chat_as_markdown_includes_turns_and_provenance(self):
        doc = os.path.join(self.tmp, "notes.md")
        with open(doc, "w") as f:
            f.write("# doc\nbody")
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "What is this about?"},
            {"role": "assistant", "content": "A test document."},
        ]
        try:
            md = chat._render_chat_as_markdown(doc, "OpenRouter", "gpt-4o-mini")
        finally:
            st.session_state.pop("_chat_messages", None)
        self.assertIn("gpt-4o-mini", md)
        self.assertIn("What is this about?", md)
        self.assertIn("A test document.", md)

    def test_render_chat_as_markdown_embeds_source_document_before_turns(self):
        doc = os.path.join(self.tmp, "fox.md")
        with open(doc, "w") as f:
            f.write("The quick brown fox jumps over the lazy dog.")
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "What animal is mentioned?"},
            {"role": "assistant", "content": "A fox and a dog."},
        ]
        try:
            md = chat._render_chat_as_markdown(doc, "OpenRouter", "gpt-4o-mini")
        finally:
            st.session_state.pop("_chat_messages", None)
        self.assertIn("## Source document", md)
        self.assertIn("The quick brown fox", md)
        # Source appears before the first user question.
        self.assertLess(
            md.index("The quick brown fox"),
            md.index("What animal is mentioned?"),
        )

    def test_render_chat_as_markdown_omits_source_section_when_unreadable(self):
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "Hello?"},
            {"role": "assistant", "content": "Hi!"},
        ]
        try:
            md = chat._render_chat_as_markdown(
                "/no/such/path/anywhere.md", "OpenRouter", "gpt-4o-mini")
        finally:
            st.session_state.pop("_chat_messages", None)
        self.assertNotIn("## Source document", md)
        self.assertIn("Hello?", md)

    def test_render_chat_as_markdown_without_doc_omits_source_section(self):
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "Hello?"},
        ]
        try:
            md = chat._render_chat_as_markdown(None, "OpenRouter", "gpt-4o-mini")
        finally:
            st.session_state.pop("_chat_messages", None)
        self.assertNotIn("## Source document", md)
        self.assertIn("Hello?", md)

    def test_default_title_is_source_title_plus_first_user_message(self):
        msgs = [
            {"role": "user", "content": "Summarize the key points"},
            {"role": "assistant", "content": "Here they are."},
            {"role": "user", "content": "Now translate to French"},
        ]
        self.assertEqual(
            chat._chat_default_title("My Notes", msgs),
            "My Notes — Summarize the key points",
        )

    def test_default_title_skips_assistant_first_turn(self):
        msgs = [
            {"role": "assistant", "content": "Hello there"},
            {"role": "user", "content": "What is this?"},
        ]
        self.assertEqual(
            chat._chat_default_title("T", msgs), "T — What is this?"
        )

    def test_default_title_without_source_is_first_question(self):
        msgs = [{"role": "user", "content": "What is the capital of France?"}]
        self.assertEqual(
            chat._chat_default_title("", msgs), "What is the capital of France?",
        )
        self.assertEqual(chat._chat_default_title("", []), "Chat")


class SaveDirValidationTests(unittest.TestCase):
    """``_save_dir_problem``: read-only checks of a candidate save directory."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_empty_path_is_a_problem(self):
        self.assertIsNotNone(chat._save_dir_problem(None))
        self.assertIsNotNone(chat._save_dir_problem(""))
        self.assertIsNotNone(chat._save_dir_problem("   "))

    def test_existing_directory_is_fine(self):
        self.assertIsNone(chat._save_dir_problem(self.tmp))

    def test_existing_file_is_rejected(self):
        f = os.path.join(self.tmp, "afile")
        open(f, "w").write("x")
        problem = chat._save_dir_problem(f)
        self.assertIsNotNone(problem)
        self.assertIn("file", problem)

    def test_missing_dir_with_writable_parent_is_fine(self):
        missing = os.path.join(self.tmp, "a", "b", "c")
        self.assertIsNone(chat._save_dir_problem(missing))

    @unittest.skipUnless(_non_root(), "needs a non-root user to test permissions")
    def test_unwritable_existing_directory_is_rejected(self):
        ro = os.path.join(self.tmp, "ro")
        os.makedirs(ro)
        os.chmod(ro, 0o555)
        try:
            problem = chat._save_dir_problem(ro)
            self.assertIsNotNone(problem)
            self.assertIn("write permission", problem)
        finally:
            os.chmod(ro, 0o755)

    @unittest.skipUnless(_non_root(), "needs a non-root user to test permissions")
    def test_missing_dir_under_unwritable_parent_is_rejected(self):
        ro = os.path.join(self.tmp, "ro")
        os.makedirs(ro)
        os.chmod(ro, 0o555)
        try:
            problem = chat._save_dir_problem(os.path.join(ro, "sub"))
            self.assertIsNotNone(problem)
            self.assertIn("write permission", problem)
        finally:
            os.chmod(ro, 0o755)


class SaveDirMemoryTests(unittest.TestCase):
    """The memorized save directory: settings round-trip and use on save."""

    def setUp(self):
        self.tmp = _make_core()

    def tearDown(self):
        core._reset_for_tests(None)

    def test_default_is_host_chat_save_dir(self):
        self.assertEqual(chat._settings_chat_save_dir(), "")
        self.assertEqual(chat._chat_save_dir(), self.tmp)

    def test_remember_then_clear_round_trip(self):
        sub = os.path.join(self.tmp, "mem")
        os.makedirs(sub)
        chat._remember_chat_save_dir(sub)
        self.assertEqual(chat._settings_chat_save_dir(), sub)
        self.assertEqual(chat._chat_save_dir(), sub)
        chat._remember_chat_save_dir("")  # cleared → host default again
        self.assertEqual(chat._chat_save_dir(), self.tmp)
        self.assertEqual(chat._settings_chat_save_dir(), "")

    def test_remember_strips_whitespace(self):
        chat._remember_chat_save_dir("  " + self.tmp + "  ")
        self.assertEqual(chat._settings_chat_save_dir(), self.tmp)

    def test_memorized_dir_survives_core_reload(self):
        # A real settings file (not the in-memory store): remembering the dir,
        # then re-registering a fresh Core over the same settings path, must
        # still see it — that is what "memorised" means across restarts.
        settings = os.path.join(self.tmp, "settings.json")
        sub = os.path.join(self.tmp, "mem")
        os.makedirs(sub)
        core._reset_for_tests(Core(
            base_dir=self.tmp, markdown_dirs=(self.tmp,),
            chat_save_dir=self.tmp, settings_path=settings,
        ))
        chat._remember_chat_save_dir(sub)
        core._reset_for_tests(Core(
            base_dir=self.tmp, markdown_dirs=(self.tmp,),
            chat_save_dir=self.tmp, settings_path=settings,
        ))
        self.assertEqual(chat._chat_save_dir(), sub)

    def test_input_change_callback_memorizes_committed_value(self):
        sub = os.path.join(self.tmp, "mem")
        os.makedirs(sub)
        st.session_state[chat._SAVE_DIR_INPUT_KEY] = sub
        chat._on_save_dir_input_change()
        self.assertEqual(chat._chat_save_dir(), sub)
        st.session_state[chat._SAVE_DIR_INPUT_KEY] = ""  # cleared → default
        chat._on_save_dir_input_change()
        self.assertEqual(chat._chat_save_dir(), self.tmp)
        st.session_state.pop(chat._SAVE_DIR_INPUT_KEY, None)

    def test_write_chat_md_uses_memorized_dir(self):
        sub = os.path.join(self.tmp, "mem")
        os.makedirs(sub)
        chat._remember_chat_save_dir(sub)
        out = chat._write_chat_md(None, "# chat")
        self.assertEqual(os.path.dirname(out), sub)

    def test_write_chat_md_explicit_dir_wins(self):
        sub = os.path.join(self.tmp, "explicit")
        os.makedirs(sub)
        out = chat._write_chat_md(None, "# chat", save_dir=sub)
        self.assertEqual(os.path.dirname(out), sub)

    @unittest.skipUnless(_non_root(), "needs a non-root user to test permissions")
    def test_write_chat_md_unwritable_target_returns_none(self):
        ro = os.path.join(self.tmp, "ro")
        os.makedirs(ro)
        os.chmod(ro, 0o555)
        try:
            # Missing subdirectory under a read-only parent: makedirs fails.
            self.assertIsNone(
                chat._write_chat_md(None, "# chat", save_dir=os.path.join(ro, "sub"))
            )
            # Existing but read-only directory: the write itself fails.
            self.assertIsNone(chat._write_chat_md(None, "# chat", save_dir=ro))
        finally:
            os.chmod(ro, 0o755)


class SaveConversationOutcomeTests(unittest.TestCase):
    """``_save_conversation`` outcomes (bare-mode st.* calls just no-op)."""

    def setUp(self):
        self.tmp = _make_core()
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]

    def tearDown(self):
        st.session_state.pop("_chat_messages", None)
        core._reset_for_tests(None)

    def test_empty_conversation_returns_none(self):
        st.session_state["_chat_messages"] = []
        self.assertIsNone(chat._save_conversation(None, "P", "m"))

    def test_invalid_dir_returns_none_and_writes_nothing(self):
        f = os.path.join(self.tmp, "afile")
        open(f, "w").write("x")
        self.assertIsNone(chat._save_conversation(None, "P", "m", save_dir=f))
        self.assertTrue(os.path.isfile(f))  # untouched — still the file it was

    def test_success_returns_existing_file_path(self):
        out = chat._save_conversation(None, "P", "m", save_dir=self.tmp)
        self.assertTrue(out and os.path.isfile(out))

    def test_success_uses_memorized_dir_when_no_dir_given(self):
        sub = os.path.join(self.tmp, "mem")
        os.makedirs(sub)
        chat._remember_chat_save_dir(sub)
        out = chat._save_conversation(None, "P", "m")
        self.assertEqual(os.path.dirname(out), sub)


class OpenRouterSendMemoryTests(unittest.TestCase):
    """_build_stream's OpenRouter branch memorizes the model + endpoint on send.

    The OpenRouter counterpart of the OpenCode branch's ``_remember_opencode_model``:
    every send promotes the picked model to the ``llm_or_models`` history and
    records the endpoint, so a fresh session reopens on them. The API key is
    deliberately NOT persisted (write-only by design).
    """

    _KEYS = (
        "chat_llm_provider",
        "chat_llm_or_model_sel",
        "chat_llm_or_api_key",
        "chat_llm_or_endpoint",
        "_chat_messages",
    )

    def setUp(self):
        for k in self._KEYS:
            st.session_state.pop(k, None)
        _make_core()

    def tearDown(self):
        for k in self._KEYS:
            st.session_state.pop(k, None)
        core._reset_for_tests(None)

    def _fake_stream(self, captured):
        def fake(messages, **kwargs):
            captured.update(kwargs)
            yield "hi"
        return fake

    def test_send_memorizes_model_and_endpoint(self):
        st.session_state["chat_llm_provider"] = "OpenRouter"
        st.session_state["chat_llm_or_model_sel"] = "test-model"
        st.session_state["chat_llm_or_api_key"] = "sk-test"
        st.session_state["chat_llm_or_endpoint"] = "https://or.example/v1"
        captured = {}
        with patch.object(
            chat.llm, "openrouter_chat_stream", self._fake_stream(captured)
        ):
            stream, err = chat._build_stream(None, {})
        self.assertIsNone(err)
        self.assertEqual("".join(stream), "hi")
        self.assertEqual(captured.get("endpoint"), "https://or.example/v1")
        llm_s = core.get_core().load_settings().get("llm") or {}
        self.assertEqual(llm_s.get("llm_or_models"), ["test-model"])
        self.assertEqual(llm_s.get("llm_or_model_sel"), "test-model")
        self.assertEqual(llm_s.get("llm_or_endpoint"), "https://or.example/v1")
        # The API key never reaches settings — write-only by design.
        self.assertNotIn("llm_or_api_key", llm_s)

    def test_send_uses_default_endpoint_when_none_selected(self):
        st.session_state["chat_llm_provider"] = "OpenRouter"
        st.session_state["chat_llm_or_model_sel"] = "test-model"
        st.session_state["chat_llm_or_api_key"] = "sk-test"
        captured = {}
        with patch.object(
            chat.llm, "openrouter_chat_stream", self._fake_stream(captured)
        ):
            stream, err = chat._build_stream(None, {})
        self.assertIsNone(err)
        self.assertEqual("".join(stream), "hi")
        llm_s = core.get_core().load_settings().get("llm") or {}
        self.assertEqual(
            llm_s.get("llm_or_endpoint"), llm.OPENROUTER_DEFAULT_ENDPOINT
        )

    def test_validation_failure_before_send_memorizes_nothing(self):
        # No API key staged: the branch returns before the remember calls, so
        # settings stay untouched.
        st.session_state["chat_llm_provider"] = "OpenRouter"
        st.session_state["chat_llm_or_model_sel"] = "test-model"
        stream, err = chat._build_stream(None, {})
        self.assertIsNone(stream)
        self.assertIn("No OpenRouter API key", err)
        llm_s = core.get_core().load_settings().get("llm") or {}
        self.assertEqual(llm_s, {})


if __name__ == "__main__":
    unittest.main()
