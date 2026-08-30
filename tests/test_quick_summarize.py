"""Tests for the Reader's ⚡ Summarize quick action (default: md_llm.reader).

The button opens a new "Summary" chat session tab for the active document,
stages the quick-action prompt into that session (per-document + per-session
scoping), and switches to the LLM chat; the chat panel pops it and sends it
through the exact ``st.chat_input`` pipeline. These exercise the session_state
plumbing and the send/cleanup logic directly — the Streamlit-facing widget
rendering is covered by the AppTest harness.
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import streamlit as st

from md_llm import chat, docs, reader
from md_llm.core import Core, _reset_for_tests


def _clear_keys():
    """Drop every key the quick action and the docs registry can touch."""
    for k in list(st.session_state.keys()):
        if not isinstance(k, str):
            continue
        if (
            k.startswith("_md_llm_")
            or k.startswith("_chat_")
            or k.startswith("_reader_")
            or k.startswith("chat_")
            or k.startswith("_app_")
        ):
            st.session_state.pop(k, None)


def _wait_done(task, timeout=5.0):
    """Block until the background worker marks ``task`` done (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline and not task.get("done"):
        time.sleep(0.01)
    return bool(task.get("done"))


class QuickPromptKeyTests(unittest.TestCase):
    def setUp(self):
        _clear_keys()

    def tearDown(self):
        _clear_keys()

    def test_reader_and_chat_quick_prompt_keys_match(self):
        docs.add_document("notes.md")
        self.assertEqual(
            reader._reader_quick_prompt_key(),
            chat._staged_quick_prompt_key(),
        )
        self.assertEqual(
            reader._reader_quick_prompt_key(),
            "_reader_quick_prompt__doc__notes.md",
        )

    def test_single_doc_mode_uses_the_legacy_bare_key(self):
        self.assertEqual(
            reader._reader_quick_prompt_key(), "_reader_quick_prompt"
        )
        self.assertEqual(
            chat._staged_quick_prompt_key(), "_reader_quick_prompt"
        )

    def test_quick_prompt_key_follows_the_active_session(self):
        docs.add_document("notes.md")
        docs.add_chat("notes.md")
        self.assertEqual(
            reader._reader_quick_prompt_key(),
            "_reader_quick_prompt__chat__2__doc__notes.md",
        )
        docs.set_active_chat(1, "notes.md")
        self.assertEqual(
            reader._reader_quick_prompt_key(), chat._staged_quick_prompt_key()
        )

    def test_staging_targets_the_new_summary_session(self):
        # The ⚡ button opens a dedicated "Summary" session (add_chat activates
        # it) before staging, so reader and chat both resolve the staged key —
        # and the conversation — to that new session.
        docs.add_document("notes.md")
        sid = docs.add_chat("notes.md", label="Summary")
        self.assertEqual(sid, 2)
        self.assertEqual(docs.chat_session_label("notes.md", 2), "Summary")
        self.assertEqual(docs.active_chat("notes.md"), 2)
        self.assertEqual(
            reader._reader_quick_prompt_key(),
            "_reader_quick_prompt__chat__2__doc__notes.md",
        )
        self.assertEqual(
            chat._staged_quick_prompt_key(), reader._reader_quick_prompt_key()
        )
        self.assertEqual(
            chat._chat_state_key("_chat_messages"),
            "_chat_messages__chat__2__doc__notes.md",
        )

    def test_default_prompt_is_the_configured_summarizer(self):
        self.assertIn("摘要助手", reader.QUICK_SUMMARY_PROMPT)
        self.assertIn("bullet points", reader.QUICK_SUMMARY_PROMPT)
        self.assertIn("不得添加原文没有的信息", reader.QUICK_SUMMARY_PROMPT)


class CurrentPromptTests(unittest.TestCase):
    def setUp(self):
        _clear_keys()

    def tearDown(self):
        _clear_keys()

    def test_defaults_to_the_factory_prompt(self):
        self.assertEqual(reader._current_quick_prompt(),
                         reader.QUICK_SUMMARY_PROMPT)

    def test_prefers_the_edited_copy(self):
        st.session_state[reader._QUICK_PROMPT_SAVED] = " custom prompt "
        self.assertEqual(reader._current_quick_prompt(), "custom prompt")

    def test_cleared_editor_falls_back_to_the_default(self):
        # The user cleared the box: the button falls back to the factory
        # default rather than sending an empty prompt.
        st.session_state[reader._QUICK_PROMPT_SAVED] = ""
        self.assertEqual(reader._current_quick_prompt(),
                         reader.QUICK_SUMMARY_PROMPT)

    def test_save_mirrors_the_widget_value(self):
        st.session_state[reader._QUICK_PROMPT_EDIT] = "edited"
        reader._save_quick_prompt_edit()
        self.assertEqual(st.session_state[reader._QUICK_PROMPT_SAVED], "edited")


class SendStagedPromptTests(unittest.TestCase):
    """chat._send_staged_quick_prompt: the staged prompt rides the exact
    chat_input pipeline — user turn appended, stream on a background worker."""

    def setUp(self):
        _clear_keys()
        self.tmp = tempfile.mkdtemp(prefix="mdllm_quick_")
        self.doc = os.path.join(self.tmp, "notes.md")
        with open(self.doc, "w") as f:
            f.write("# doc\nbody")
        _reset_for_tests(Core(
            base_dir=self.tmp,
            markdown_dirs=(self.tmp,),
            chat_save_dir=self.tmp,
        ))
        st.session_state["chat_llm_provider"] = "OpenRouter"
        st.session_state["chat_llm_or_model_sel"] = "test-model"

    def tearDown(self):
        _clear_keys()
        _reset_for_tests()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _stage(self, prompt="你是一个摘要助手，请概括给定文本。"):
        st.session_state[chat._staged_quick_prompt_key()] = prompt

    def test_staged_prompt_is_sent_as_the_next_user_turn(self):
        self._stage("summarize this")
        with patch.object(
            chat, "_build_stream",
            return_value=(iter(["hello ", "world"]), None),
        ):
            task = chat._send_staged_quick_prompt(self.doc)

        self.assertIsNotNone(task)
        # The staged prompt is consumed and appended verbatim as a user turn.
        self.assertNotIn(chat._staged_quick_prompt_key(), st.session_state)
        msgs = st.session_state["_chat_messages"]
        self.assertEqual(
            msgs, [{"role": "user", "content": "summarize this"}],
        )
        # The task is live under the session's bg-task key and streams.
        self.assertIs(st.session_state["_chat_bg_task"], task)
        self.assertTrue(_wait_done(task))
        self.assertEqual(task["text"], "hello world")
        self.assertIsNone(task["error"])

    def test_appends_to_an_existing_conversation(self):
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        self._stage("now summarize")
        with patch.object(
            chat, "_build_stream",
            return_value=(iter(["ok"]), None),
        ):
            task = chat._send_staged_quick_prompt(self.doc)
        self.assertTrue(_wait_done(task))
        msgs = st.session_state["_chat_messages"]
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[-1], {"role": "user", "content": "now summarize"})

    def test_validation_failure_rolls_the_turn_back(self):
        self._stage("summarize this")
        with patch.object(
            chat, "_build_stream", return_value=(None, "no model picked"),
        ):
            ret = chat._send_staged_quick_prompt(self.doc)
        self.assertIsNone(ret)
        self.assertNotIn(chat._staged_quick_prompt_key(), st.session_state)
        # The dangling user turn is rolled back (same as a failed typed send:
        # the setdefault-ed list may remain, but it holds nothing).
        self.assertFalse(st.session_state.get("_chat_messages"))
        self.assertEqual(
            st.session_state["_chat_last_error"], "no model picked",
        )
        self.assertNotIn("_chat_bg_task", st.session_state)

    def test_nothing_staged_is_a_noop(self):
        with patch.object(chat, "_build_stream") as build:
            ret = chat._send_staged_quick_prompt(self.doc)
        self.assertIsNone(ret)
        build.assert_not_called()
        self.assertNotIn("_chat_messages", st.session_state)

    def test_whitespace_only_prompt_is_consumed_without_sending(self):
        self._stage("   \n  ")
        with patch.object(chat, "_build_stream") as build:
            ret = chat._send_staged_quick_prompt(self.doc)
        self.assertIsNone(ret)
        build.assert_not_called()
        self.assertNotIn(chat._staged_quick_prompt_key(), st.session_state)


class QuickPromptCleanupTests(unittest.TestCase):
    """A staged prompt must never outlive its document or chat session —
    otherwise it would fire into another conversation's next turn."""

    def setUp(self):
        _clear_keys()

    def tearDown(self):
        _clear_keys()

    def test_remove_document_sweeps_its_staged_prompt(self):
        docs.add_document("a.md")
        st.session_state["_reader_quick_prompt__doc__a.md"] = "p"
        docs.remove_document("a.md")
        self.assertNotIn("_reader_quick_prompt__doc__a.md", st.session_state)

    def test_remove_last_document_pops_the_legacy_key(self):
        docs.add_document("a.md")
        st.session_state["_reader_quick_prompt"] = "p"
        docs.remove_document("a.md")
        self.assertNotIn("_reader_quick_prompt", st.session_state)

    def test_remove_chat_sweeps_its_staged_prompt(self):
        docs.add_document("a.md")
        docs.add_chat("a.md")  # sessions [1, 2], active 2
        st.session_state["_reader_quick_prompt__chat__2__doc__a.md"] = "p"
        docs.remove_chat(2, "a.md")
        self.assertNotIn(
            "_reader_quick_prompt__chat__2__doc__a.md", st.session_state,
        )

    def test_close_reader_clears_the_staged_prompt_in_single_doc_mode(self):
        st.session_state["_reader_quick_prompt"] = "p"
        reader._close_reader()
        self.assertNotIn("_reader_quick_prompt", st.session_state)

    def test_staged_prompt_is_scoped_per_document(self):
        docs.add_document("a.md")
        key_a = reader._reader_quick_prompt_key()
        docs.add_document("b.md")
        key_b = reader._reader_quick_prompt_key()
        self.assertNotEqual(key_a, key_b)


if __name__ == "__main__":
    unittest.main()
