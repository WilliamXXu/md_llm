"""Tests for the LLM chat panel's Resend-request button (``md_llm.chat``).

Resend re-runs the conversation's latest user turn through the exact send
pipeline: a trailing assistant reply is dropped first (so the new reply
replaces the old outcome instead of duplicating the Q/A pair) and put back if
the resend fails validation — an aborted retry must never lose the previous
answer. These exercise the session_state plumbing directly; the Streamlit
widget rendering is covered by the AppTest harness.
"""

import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import streamlit as st

from md_llm import chat, docs
from md_llm.core import Core, _reset_for_tests


def _clear_keys():
    """Drop every key the chat panel and the docs registry can touch."""
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


class ResendRequestTests(unittest.TestCase):
    """chat._resend_last_request: resend minus the append — trim, build,
    background worker, per-session keys."""

    def setUp(self):
        _clear_keys()
        self.tmp = tempfile.mkdtemp(prefix="mdllm_resend_")
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

    def test_resend_replaces_the_trailing_reply_without_appending(self):
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ]
        with patch.object(
            chat, "_build_stream",
            return_value=(iter(["new answer"]), None),
        ):
            task = chat._resend_last_request(None)

        self.assertIsNotNone(task)
        # The trailing reply is dropped and NO duplicate user turn is
        # appended: the resent request replaces the old outcome.
        self.assertEqual(
            st.session_state["_chat_messages"],
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
            ],
        )
        # The task is live under the session's bg-task key and streams.
        self.assertIs(st.session_state["_chat_bg_task"], task)
        self.assertTrue(_wait_done(task))
        self.assertEqual(task["text"], "new answer")
        self.assertIsNone(task["error"])

    def test_resend_also_drops_the_empty_response_placeholder(self):
        # The placeholder a zero-content stream leaves behind is an outcome
        # too: resending must replace it, not pile a second one on.
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "question"},
            {"role": "assistant",
             "content": "_(empty response — nothing came back.)_"},
        ]
        with patch.object(
            chat, "_build_stream", return_value=(iter(["real answer"]), None),
        ):
            task = chat._resend_last_request(None)
        self.assertTrue(_wait_done(task))
        self.assertEqual(
            st.session_state["_chat_messages"],
            [{"role": "user", "content": "question"}],
        )

    def test_resend_after_a_failed_call_sends_the_dangling_user_turn(self):
        # A failed stream leaves the request in the history with no reply:
        # resend re-runs it verbatim, trimming nothing.
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "the question that failed"},
        ]
        with patch.object(
            chat, "_build_stream", return_value=(iter(["late reply"]), None),
        ):
            task = chat._resend_last_request(None)
        self.assertTrue(_wait_done(task))
        self.assertEqual(task["text"], "late reply")
        self.assertEqual(
            st.session_state["_chat_messages"],
            [{"role": "user", "content": "the question that failed"}],
        )

    def test_validation_failure_restores_the_dropped_reply(self):
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        with patch.object(
            chat, "_build_stream", return_value=(None, "no api key"),
        ):
            ret = chat._resend_last_request(None)
        self.assertIsNone(ret)
        # The aborted retry must not lose the previous answer…
        self.assertEqual(
            st.session_state["_chat_messages"],
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "previous answer"},
            ],
        )
        # …and the failure surfaces as the transient error bubble.
        self.assertEqual(st.session_state["_chat_last_error"], "no api key")
        self.assertNotIn("_chat_bg_task", st.session_state)

    def test_empty_conversation_warns_without_sending(self):
        with patch.object(chat, "_build_stream") as build:
            ret = chat._resend_last_request(None)
        self.assertIsNone(ret)
        build.assert_not_called()
        self.assertNotIn("_chat_messages", st.session_state)
        self.assertNotIn("_chat_last_error", st.session_state)
        self.assertNotIn("_chat_bg_task", st.session_state)

    def test_conversation_without_user_turns_cannot_resend(self):
        st.session_state["_chat_messages"] = [
            {"role": "assistant", "content": "only an answer"},
        ]
        with patch.object(chat, "_build_stream") as build:
            ret = chat._resend_last_request(None)
        self.assertIsNone(ret)
        build.assert_not_called()
        self.assertNotIn("_chat_bg_task", st.session_state)

    def test_resend_targets_the_active_session_keys(self):
        docs.add_document("notes.md")
        docs.add_chat("notes.md")  # session 2 is now the active one
        st.session_state["_chat_messages"] = [  # session 1's conversation
            {"role": "user", "content": "session 1 question"},
        ]
        k2 = chat._chat_state_key(chat._CHAT_MESSAGES)
        self.assertEqual(k2, "_chat_messages__chat__2__doc__notes.md")
        st.session_state[k2] = [
            {"role": "user", "content": "session 2 question"},
            {"role": "assistant", "content": "session 2 answer"},
        ]
        with patch.object(
            chat, "_build_stream", return_value=(iter(["ok"]), None),
        ):
            task = chat._resend_last_request(None)
        self.assertIsNotNone(task)
        self.assertIs(
            st.session_state["_chat_bg_task__chat__2__doc__notes.md"], task,
        )
        # Only the ACTIVE session's conversation was trimmed.
        self.assertEqual(
            st.session_state[k2],
            [{"role": "user", "content": "session 2 question"}],
        )
        self.assertEqual(
            st.session_state["_chat_messages"],
            [{"role": "user", "content": "session 1 question"}],
        )

    def test_has_last_request_needs_a_user_turn(self):
        self.assertFalse(chat._has_last_request())
        st.session_state["_chat_messages"] = [
            {"role": "assistant", "content": "only an answer"},
        ]
        self.assertFalse(chat._has_last_request())
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "question"},
        ]
        self.assertTrue(chat._has_last_request())


class ResendButtonKeyTests(unittest.TestCase):
    """The button's widget key must stay out of the control snapshot."""

    def setUp(self):
        _clear_keys()

    def tearDown(self):
        _clear_keys()

    def test_button_key_is_outside_the_snapshot_prefixes(self):
        # Regression guard (mirrors test_chat_controls_state.py): a button key
        # captured by _chat_control_keys would be re-injected by
        # _restore_chat_controls before mount, and Streamlit refuses
        # session_state writes to button keys.
        self.assertTrue(chat._RESEND_BUTTON_KEY.startswith("_"))
        self.assertFalse(chat._RESEND_BUTTON_KEY.startswith("chat_"))
        self.assertFalse(chat._RESEND_BUTTON_KEY.startswith("_chat_ssh_"))
        st.session_state[chat._RESEND_BUTTON_KEY] = True
        self.assertNotIn(
            chat._RESEND_BUTTON_KEY, set(chat._chat_control_keys()),
        )


if __name__ == "__main__":
    unittest.main()
