"""Tests for the chat panel's ⏹ Stop-reply button and ✕ per-message rewind.

Ported from the archived NiceGUI line's tests/test_chat_stop_delete.py and
adapted to the Streamlit line's session_state plumbing.

Stop is cooperative plus eager: the ⏹ button sets the shared task dict's
``stop`` flag AND kills the agent subprocess behind the stream (OpenCode/
Cline/ZCode register their Popen under the task's ``stop_key`` in
``md_llm.llm``), the background worker breaks out at its next chunk — or the
kill unblocks a wedged read into an EOF, which must not read as an error —
closes the stream, and keeps the partial text; the finalizer folds the
partial in as the reply, or records no reply at all when nothing had
arrived. Delete (the ✕ under each history bubble) rewinds the ACTIVE
session's conversation to just before the clicked message.
"""

import shutil
import tempfile
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


def _task(**extra):
    """A fresh worker task dict with the given overrides."""
    task = {
        "text": "", "done": False, "error": None, "source": "t",
        "stop": False, "stopped": False, "stop_key": "tok-1",
    }
    task.update(extra)
    return task


class StopReplyTests(unittest.TestCase):
    """chat._stop_streaming_reply: flag the ACTIVE session's running task."""

    def setUp(self):
        _clear_keys()
        self.tmp = tempfile.mkdtemp(prefix="mdllm_stop_")
        _reset_for_tests(Core(
            base_dir=self.tmp,
            markdown_dirs=(self.tmp,),
            chat_save_dir=self.tmp,
        ))

    def tearDown(self):
        _clear_keys()
        _reset_for_tests()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stop_flags_the_running_task_and_kills_the_agent_proc(self):
        running = _task(source="LLM chat (OpenRouter · m)", stop_key="tok-9")
        st.session_state["_chat_bg_task"] = running
        with patch.object(chat.llm, "kill_agent_proc") as kill:
            self.assertTrue(chat._stop_streaming_reply())
        self.assertTrue(running["stop"])
        kill.assert_called_once_with("tok-9")

    def test_stop_without_a_running_task_is_a_noop(self):
        # No task at all…
        self.assertFalse(chat._stop_streaming_reply())
        # …a finished task…
        st.session_state["_chat_bg_task"] = _task(text="done", done=True)
        self.assertFalse(chat._stop_streaming_reply())
        self.assertFalse(st.session_state["_chat_bg_task"]["stop"])
        # …and a stop that was already requested (idempotent button).
        st.session_state["_chat_bg_task"] = _task(stop=True)
        self.assertFalse(chat._stop_streaming_reply())

    def test_stop_targets_the_active_session_task(self):
        docs.add_document("notes.md")
        docs.add_chat("notes.md")  # session 2 is now the active one
        session1 = _task(source="s1")
        session2 = _task(source="s2")
        st.session_state["_chat_bg_task"] = session1  # legacy key = session 1
        k2 = chat._chat_state_key(chat._CHAT_BG_TASK)
        st.session_state[k2] = session2
        self.assertTrue(chat._stop_streaming_reply())
        self.assertTrue(session2.get("stop"))
        self.assertFalse(session1["stop"])


class StreamWorkerStopTests(unittest.TestCase):
    """chat._stream_worker honors the stop flag between chunks."""

    def test_worker_stops_on_flag_keeps_partial_and_closes_stream(self):
        # The generator plays the user too: it sets the stop flag right
        # before yielding its 6th chunk, exactly where a button click between
        # worker iterations would land.
        events = []
        task = _task()

        def gen():
            try:
                for i in range(50):
                    if i == 5:
                        task["stop"] = True
                    events.append(i)
                    yield f"part{i} "
            finally:
                events.append("closed")

        with patch.object(chat.llm, "kill_agent_proc") as kill:
            chat._stream_worker(task, gen(), {})
        self.assertTrue(task["done"])
        self.assertTrue(task["stopped"])
        self.assertIsNone(task["error"])
        # Everything streamed up to (including) the chunk in flight when the
        # stop landed is kept — and nothing after it (the worker strips the
        # trailing space, as it always has).
        self.assertEqual(
            task["text"], "".join(f"part{i} " for i in range(6)).strip(),
        )
        self.assertEqual(events, [0, 1, 2, 3, 4, 5, "closed"])
        # The subprocess behind the stream is killed too (no-op for plain
        # generators, but the worker always asks).
        kill.assert_called_once_with(task["stop_key"])

    def test_worker_without_stop_runs_to_completion_unchanged(self):
        task = _task()
        chat._stream_worker(task, iter(["a", "b", "c"]), {})
        self.assertTrue(task["done"])
        self.assertEqual(task["text"], "abc")
        self.assertIsNone(task["error"])
        self.assertFalse(task["stopped"])  # normal path leaves it unset
        self.assertFalse(task["stop"])

    def test_late_stop_on_an_exhausted_stream_is_not_marked_stopped(self):
        # A stop that lands during the final next() (after every chunk was
        # already yielded) must not mislabel a COMPLETE reply as a stopped
        # partial — only an actual early break marks the task stopped.
        task = _task()

        def gen():
            yield "a"
            yield "b"
            task["stop"] = True  # resumes-and-sets during the last next()

        chat._stream_worker(task, gen(), {})
        self.assertTrue(task["done"])
        self.assertFalse(task["stopped"])
        self.assertEqual(task["text"], "ab")

    def test_kill_induced_error_after_stop_is_suppressed(self):
        # The eager kill unblocks a wedged read as an EOF / non-zero exit,
        # which the generator surfaces as an exception (_safe_stream captures
        # it into the holder) — a user stop must keep the partial text, not
        # turn into a failure bubble. The flag is set the way the Stop button
        # does it: while the worker sits blocked in the read.
        task = _task()

        def gen():
            yield "partial "
            task["stop"] = True  # the button, mid-read
            raise RuntimeError("opencode run exited -9: killed")

        chat._stream_worker(task, gen(), {})
        self.assertTrue(task["done"])
        self.assertTrue(task["stopped"])
        self.assertIsNone(task["error"])
        self.assertEqual(task["text"], "partial")

    def test_holder_error_still_surfaces_when_unstopped(self):
        # A stream whose generator raised (captured by _safe_stream's holder)
        # still reports the error — the stop path must not swallow it.
        task = _task()
        chat._stream_worker(task, iter([]), {"error": "provider blew up"})
        self.assertEqual(task["error"], "provider blew up")
        self.assertTrue(task["done"])

    def test_close_stream_quietly_tolerates_no_close_and_raising_close(self):
        # Plain iterators (no close method) and close() raising must both be
        # safe — a cleanup failure never masks the stop.
        chat._close_stream_quietly(iter(["x"]))       # list_iterator
        chat._close_stream_quietly(iter(("x",)))      # tuple_iterator

        class Boom:
            def close(self):
                raise RuntimeError("boom")

        chat._close_stream_quietly(Boom())  # must not raise


class FinalizeStoppedTaskTests(unittest.TestCase):
    """chat._finalize_chat_task: a stopped task keeps its partial text."""

    def setUp(self):
        _clear_keys()

    def tearDown(self):
        _clear_keys()

    def test_stopped_with_text_appends_the_partial_reply(self):
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "q"},
        ]
        chat._finalize_chat_task(_task(
            text="partial answer", done=True, stopped=True,
        ))
        self.assertEqual(
            st.session_state["_chat_messages"],
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "partial answer"},
            ],
        )
        self.assertNotIn("_chat_last_error", st.session_state)

    def test_stopped_without_text_records_no_reply_at_all(self):
        # No placeholder: the call didn't fail, the user ended it. The bare
        # user turn stays, ready for a Resend.
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "q"},
        ]
        chat._finalize_chat_task(_task(text="", done=True, stopped=True))
        self.assertEqual(
            st.session_state["_chat_messages"],
            [{"role": "user", "content": "q"}],
        )
        self.assertNotIn("_chat_last_error", st.session_state)

    def test_unstopped_empty_reply_still_gets_the_placeholder(self):
        # The stop branch must not change the normal path: a completed call
        # that produced nothing still records the placeholder.
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "q"},
        ]
        chat._finalize_chat_task(_task(text="", done=True))
        self.assertEqual(
            st.session_state["_chat_messages"][-1]["content"],
            "_(empty response — nothing came back.)_",
        )

    def test_resend_after_stop_replaces_the_partial_reply(self):
        # The intended rewind flow: stop keeps a partial answer; a Resend
        # drops it and re-runs the question through the CURRENT controls.
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "partial"},
        ]
        with patch.object(
            chat, "_build_stream", return_value=(iter(["full answer"]), None),
        ):
            task = chat._resend_last_request(None)
        self.assertIsNotNone(task)
        self.assertEqual(
            st.session_state["_chat_messages"],
            [{"role": "user", "content": "q"}],
        )


class StopKeyWiringTests(unittest.TestCase):
    """Every send path must hand its task's ``stop_key`` to _build_stream.

    The agent generators register their subprocess under the ``stop_key``
    the SENDER passes in — if a path builds the stream before minting the
    task's key (or never passes it), the generator registers nothing and
    ⏹ Stop's kill_agent_proc(token) finds no process: the cooperative flag
    alone cannot unblock a wedged agent read, and the button dies. The typed
    send path shares the pattern but needs a full render_chat to exercise.
    """

    def setUp(self):
        _clear_keys()

    def tearDown(self):
        _clear_keys()

    def _capture_build(self):
        def fake_build(context_path, holder, stop_key=None):
            self.captured_stop_key = stop_key
            return iter(["reply"]), None

        return fake_build

    def test_resend_passes_the_task_stop_key_to_the_stream_builder(self):
        st.session_state["_chat_messages"] = [{"role": "user", "content": "q"}]
        with patch.object(
            chat, "_build_stream", side_effect=self._capture_build()
        ):
            task = chat._resend_last_request(None)
        self.assertIsNotNone(task)
        self.assertTrue(self.captured_stop_key)
        self.assertEqual(task["stop_key"], self.captured_stop_key)

    def test_staged_prompt_passes_the_task_stop_key_to_the_stream_builder(self):
        st.session_state[chat._staged_quick_prompt_key()] = "go"
        with patch.object(
            chat, "_build_stream", side_effect=self._capture_build()
        ):
            task = chat._send_staged_quick_prompt(None)
        self.assertIsNotNone(task)
        self.assertTrue(self.captured_stop_key)
        self.assertEqual(task["stop_key"], self.captured_stop_key)

    def test_build_stream_forwards_stop_key_to_the_agent_generator(self):
        # The last link of the chain: _build_stream must hand the stop_key to
        # the agent generator, whose subprocess is registered under it.
        st.session_state["chat_llm_provider"] = "OpenCode"
        st.session_state["chat_llm_opencode_model_sel"] = "m"
        with patch.object(
            chat, "_send_context_and_turns", return_value=[]
        ), patch.object(
            chat, "_session_sandbox_dir", return_value="/tmp/sb"
        ), patch.object(
            chat, "_remember_opencode_model"
        ), patch.object(
            chat.llm, "opencode_chat_stream", return_value=iter(["x"])
        ) as m_oc:
            gen, err = chat._build_stream(None, {}, stop_key="tok")
        self.assertIsNone(err)
        self.assertEqual(m_oc.call_args.kwargs.get("stop_key"), "tok")


class DeleteMessageTests(unittest.TestCase):
    """chat._delete_messages_from: rewind to just before the clicked message."""

    def setUp(self):
        _clear_keys()
        self.tmp = tempfile.mkdtemp(prefix="mdllm_delete_")
        _reset_for_tests(Core(
            base_dir=self.tmp,
            markdown_dirs=(self.tmp,),
            chat_save_dir=self.tmp,
        ))
        st.session_state["_chat_messages"] = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]

    def tearDown(self):
        _clear_keys()
        _reset_for_tests()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_delete_mid_conversation_drops_it_and_everything_after(self):
        chat._delete_messages_from(2)
        self.assertEqual(
            st.session_state["_chat_messages"],
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        )

    def test_delete_last_message_drops_only_it(self):
        chat._delete_messages_from(3)
        self.assertEqual(len(st.session_state["_chat_messages"]), 3)

    def test_delete_first_message_pops_the_now_empty_conversation(self):
        chat._delete_messages_from(0)
        # Matches what "Clear conversation" leaves: no key at all.
        self.assertNotIn("_chat_messages", st.session_state)

    def test_out_of_range_and_missing_conversation_are_noops(self):
        chat._delete_messages_from(99)
        chat._delete_messages_from(-1)
        self.assertEqual(len(st.session_state["_chat_messages"]), 4)
        st.session_state.pop("_chat_messages", None)
        chat._delete_messages_from(0)  # missing conversation: no raise

    def test_delete_targets_the_active_session_conversation(self):
        docs.add_document("notes.md")
        docs.add_chat("notes.md")  # session 2 is now the active one
        st.session_state["_chat_messages"] = [  # session 1's conversation
            {"role": "user", "content": "session 1 question"},
        ]
        k2 = chat._chat_state_key(chat._CHAT_MESSAGES)
        st.session_state[k2] = [
            {"role": "user", "content": "s2 q1"},
            {"role": "assistant", "content": "s2 a1"},
            {"role": "user", "content": "s2 q2"},
        ]
        chat._delete_messages_from(1)
        # Only the ACTIVE session's conversation was rewound…
        self.assertEqual(
            st.session_state[k2],
            [{"role": "user", "content": "s2 q1"}],
        )
        # …and session 1's was untouched.
        self.assertEqual(
            st.session_state["_chat_messages"],
            [{"role": "user", "content": "session 1 question"}],
        )


class StopAndDeleteButtonKeyTests(unittest.TestCase):
    """The buttons' widget keys must stay out of the control snapshot."""

    def setUp(self):
        _clear_keys()

    def tearDown(self):
        _clear_keys()

    def test_button_keys_are_outside_the_snapshot_prefixes(self):
        # Regression guard (mirrors ResendButtonKeyTests): a button key
        # captured by _chat_control_keys would be re-injected by
        # _restore_chat_controls before mount, and Streamlit refuses
        # session_state writes to button keys.
        for key in ("_chat_stop_reply", "_chat_del_0"):
            self.assertTrue(key.startswith("_"))
            self.assertFalse(key.startswith("chat_"))
            self.assertFalse(key.startswith("_chat_ssh_"))
            st.session_state[key] = True
            self.assertNotIn(key, set(chat._chat_control_keys()))


if __name__ == "__main__":
    unittest.main()
