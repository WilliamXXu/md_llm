"""Tests for the optional multi-document session (``md_llm.docs``).

By default md_llm is single-document (legacy bare session keys). Opening files
with ``keep_open=True`` (or :func:`docs.add_document`) starts multi-document
mode: each document keeps its own namespaced chat state under
``<base>__doc__<relpath>``, the active document drives the Reader, and closing
the last document returns to single-document mode. These exercise the registry
and the key-namespacing logic directly.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import streamlit as st

from md_llm import chat, docs
from md_llm.core import Core, _reset_for_tests


def _clear_doc_state():
    """Drop every key the docs registry and its scoped state can touch."""
    for k in list(st.session_state.keys()):
        if not isinstance(k, str):
            continue
        if (
            k.startswith("_md_llm_")
            or k.startswith("_chat_")
            or k.startswith("_reader_")
            or k.startswith("_app_")
        ):
            st.session_state.pop(k, None)


class DocKeyTests(unittest.TestCase):
    def test_no_doc_uses_the_legacy_key(self):
        self.assertEqual(docs.doc_key("_chat_messages", None), "_chat_messages")
        self.assertEqual(docs.doc_key("_chat_messages", ""), "_chat_messages")

    def test_doc_namespaces_the_key(self):
        self.assertEqual(
            docs.doc_key("_chat_messages", "notes.md"),
            "_chat_messages__doc__notes.md",
        )

    def test_two_docs_get_distinct_keys(self):
        self.assertNotEqual(
            docs.doc_key("_chat_messages", "a.md"),
            docs.doc_key("_chat_messages", "b.md"),
        )


class RegistryTests(unittest.TestCase):
    def setUp(self):
        _clear_doc_state()

    def tearDown(self):
        _clear_doc_state()

    def test_add_document_registers_and_activates(self):
        docs.add_document("a.md")
        self.assertEqual(docs.open_documents(), ["a.md"])
        self.assertEqual(docs.active_document(), "a.md")
        # The Reader target follows the active document.
        self.assertEqual(st.session_state["_reader_target"], "a.md")
        self.assertTrue(docs.is_multi())

    def test_add_second_document_keeps_the_first(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        self.assertEqual(docs.open_documents(), ["a.md", "b.md"])
        self.assertEqual(docs.active_document(), "b.md")

    def test_add_document_is_idempotent(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        docs.add_document("a.md")
        self.assertEqual(docs.open_documents(), ["a.md", "b.md"])
        self.assertEqual(docs.active_document(), "a.md")

    def test_set_active_switches_reader_target(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        docs.set_active_document("a.md")
        self.assertEqual(docs.active_document(), "a.md")
        self.assertEqual(st.session_state["_reader_target"], "a.md")

    def test_remove_document_falls_back_to_the_next(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        docs.remove_document("b.md")
        self.assertEqual(docs.open_documents(), ["a.md"])
        self.assertEqual(docs.active_document(), "a.md")
        self.assertEqual(st.session_state["_reader_target"], "a.md")

    def test_removing_a_non_active_doc_keeps_the_active(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        docs.remove_document("a.md")
        self.assertEqual(docs.active_document(), "b.md")

    def test_remove_last_document_returns_to_single_doc_mode(self):
        docs.add_document("a.md")
        st.session_state["_chat_messages__doc__a.md"] = [
            {"role": "user", "content": "hi"},
        ]
        st.session_state["_reader_target"] = "a.md"
        docs.remove_document("a.md")
        self.assertEqual(docs.open_documents(), [])
        self.assertFalse(docs.is_multi())
        self.assertIsNone(docs.active_document())
        self.assertNotIn("_reader_target", st.session_state)

    def test_remove_document_drops_its_namespaced_state(self):
        docs.add_document("a.md")
        st.session_state["_chat_messages__doc__a.md"] = [
            {"role": "user", "content": "hi"},
        ]
        st.session_state["_chat_bg_task__doc__a.md"] = {"done": True}
        st.session_state["_reader_quick_prompt__doc__a.md"] = "summarize"
        docs.remove_document("a.md")
        self.assertNotIn("_chat_messages__doc__a.md", st.session_state)
        self.assertNotIn("_chat_bg_task__doc__a.md", st.session_state)
        self.assertNotIn("_reader_quick_prompt__doc__a.md", st.session_state)

    def test_remove_document_keeps_other_docs_state(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        st.session_state["_chat_messages__doc__a.md"] = [{"role": "user"}]
        st.session_state["_chat_messages__doc__b.md"] = [{"role": "user"}]
        docs.remove_document("a.md")
        self.assertNotIn("_chat_messages__doc__a.md", st.session_state)
        self.assertIn("_chat_messages__doc__b.md", st.session_state)

    def test_reset_documents_leaves_the_reader_target(self):
        # open_in_reader() without keep_open sets the target itself and calls
        # reset_documents(); the registry must go but the target must stay.
        docs.add_document("a.md")
        st.session_state["_reader_target"] = "a.md"
        docs.reset_documents()
        self.assertFalse(docs.is_multi())
        self.assertIsNone(docs.active_document())
        self.assertEqual(st.session_state["_reader_target"], "a.md")

    def test_active_document_self_heals_to_the_first_open(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        st.session_state[docs._ACTIVE_DOC] = "ghost.md"
        self.assertEqual(docs.active_document(), "a.md")
        self.assertEqual(st.session_state[docs._ACTIVE_DOC], "a.md")


class DuplicateDocumentTests(unittest.TestCase):
    """add_document() refuses a SECOND copy of an already-open file — whether
    via a differently-spelled relpath, an absolute path, or a symlink — and
    warns via the _warn_already_open dialog while activating the existing
    document. The exact-same relpath stays the documented idempotent re-open
    (no warning)."""

    def setUp(self):
        _clear_doc_state()
        self._tmp = tempfile.mkdtemp(prefix="mdllm_dup_")
        with open(os.path.join(self._tmp, "a.md"), "w") as f:
            f.write("# a")
        with open(os.path.join(self._tmp, "b.md"), "w") as f:
            f.write("# b")
        _reset_for_tests(Core(
            base_dir=self._tmp,
            markdown_dirs=(self._tmp,),
            chat_save_dir=self._tmp,
        ))

    def tearDown(self):
        _clear_doc_state()
        _reset_for_tests()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_reopen_same_string_is_idempotent_without_warning(self):
        with patch("md_llm.docs._warn_already_open") as warn:
            docs.add_document("a.md")
            docs.add_document("a.md")
        warn.assert_not_called()
        self.assertEqual(docs.open_documents(), ["a.md"])
        self.assertEqual(docs.active_document(), "a.md")

    def test_same_file_via_dot_slash_is_refused_with_warning(self):
        docs.add_document("a.md")
        with patch("md_llm.docs._warn_already_open") as warn:
            ret = docs.add_document("./a.md")
        warn.assert_called_once_with("a.md")
        self.assertEqual(ret, "a.md")
        self.assertEqual(docs.open_documents(), ["a.md"])
        self.assertEqual(docs.active_document(), "a.md")
        self.assertEqual(st.session_state["_reader_target"], "a.md")

    def test_same_file_via_absolute_path_is_refused(self):
        docs.add_document("a.md")
        with patch("md_llm.docs._warn_already_open") as warn:
            docs.add_document(os.path.join(self._tmp, "a.md"))
        warn.assert_called_once_with("a.md")
        self.assertEqual(docs.open_documents(), ["a.md"])

    def test_same_file_via_symlink_is_refused(self):
        link = os.path.join(self._tmp, "link.md")
        try:
            os.symlink(os.path.join(self._tmp, "a.md"), link)
        except OSError:
            self.skipTest("symlinks unavailable on this platform")
        docs.add_document("a.md")
        with patch("md_llm.docs._warn_already_open") as warn:
            docs.add_document("link.md")
        warn.assert_called_once_with("a.md")
        self.assertEqual(docs.open_documents(), ["a.md"])

    def test_distinct_files_both_open_without_warning(self):
        with patch("md_llm.docs._warn_already_open") as warn:
            docs.add_document("a.md")
            docs.add_document("b.md")
        warn.assert_not_called()
        self.assertEqual(docs.open_documents(), ["a.md", "b.md"])

    def test_refused_copy_never_touches_a_second_conversation(self):
        docs.add_document("a.md")
        st.session_state["_chat_messages__doc__a.md"] = [
            {"role": "user", "content": "hi"},
        ]
        with patch("md_llm.docs._warn_already_open"):
            docs.add_document("./a.md")
        self.assertNotIn("_chat_messages__doc__./a.md", st.session_state)
        self.assertEqual(
            st.session_state["_chat_messages__doc__a.md"][0]["content"], "hi",
        )

    def test_missing_file_deduped_by_normalized_path(self):
        with patch("md_llm.docs._warn_already_open") as warn:
            docs.add_document("ghost.md")
            docs.add_document("./ghost.md")
        warn.assert_called_once_with("ghost.md")
        self.assertEqual(docs.open_documents(), ["ghost.md"])

    def test_string_normalization_fallback_without_core(self):
        # Without an injected Core the identity falls back to normpath of the
        # raw string — the duplicate guard still holds for spelling variants.
        _reset_for_tests()
        with patch("md_llm.docs._warn_already_open") as warn:
            docs.add_document("a.md")
            docs.add_document("./a.md")
        warn.assert_called_once_with("a.md")
        self.assertEqual(docs.open_documents(), ["a.md"])


class ChatSessionKeyTests(unittest.TestCase):
    """chat_key(): session-1 keys are the legacy doc keys; 2+ are namespaced."""

    def test_session_one_uses_legacy_keys(self):
        self.assertEqual(docs.chat_key("_chat_messages", 1, None), "_chat_messages")
        self.assertEqual(docs.chat_key("_chat_messages", None, None), "_chat_messages")
        self.assertEqual(
            docs.chat_key("_chat_messages", 1, "a.md"),
            "_chat_messages__doc__a.md",
        )

    def test_extra_sessions_get_their_own_keys(self):
        self.assertEqual(
            docs.chat_key("_chat_messages", 2, "a.md"),
            "_chat_messages__chat__2__doc__a.md",
        )
        self.assertEqual(
            docs.chat_key("_chat_messages", 2, None),
            "_chat_messages__chat__2",
        )

    def test_session_keys_never_collide(self):
        self.assertNotEqual(
            docs.chat_key("_chat_messages", 1, "a.md"),
            docs.chat_key("_chat_messages", 2, "a.md"),
        )
        self.assertNotEqual(
            docs.chat_key("_chat_messages", 1, "a.md"),
            docs.chat_key("_chat_messages", 1, "b.md"),
        )

    def test_session_keys_still_end_with_the_doc_suffix(self):
        # Closing a document (suffix __doc__<rel>) must sweep its sessions too.
        self.assertTrue(
            docs.chat_key("_chat_messages", 2, "a.md").endswith("__doc__a.md")
        )


class ChatSessionRegistryTests(unittest.TestCase):
    def setUp(self):
        _clear_doc_state()

    def tearDown(self):
        _clear_doc_state()

    def test_default_is_a_single_session(self):
        self.assertEqual(docs.chat_sessions("a.md"), [1])
        self.assertEqual(docs.active_chat("a.md"), 1)
        self.assertEqual(docs.chat_session_label("a.md", 1), "Chat 1")

    def test_add_chat_registers_and_activates(self):
        sid = docs.add_chat("a.md")
        self.assertEqual(sid, 2)
        self.assertEqual(docs.chat_sessions("a.md"), [1, 2])
        self.assertEqual(docs.active_chat("a.md"), 2)
        self.assertEqual(docs.chat_session_label("a.md", 2), "Chat 2")

    def test_add_chat_label_override(self):
        # The Reader's ⚡ Summarize quick action opens each of its sessions
        # as "Summary"; the default numbering is untouched for "+ New".
        docs.add_chat("a.md", label="Summary")
        self.assertEqual(docs.chat_session_label("a.md", 2), "Summary")
        docs.add_chat("a.md")
        self.assertEqual(docs.chat_session_label("a.md", 3), "Chat 3")
        docs.add_chat("a.md", label="Summary")
        self.assertEqual(docs.chat_session_label("a.md", 4), "Summary")
        self.assertEqual(docs.chat_sessions("a.md"), [1, 2, 3, 4])

    def test_sessions_are_per_document(self):
        docs.add_chat("a.md")
        self.assertEqual(docs.chat_sessions("a.md"), [1, 2])
        self.assertEqual(docs.chat_sessions("b.md"), [1])

    def test_set_active_switches_without_creating(self):
        docs.add_chat("a.md")
        docs.set_active_chat(1, "a.md")
        self.assertEqual(docs.active_chat("a.md"), 1)

    def test_active_chat_self_heals_to_the_first(self):
        docs.add_chat("a.md")
        st.session_state[docs._chat_active_key("a.md")] = 99
        self.assertEqual(docs.active_chat("a.md"), 1)
        self.assertEqual(st.session_state[docs._chat_active_key("a.md")], 1)

    def test_remove_chat_drops_its_state_and_activates_fallback(self):
        docs.add_chat("a.md")
        st.session_state["_chat_messages__chat__2__doc__a.md"] = [{"role": "user"}]
        st.session_state["_chat_bg_task__chat__2__doc__a.md"] = {"done": True}
        docs.remove_chat(2, "a.md")
        self.assertEqual(docs.chat_sessions("a.md"), [1])
        self.assertEqual(docs.active_chat("a.md"), 1)
        self.assertNotIn("_chat_messages__chat__2__doc__a.md", st.session_state)
        self.assertNotIn("_chat_bg_task__chat__2__doc__a.md", st.session_state)

    def test_remove_chat_keeps_other_sessions_state(self):
        docs.add_chat("a.md")
        docs.add_chat("a.md")  # sessions [1, 2, 3], active 3
        st.session_state["_chat_messages__doc__a.md"] = [{"role": "user"}]
        st.session_state["_chat_messages__chat__2__doc__a.md"] = [{"role": "user"}]
        st.session_state["_chat_messages__chat__3__doc__a.md"] = [{"role": "user"}]
        docs.remove_chat(2, "a.md")
        self.assertIn("_chat_messages__doc__a.md", st.session_state)
        self.assertIn("_chat_messages__chat__3__doc__a.md", st.session_state)
        self.assertNotIn("_chat_messages__chat__2__doc__a.md", st.session_state)
        self.assertEqual(docs.active_chat("a.md"), 3)  # non-active close: no switch

    def test_remove_chat_never_removes_the_last_session(self):
        docs.add_chat("a.md")
        docs.remove_chat(2, "a.md")
        docs.remove_chat(1, "a.md")  # last one — no-op
        self.assertEqual(docs.chat_sessions("a.md"), [1])
        self.assertEqual(docs.active_chat("a.md"), 1)

    def test_remove_chat_unknown_session_is_a_noop(self):
        self.assertEqual(docs.chat_sessions("a.md"), [1])
        docs.remove_chat(99, "a.md")
        self.assertEqual(docs.chat_sessions("a.md"), [1])

    def test_remove_session_one_drops_legacy_keys_but_not_other_sessions(self):
        docs.add_chat("a.md")  # sessions [1, 2]
        st.session_state["_chat_messages__doc__a.md"] = [{"role": "user"}]
        st.session_state["_reader_quick_prompt__doc__a.md"] = "summarize"
        st.session_state["_chat_messages__chat__2__doc__a.md"] = [{"role": "user"}]
        docs.remove_chat(1, "a.md")
        self.assertNotIn("_chat_messages__doc__a.md", st.session_state)
        self.assertNotIn("_reader_quick_prompt__doc__a.md", st.session_state)
        self.assertIn("_chat_messages__chat__2__doc__a.md", st.session_state)
        self.assertEqual(docs.chat_sessions("a.md"), [2])
        self.assertEqual(docs.active_chat("a.md"), 2)

    def test_remove_chat_keeps_the_registry_for_the_remaining_session(self):
        docs.add_chat("a.md")
        docs.remove_chat(2, "a.md")
        self.assertIn(docs._chat_sessions_key("a.md"), st.session_state)

    def test_remove_document_sweeps_its_session_registry(self):
        docs.add_document("a.md")
        docs.add_chat("a.md")
        st.session_state["_chat_messages__chat__2__doc__a.md"] = [{"role": "user"}]
        docs.remove_document("a.md")
        self.assertNotIn(docs._chat_sessions_key("a.md"), st.session_state)
        self.assertNotIn(docs._chat_active_key("a.md"), st.session_state)
        self.assertNotIn("_chat_messages__chat__2__doc__a.md", st.session_state)


class CloseGuardTests(unittest.TestCase):
    def setUp(self):
        _clear_doc_state()

    def tearDown(self):
        _clear_doc_state()

    def test_empty_chat_has_no_messages(self):
        docs.add_document("a.md")
        self.assertFalse(docs.doc_chat_has_messages("a.md"))

    def test_session_one_messages_count(self):
        docs.add_document("a.md")
        st.session_state["_chat_messages__doc__a.md"] = [{"role": "user"}]
        self.assertTrue(docs.doc_chat_has_messages("a.md"))

    def test_extra_session_messages_count(self):
        docs.add_document("a.md")
        docs.add_chat("a.md")
        st.session_state["_chat_messages__chat__2__doc__a.md"] = [{"role": "user"}]
        self.assertTrue(docs.doc_chat_has_messages("a.md"))

    def test_other_docs_messages_do_not_count(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        st.session_state["_chat_messages__doc__b.md"] = [{"role": "user"}]
        self.assertFalse(docs.doc_chat_has_messages("a.md"))

    def test_close_with_empty_chat_removes_immediately(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        with (
            patch("md_llm.docs.st.rerun") as rerun,
            patch("md_llm.docs._confirm_close_document") as confirm,
        ):
            docs.close_document("a.md")
        confirm.assert_not_called()
        self.assertEqual(docs.open_documents(), ["b.md"])
        rerun.assert_called_once()

    def test_close_with_non_empty_chat_confirms_first(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        st.session_state["_chat_messages__doc__a.md"] = [{"role": "user"}]
        with (
            patch("md_llm.docs.st.rerun") as rerun,
            patch("md_llm.docs._confirm_close_document") as confirm,
        ):
            docs.close_document("a.md")
        confirm.assert_called_once_with("a.md")
        rerun.assert_not_called()
        self.assertEqual(docs.open_documents(), ["a.md", "b.md"])


class ChatScopingTests(unittest.TestCase):
    def setUp(self):
        _clear_doc_state()

    def tearDown(self):
        _clear_doc_state()

    def test_chat_state_key_is_scoped_to_the_active_doc(self):
        docs.add_document("notes.md")
        self.assertEqual(
            chat._chat_state_key("_chat_messages"),
            "_chat_messages__doc__notes.md",
        )

    def test_chat_state_key_uses_legacy_key_in_single_doc_mode(self):
        self.assertEqual(chat._chat_state_key("_chat_messages"), "_chat_messages")
        self.assertEqual(chat._chat_state_key("_chat_bg_task"), "_chat_bg_task")

    def test_each_doc_gets_its_own_messages_list(self):
        docs.add_document("a.md")
        st.session_state[chat._chat_state_key("_chat_messages")] = [
            {"role": "user", "content": "about a"},
        ]
        docs.add_document("b.md")
        st.session_state[chat._chat_state_key("_chat_messages")] = [
            {"role": "user", "content": "about b"},
        ]
        docs.set_active_document("a.md")
        self.assertEqual(
            st.session_state[chat._chat_state_key("_chat_messages")][0]["content"],
            "about a",
        )
        docs.set_active_document("b.md")
        self.assertEqual(
            st.session_state[chat._chat_state_key("_chat_messages")][0]["content"],
            "about b",
        )

    def test_chat_state_key_follows_the_active_session(self):
        docs.add_document("notes.md")
        docs.add_chat("notes.md")
        self.assertEqual(
            chat._chat_state_key("_chat_messages"),
            "_chat_messages__chat__2__doc__notes.md",
        )
        docs.set_active_chat(1, "notes.md")
        self.assertEqual(
            chat._chat_state_key("_chat_messages"),
            "_chat_messages__doc__notes.md",
        )

    def test_each_session_gets_its_own_messages_list(self):
        docs.add_document("notes.md")
        docs.add_chat("notes.md")
        st.session_state[chat._chat_state_key("_chat_messages")] = [
            {"role": "user", "content": "session 2 question"},
        ]
        docs.set_active_chat(1, "notes.md")
        st.session_state[chat._chat_state_key("_chat_messages")] = [
            {"role": "user", "content": "session 1 question"},
        ]
        self.assertEqual(
            st.session_state[chat._chat_state_key("_chat_messages")][0]["content"],
            "session 1 question",
        )
        docs.set_active_chat(2, "notes.md")
        self.assertEqual(
            st.session_state[chat._chat_state_key("_chat_messages")][0]["content"],
            "session 2 question",
        )


class RenderDocButtonsTests(unittest.TestCase):
    """render_doc_buttons(): a permanent "(no document)" placeholder (with a
    disabled Close) is ALWAYS rendered first, followed by one switch + Close
    row per open document."""

    def setUp(self):
        _clear_doc_state()
        _reset_for_tests(Core(
            base_dir="/tmp/mdllm_test",
            markdown_dirs=("/tmp/mdllm_test",),
            chat_save_dir="/tmp/mdllm_test",
        ))

    def tearDown(self):
        _clear_doc_state()
        _reset_for_tests()

    def test_placeholder_always_first_when_no_document_is_open(self):
        col_switch = MagicMock()
        col_switch.button.return_value = False
        col_close = MagicMock()
        with patch("md_llm.docs.st.columns", return_value=[col_switch, col_close]):
            docs.render_doc_buttons()
        # exactly one row — the placeholder — labelled "(no document, direct LLM chat)"
        col_switch.button.assert_called_once()
        self.assertEqual(col_switch.button.call_args.args[0], "(no document, direct LLM chat)")
        self.assertEqual(
            col_switch.button.call_args.kwargs["type"], "primary",
        )
        col_close.button.assert_called_once()
        self.assertTrue(col_close.button.call_args.kwargs["disabled"])

    def test_placeholder_click_jumps_to_chat(self):
        col_switch = MagicMock()
        col_switch.button.return_value = True  # placeholder clicked
        col_close = MagicMock()
        with (
            patch("md_llm.docs.st.columns", return_value=[col_switch, col_close]),
            patch("md_llm.docs.st.rerun") as rerun,
        ):
            docs.render_doc_buttons()
        self.assertEqual(st.session_state["_app_tabs"], "LLM chat")
        self.assertTrue(docs.is_no_doc_active())
        rerun.assert_called_once()

    def test_placeholder_remains_alongside_open_documents(self):
        docs.add_document("a.md")
        col_switch = MagicMock()
        col_switch.button.return_value = False
        col_close = MagicMock()
        with patch("md_llm.docs.st.columns", return_value=[col_switch, col_close]):
            docs.render_doc_buttons()
        # 2 rows: placeholder + a.md
        self.assertEqual(col_switch.button.call_count, 2)
        self.assertEqual(col_close.button.call_count, 2)
        # first call is the placeholder
        self.assertEqual(col_switch.button.call_args_list[0].args[0],
                         "(no document, direct LLM chat)")
        self.assertTrue(col_close.button.call_args_list[0].kwargs["disabled"])
        # second call is the document (keyed by relpath)
        self.assertEqual(
            col_switch.button.call_args_list[1].kwargs["key"],
            "_md_llm_doc_btn_a.md",
        )
        self.assertFalse(col_close.button.call_args_list[1].kwargs.get(
            "disabled", False))
        # a.md is active (not no-doc), so it is primary
        self.assertEqual(
            col_switch.button.call_args_list[1].kwargs["type"], "primary",
        )

    def test_one_switch_button_per_document_plus_placeholder(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        col_switch = MagicMock()
        col_switch.button.return_value = False
        col_close = MagicMock()
        with patch("md_llm.docs.st.columns", return_value=[col_switch, col_close]):
            docs.render_doc_buttons()
        # 3 rows: placeholder + a.md + b.md
        self.assertEqual(col_switch.button.call_count, 3)
        self.assertEqual(col_close.button.call_count, 3)
        # first is always the placeholder; the rest are doc buttons keyed by rel
        keys = [c.kwargs["key"] for c in col_switch.button.call_args_list]
        self.assertEqual(
            keys,
            ["_md_llm_doc_btn_none", "_md_llm_doc_btn_a.md",
             "_md_llm_doc_btn_b.md"],
        )

    def test_click_switches_the_active_document(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        col_switch = MagicMock()
        # placeholder=False, a.md=False, b.md=True
        col_switch.button.side_effect = [False, False, True]
        col_close = MagicMock()
        col_close.button.return_value = False
        with (
            patch("md_llm.docs.st.columns", return_value=[col_switch, col_close]),
            patch("md_llm.docs.st.rerun") as rerun,
        ):
            docs.render_doc_buttons()
        self.assertEqual(docs.active_document(), "b.md")
        self.assertEqual(st.session_state["_reader_target"], "b.md")
        self.assertEqual(st.session_state["_app_tabs"], "Reader")
        rerun.assert_called_once()

    def test_click_placeholder_activates_no_doc_context(self):
        docs.add_document("a.md")
        col_switch = MagicMock()
        col_switch.button.side_effect = [True, False]  # placeholder clicked
        col_close = MagicMock()
        col_close.button.return_value = False
        with (
            patch("md_llm.docs.st.columns", return_value=[col_switch, col_close]),
            patch("md_llm.docs.st.rerun"),
        ):
            docs.render_doc_buttons()
        self.assertTrue(docs.is_no_doc_active())
        self.assertIsNone(docs.active_document())
        self.assertNotIn("_reader_target", st.session_state)

    def test_click_close_removes_the_document(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        col_switch = MagicMock()
        col_switch.button.return_value = False
        col_close = MagicMock()
        # placeholder close (disabled, won't fire), close a.md, close b.md
        col_close.button.side_effect = [False, True, False]
        with (
            patch("md_llm.docs.st.columns", return_value=[col_switch, col_close]),
            patch("md_llm.docs.st.rerun") as rerun,
        ):
            docs.render_doc_buttons()
        self.assertEqual(docs.open_documents(), ["b.md"])
        self.assertEqual(docs.active_document(), "b.md")
        rerun.assert_called_once()


if __name__ == "__main__":
    unittest.main()
