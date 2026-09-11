"""Tests for the Reader's safety-locked in-place editing.

These cover the pure pieces: the atomic write helper
(``md_llm.state._write_text``), the unsaved-draft checks
(``reader._doc_edit_dirty`` / ``docs.doc_has_unsaved_edits``), and the
editor key lifecycle (per-document namespacing, cleanup on close). The
lock toggle, the draft textarea, and the confirmation dialogs are
Streamlit / browser behaviour and are not unit-testable here.
"""

import os
import shutil
import tempfile
import unittest

import streamlit as st
from streamlit.testing.v1 import AppTest

from md_llm import docs, reader, state
from md_llm.core import Core, _reset_for_tests


def _clear_edit_state():
    """Drop every reader/docs session key the editor touches."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and (
            k.startswith("_reader_") or k.startswith("_md_llm_")
        ):
            st.session_state.pop(k, None)


class WriteTextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "doc.md")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        _reset_for_tests()

    def test_round_trips_with_read_text(self):
        self.assertTrue(state._write_text(self.path, "# Hello\n\nbody\n"))
        self.assertEqual(state._read_text(self.path), "# Hello\n\nbody\n")

    def test_overwrites_existing_content(self):
        state._write_text(self.path, "old")
        self.assertTrue(state._write_text(self.path, "new"))
        self.assertEqual(state._read_text(self.path), "new")

    def test_writes_utf8(self):
        text = "# 标题\n\nEm—dash « guillemets » 📖\n"
        self.assertTrue(state._write_text(self.path, text))
        self.assertEqual(state._read_text(self.path), text)

    def test_fails_without_creating_directories(self):
        missing = os.path.join(self.tmp, "no", "such", "dir", "doc.md")
        self.assertFalse(state._write_text(missing, "x"))
        self.assertFalse(os.path.isfile(missing))

    def test_fails_on_empty_path(self):
        self.assertFalse(state._write_text("", "x"))

    def test_leaves_no_temp_files_behind(self):
        state._write_text(self.path, "content")
        leftovers = [f for f in os.listdir(self.tmp) if f != "doc.md"]
        self.assertEqual(leftovers, [])


class DocEditDirtyTests(unittest.TestCase):
    """_doc_edit_dirty needs a Core so the path guard can resolve ``rel``."""

    def setUp(self):
        _clear_edit_state()
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "notes.md")
        state._write_text(self.path, "on disk\n")
        _reset_for_tests(Core(
            base_dir=self.tmp,
            markdown_dirs=(self.tmp,),
            chat_save_dir=self.tmp,
        ))
        # _doc_edit_dirty compares the draft against whatever the Reader is
        # displaying (the _READER_TARGET key), not against ``rel``.
        st.session_state["_reader_target"] = "notes.md"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        _reset_for_tests()
        _clear_edit_state()

    def test_no_draft_is_clean(self):
        self.assertFalse(reader._doc_edit_dirty("notes.md"))

    def test_draft_matching_disk_is_clean(self):
        st.session_state[
            docs.doc_key("_reader_edit_draft", "notes.md")
        ] = "on disk\n"
        self.assertFalse(reader._doc_edit_dirty("notes.md"))

    def test_draft_differing_from_disk_is_dirty(self):
        st.session_state[
            docs.doc_key("_reader_edit_draft", "notes.md")
        ] = "edited\n"
        self.assertTrue(reader._doc_edit_dirty("notes.md"))

    def test_draft_for_missing_file_is_dirty(self):
        st.session_state["_reader_target"] = "gone.md"
        st.session_state[docs.doc_key("_reader_edit_draft", "gone.md")] = "x"
        self.assertTrue(reader._doc_edit_dirty("gone.md"))

    def test_draft_key_is_doc_scoped(self):
        st.session_state[
            docs.doc_key("_reader_edit_draft", "notes.md")
        ] = "edited\n"
        # Another open document must not inherit the draft.
        self.assertFalse(reader._doc_edit_dirty("other.md"))


class DocHasUnsavedEditsTests(unittest.TestCase):
    """The docs-side dirty check (sidebar ✕ guard) agrees with the reader's."""

    def setUp(self):
        _clear_edit_state()
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "notes.md")
        state._write_text(self.path, "on disk\n")
        _reset_for_tests(Core(
            base_dir=self.tmp,
            markdown_dirs=(self.tmp,),
            chat_save_dir=self.tmp,
        ))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        _reset_for_tests()
        _clear_edit_state()

    def test_absent_draft_is_clean(self):
        self.assertFalse(docs.doc_has_unsaved_edits("notes.md"))

    def test_matching_draft_is_clean(self):
        st.session_state[
            docs.doc_key("_reader_edit_draft", "notes.md")
        ] = "on disk\n"
        self.assertFalse(docs.doc_has_unsaved_edits("notes.md"))

    def test_differing_draft_is_dirty(self):
        st.session_state[
            docs.doc_key("_reader_edit_draft", "notes.md")
        ] = "edited\n"
        self.assertTrue(docs.doc_has_unsaved_edits("notes.md"))

    def test_missing_file_is_dirty(self):
        st.session_state[docs.doc_key("_reader_edit_draft", "gone.md")] = "x"
        self.assertTrue(docs.doc_has_unsaved_edits("gone.md"))

    def test_without_core_draft_counts_as_dirty(self):
        _reset_for_tests()
        st.session_state[
            docs.doc_key("_reader_edit_draft", "notes.md")
        ] = "edited\n"
        self.assertTrue(docs.doc_has_unsaved_edits("notes.md"))


class CloseCleanupTests(unittest.TestCase):
    """Closing a document must drop its editor state with its chat state."""

    def setUp(self):
        _clear_edit_state()

    def tearDown(self):
        _clear_edit_state()

    def test_single_document_close_drops_bare_editor_keys(self):
        st.session_state["_reader_target"] = "notes.md"
        for base in (
            "_reader_edit_draft",
            "_reader_edit_area",
            "_reader_edit_area__g0",
            "_reader_edit_area__g1",
            "_reader_edit_area_gen",
            "_reader_edit_base_mtime",
            "_reader_edit_unlocked",
            "_reader_edit_block_index",
            "_reader_edit_block_area",
        ):
            st.session_state[base] = "x"
        reader._close_reader()
        self.assertNotIn("_reader_target", st.session_state)
        for key in (
            "_reader_edit_draft",
            "_reader_edit_area",
            "_reader_edit_area__g0",
            "_reader_edit_area__g1",
            "_reader_edit_area_gen",
            "_reader_edit_base_mtime",
            "_reader_edit_unlocked",
            "_reader_edit_block_index",
            "_reader_edit_block_area",
        ):
            self.assertNotIn(key, st.session_state)

    def test_multi_document_close_sweeps_suffixed_editor_keys(self):
        docs.add_document("a.md")
        docs.add_document("b.md")
        for base in ("_reader_edit_draft", "_reader_edit_unlocked",
                     "_reader_edit_area__g2", "_reader_edit_area_gen"):
            st.session_state[docs.doc_key(base, "a.md")] = "x"
        docs.remove_document("a.md")
        self.assertNotIn(
            docs.doc_key("_reader_edit_draft", "a.md"), st.session_state
        )
        self.assertNotIn(
            docs.doc_key("_reader_edit_unlocked", "a.md"), st.session_state
        )
        # Rotated raw-textarea keys end with the doc suffix too, so the
        # suffix sweep catches every generation.
        self.assertNotIn(
            docs.doc_key("_reader_edit_area__g2", "a.md"), st.session_state
        )
        self.assertNotIn(
            docs.doc_key("_reader_edit_area_gen", "a.md"), st.session_state
        )
        # A different document's editor state is untouched.
        docs.add_document("c.md")
        st.session_state[docs.doc_key("_reader_edit_draft", "c.md")] = "y"
        docs.remove_document("c.md")
        self.assertNotIn(
            docs.doc_key("_reader_edit_draft", "c.md"), st.session_state
        )

    def test_edit_keys_carry_the_doc_suffix(self):
        # The suffix is what makes the close-time sweep above work; guard it
        # against accidental renames of the base keys.
        for base in (
            "_reader_edit_draft",
            "_reader_edit_area",
            "_reader_edit_base_mtime",
            "_reader_edit_unlocked",
            "_reader_edit_lock",
            "_reader_edit_block_index",
            "_reader_edit_block_area",
            "_reader_edit_area_gen",
        ):
            self.assertEqual(
                docs.doc_key(base, "a.md"), f"{base}__doc__a.md"
            )


class _HostAppTest(unittest.TestCase):
    """End-to-end widget flows over a minimal host app (Streamlit AppTest).

    The unit tests above cover the pure helpers; these exercise the render
    path itself: lock default, editor seeding, save/revert, and the close
    guards — the parts that only misbehave with real widgets in a run.
    """

    # Session-state writes the AppTest proxy reads by subscript only.
    @staticmethod
    def _sget(at, key):
        try:
            return at.session_state[key]
        except (KeyError, AttributeError):
            return None

    @classmethod
    def _make_host(cls, files=None):
        tmp = tempfile.mkdtemp()
        for name, body in (files or {"notes.md": "on disk\n"}).items():
            with open(os.path.join(tmp, name), "w") as f:
                f.write(body)
        host = os.path.join(tmp, "host_app.py")
        with open(host, "w") as f:
            f.write(
                "import streamlit as st\n"
                "import md_llm\n"
                "\n"
                f"md_llm.init(md_llm.Core(base_dir={tmp!r},"
                f" markdown_dirs=({tmp!r},), chat_save_dir={tmp!r}))\n"
                'md_llm.open_in_reader("notes.md")\n'
                "md_llm.render_reader()\n"
            )
        return tmp, os.path.join(tmp, "notes.md"), host

    @classmethod
    def tearDownClass(cls):
        _reset_for_tests()

    def _element(self, at, kind, key):
        try:
            return getattr(at, kind)(key=key)
        except KeyError:
            return None

    def _raw_key(self, at, doc=""):
        """The whole-file textarea's current (generation-carrying) widget key.

        The key rotates whenever the draft is reset programmatically (block
        commit / revert / reload), so tests resolve it from session state
        instead of hardcoding ``_reader_edit_area``.
        """
        gen = self._sget(
            at, f"_reader_edit_area_gen{f'__doc__{doc}' if doc else ''}"
        ) or 0
        base = f"_reader_edit_area__g{gen}"
        return f"{base}__doc__{doc}" if doc else base


class LockDefaultAppTests(_HostAppTest):
    def setUp(self):
        self.tmp, self.notes, self.host = self._make_host()
        self.at = AppTest.from_file(self.host)
        self.at.run()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fresh_session_starts_locked(self):
        self.assertFalse(self.at.exception)
        self.assertIsNone(self._element(self.at, "button", "_reader_edit_save_btn"))
        # The quick-prompt expander owns a text_area; the editor's is absent.
        with self.assertRaises(KeyError):
            self.at.text_area(key=self._raw_key(self.at))

    def test_unlock_renders_editor_seeded_from_disk(self):
        self.at.session_state["_reader_edit_unlocked"] = True
        self.at.run()
        editor = self._element(self.at, "text_area", self._raw_key(self.at))
        self.assertIsNotNone(editor)
        self.assertEqual(editor.value, "on disk\n")

    def test_save_writes_disk_and_clears_draft(self):
        self.at.session_state["_reader_edit_unlocked"] = True
        self.at.run()
        self.at.text_area(key=self._raw_key(self.at)).set_value("edited body\n")
        self.at.run()
        self.at.button(key="_reader_edit_save_btn").click()
        self.at.run()
        self.assertFalse(self.at.exception)
        with open(self.notes) as f:
            self.assertEqual(f.read(), "edited body\n")
        self.assertIsNone(self._sget(self.at, "_reader_edit_draft"))

    def test_revert_discards_draft_and_keeps_file(self):
        self.at.session_state["_reader_edit_unlocked"] = True
        self.at.run()
        self.at.text_area(key=self._raw_key(self.at)).set_value("junk\n")
        self.at.run()
        self.at.button(key="_reader_edit_revert_btn").click()
        self.at.run()
        self.assertFalse(self.at.exception)
        with open(self.notes) as f:
            self.assertEqual(f.read(), "on disk\n")
        # The revert rotates the raw textarea's key; the fresh textarea is
        # seeded from disk, not from the discarded draft.
        self.assertEqual(
            self.at.text_area(key=self._raw_key(self.at)).value, "on disk\n"
        )

    def test_dirty_close_is_blocked(self):
        self.at.session_state["_reader_edit_unlocked"] = True
        self.at.run()
        self.at.text_area(key=self._raw_key(self.at)).set_value("dirty edit\n")
        self.at.run()
        self.at.button(key="_reader_close_doc_btn").click()
        self.at.run()
        self.assertFalse(self.at.exception)
        # The dialog intercepted: the document is still open.
        self.assertEqual(self._sget(self.at, "_reader_target"), "notes.md")

    def test_conflict_stands_between_draft_and_outside_change(self):
        self.at.session_state["_reader_edit_unlocked"] = True
        self.at.run()
        with open(self.notes, "w") as f:
            f.write("changed outside\n")
        os.utime(self.notes, (os.path.getmtime(self.notes) + 5,) * 2)
        self.at.text_area(key=self._raw_key(self.at)).set_value("my draft\n")
        self.at.run()
        self.at.button(key="_reader_edit_save_btn").click()
        self.at.run()
        self.assertFalse(self.at.exception)
        # The conflict dialog fired instead of the write.
        with open(self.notes) as f:
            self.assertEqual(f.read(), "changed outside\n")
        self.assertEqual(
            self._sget(self.at, "_reader_edit_pending"), "my draft\n"
        )


class MultiDocEditAppTests(_HostAppTest):
    def setUp(self):
        files = {"a.md": "aaa\n", "b.md": "bbb\n"}
        self.tmp, _, self.host = self._make_host(files)
        host = open(self.host).read().replace(
            'md_llm.open_in_reader("notes.md")',
            'if "boot" not in st.session_state:\n'
            '    st.session_state["boot"] = True\n'
            '    md_llm.open_in_reader("a.md", keep_open=True)\n'
            '    md_llm.open_in_reader("b.md", keep_open=True)',
        )
        with open(self.host, "w") as f:
            f.write(host)
        self.at = AppTest.from_file(self.host)
        self.at.run()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_each_document_keys_its_own_editor(self):
        self.assertFalse(self.at.exception)
        self.assertEqual(self._sget(self.at, "_reader_target"), "b.md")
        self.at.session_state["_reader_edit_unlocked__doc__b.md"] = True
        self.at.run()
        editor = self._element(
            self.at, "text_area", self._raw_key(self.at, doc="b.md")
        )
        self.assertIsNotNone(editor)
        self.assertEqual(editor.value, "bbb\n")

    def test_save_writes_only_the_active_document(self):
        self.at.session_state["_reader_edit_unlocked__doc__b.md"] = True
        self.at.run()
        self.at.text_area(
            key=self._raw_key(self.at, doc="b.md")
        ).set_value("bee\n")
        self.at.run()
        self.at.button(key="_reader_edit_save_btn").click()
        self.at.run()
        self.assertFalse(self.at.exception)
        with open(os.path.join(self.tmp, "b.md")) as f:
            self.assertEqual(f.read(), "bee\n")
        with open(os.path.join(self.tmp, "a.md")) as f:
            self.assertEqual(f.read(), "aaa\n")


class MdBlocksTests(unittest.TestCase):
    """The block splitter behind the in-place editor."""

    def test_top_level_blocks_span_whole_constructs(self):
        src = (
            "# T\n"
            "\n"
            "para one.\n"
            "\n"
            "- a\n"
            "- b\n"
            "\n"
            "| h |\n"
            "|---|\n"
            "| 1 |\n"
            "\n"
            "```py\n"
            "# not a heading\n"
            "```\n"
            "\n"
            "tail\n"
        )
        blocks, refs = reader._md_blocks(src)
        self.assertEqual(
            [b[2] for b in blocks],
            [
                "# T\n",
                "para one.\n",
                "- a\n- b\n\n",  # lists absorb their trailing blank line
                "| h |\n|---|\n| 1 |\n",
                "```py\n# not a heading\n```\n",
                "tail\n",
            ],
        )
        self.assertEqual(refs, "")

    def test_reference_definitions_come_back_as_source(self):
        src = "para [r][1].\n\n[1]: https://x\n\ntail\n"
        blocks, refs = reader._md_blocks(src)
        self.assertEqual([b[2] for b in blocks], ["para [r][1].\n", "tail\n"])
        self.assertEqual(refs, "[1]: https://x\n")

    def test_empty_text_has_no_blocks(self):
        self.assertEqual(reader._md_blocks(""), ([], ""))


class BlockEditAppTests(_HostAppTest):
    """The unlocked .md view: rendered blocks with in-place ✎ editors.

    Clicking a handle swaps just that block for a source editor; committing
    splices the block's lines back into the draft (never the disk — 💾 Save
    does that), and everything else stays rendered.
    """

    BODY = (
        "# Title\n"
        "\n"
        "First paragraph.\n"
        "\n"
        "## Section\n"
        "\n"
        "- a\n"
        "- b\n"
        "\n"
        "Last paragraph.\n"
    )

    def setUp(self):
        self.tmp, self.notes, self.host = self._make_host(
            {"notes.md": self.BODY}
        )
        self.at = AppTest.from_file(self.host)
        self.at.run()
        self.at.session_state["_reader_edit_unlocked"] = True
        self.at.run()

    def test_unlock_renders_blocks_with_handles(self):
        self.assertFalse(self.at.exception)
        # The document renders as separate blocks (each with its own handle),
        # not one whole-document markdown blob.
        self.assertIsNotNone(
            self._element(self.at, "button", "_reader_blk_btn_0")
        )
        self.assertIsNotNone(
            self._element(self.at, "button", "_reader_blk_btn_1")
        )
        rendered = [
            m.value for m in self.at.markdown if m.value
            and "<style>" not in m.value
        ]
        self.assertTrue(
            any(v.strip() == "First paragraph." for v in rendered)
        )

    def test_handle_opens_in_place_editor_for_that_block(self):
        self.at.button(key="_reader_blk_btn_1").click()
        self.at.run()
        area = self._element(self.at, "text_area", "_reader_edit_block_area")
        self.assertIsNotNone(area)
        self.assertEqual(area.value, "First paragraph.\n")

    def test_commit_splices_draft_closes_editor_keeps_disk(self):
        self.at.button(key="_reader_blk_btn_1").click()
        self.at.run()
        self.at.text_area(key="_reader_edit_block_area").set_value(
            "Rewritten!\n"
        )
        self.at.run()
        self.assertFalse(self.at.exception)
        self.assertEqual(
            self._sget(self.at, "_reader_edit_draft"),
            self.BODY.replace("First paragraph.\n", "Rewritten!\n"),
        )
        with open(self.notes) as f:
            self.assertEqual(f.read(), self.BODY)
        # The editor closed: no block textarea is mounted any more.
        with self.assertRaises(KeyError):
            self.at.text_area(key="_reader_edit_block_area")

    def test_list_replacement_regains_its_blank_separator(self):
        # A list block absorbs the blank line after it; replacing the list
        # without retyping that blank must not fuse it into the next block.
        self.at.button(key="_reader_blk_btn_3").click()  # the "- a / - b" list
        self.at.run()
        self.at.text_area(key="_reader_edit_block_area").set_value("- x\n- y\n")
        self.at.run()
        self.assertFalse(self.at.exception)
        self.assertEqual(
            self._sget(self.at, "_reader_edit_draft"),
            self.BODY.replace("- a\n- b\n", "- x\n- y\n"),
        )

    def test_commit_then_save_writes_disk_and_clears_draft(self):
        self.at.button(key="_reader_blk_btn_1").click()
        self.at.run()
        self.at.text_area(key="_reader_edit_block_area").set_value(
            "Rewritten!\n"
        )
        self.at.run()
        self.at.button(key="_reader_edit_save_btn").click()
        self.at.run()
        self.assertFalse(self.at.exception)
        with open(self.notes) as f:
            self.assertEqual(
                f.read(), self.BODY.replace("First paragraph.\n", "Rewritten!\n")
            )
        self.assertIsNone(self._sget(self.at, "_reader_edit_draft"))

    def test_cancel_closes_without_splicing(self):
        self.at.button(key="_reader_blk_btn_1").click()
        self.at.run()
        self.at.button(key="_reader_blk_cancel_btn").click()
        self.at.run()
        self.assertFalse(self.at.exception)
        self.assertIsNone(self._sget(self.at, "_reader_edit_draft"))
        with self.assertRaises(KeyError):
            self.at.text_area(key="_reader_edit_block_area")

    def test_block_edit_is_blocked_by_the_dirty_close_guard(self):
        self.at.button(key="_reader_blk_btn_1").click()
        self.at.run()
        self.at.text_area(key="_reader_edit_block_area").set_value(
            "Rewritten!\n"
        )
        self.at.run()
        self.at.button(key="_reader_close_doc_btn").click()
        self.at.run()
        # The dialog intercepted: the document is still open.
        self.assertEqual(self._sget(self.at, "_reader_target"), "notes.md")

    def test_block_commit_reseeds_the_raw_expander(self):
        # The raw-source expander must never hold a stale pre-splice copy:
        # after a block commit its textarea REMOUNTS (key rotation) seeded
        # from the updated draft.
        self.at.button(key="_reader_blk_btn_1").click()
        self.at.run()
        self.at.text_area(key="_reader_edit_block_area").set_value(
            "Rewritten!\n"
        )
        self.at.run()
        self.assertEqual(
            self.at.text_area(key=self._raw_key(self.at)).value,
            self.BODY.replace("First paragraph.\n", "Rewritten!\n"),
        )

    def test_raw_commit_closes_the_block_editor(self):
        self.at.button(key="_reader_blk_btn_1").click()
        self.at.run()
        self.at.text_area(key=self._raw_key(self.at)).set_value(
            "whole new doc\n"
        )
        self.at.run()
        self.assertFalse(self.at.exception)
        self.assertEqual(
            self._sget(self.at, "_reader_edit_draft"), "whole new doc\n"
        )
        with self.assertRaises(KeyError):
            self.at.text_area(key="_reader_edit_block_area")


if __name__ == "__main__":
    unittest.main()
