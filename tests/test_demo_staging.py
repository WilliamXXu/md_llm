"""Tests for the demo's uploader staging (``md_llm.demo._stage_new_uploads``).

st.file_uploader returns the same value on every rerun, so staging must tell
a genuine (re-)pick from a plain widget-value replay. The regression these
guard against: closing a document (its ✕) triggered a rerun that pruned the
file's name from the staged set and instantly re-opened the document the
user had just closed. The fix keys staging on the uploader's per-upload
``file_id`` (new on every browser upload, stable while the value is merely
replayed), exercised here through the same sequence of calls.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st

from md_llm import demo, docs


class _FakeUpload:
    """Minimal stand-in for streamlit.runtime.uploaded_file_manager.UploadedFile."""

    def __init__(self, name, file_id, data=b"content", size=None):
        self.name = name
        self.file_id = file_id
        self.size = size if size is not None else len(data)
        self._data = data

    def getvalue(self):
        return self._data


def _clear_state():
    for k in list(st.session_state.keys()):
        if not isinstance(k, str):
            continue
        if (
            k.startswith("_md_llm_")
            or k.startswith("_chat_")
            or k.startswith("_reader_")
            or k.startswith("_app_")
            or k.startswith("_demo_")
        ):
            st.session_state.pop(k, None)


class StageNewUploadsTests(unittest.TestCase):
    def setUp(self):
        _clear_state()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._uploads = Path(self._tmp.name)
        patcher = patch.object(demo, "_UPLOADS_DIR", self._uploads)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        _clear_state()

    def test_fresh_pick_stages_and_opens(self):
        up = _FakeUpload("notes.md", "fid-1")
        demo._stage_new_uploads([up])
        self.assertEqual(docs.open_documents(), ["notes.md"])
        self.assertTrue((self._uploads / "notes.md").read_bytes() == b"content")

    def test_replay_after_close_does_not_reopen(self):
        """The regression: the rerun a ✕ close triggers must not re-open."""
        up = _FakeUpload("notes.md", "fid-1")
        demo._stage_new_uploads([up])                      # pick
        demo._stage_new_uploads([up])                      # plain rerun replay
        self.assertEqual(docs.open_documents(), ["notes.md"])
        docs.remove_document("notes.md")                   # the ✕ / Clear close
        demo._stage_new_uploads([up])                      # the close's rerun
        self.assertEqual(docs.open_documents(), [])

    def test_repick_with_new_file_id_reopens(self):
        up = _FakeUpload("notes.md", "fid-1")
        demo._stage_new_uploads([up])
        docs.remove_document("notes.md")
        demo._stage_new_uploads([_FakeUpload("notes.md", "fid-2")])
        self.assertEqual(docs.open_documents(), ["notes.md"])

    def test_replay_keeps_other_documents(self):
        a = _FakeUpload("a.md", "fid-a")
        b = _FakeUpload("b.md", "fid-b")
        demo._stage_new_uploads([a, b])
        docs.remove_document("a.md")
        demo._stage_new_uploads([a, b])                    # replay, a.md closed
        self.assertEqual(docs.open_documents(), ["b.md"])
        self.assertEqual(docs.active_document(), "b.md")

    def test_new_pick_alongside_closed_replay_stages_only_the_new(self):
        a = _FakeUpload("a.md", "fid-a")
        demo._stage_new_uploads([a])
        docs.remove_document("a.md")
        demo._stage_new_uploads([a, _FakeUpload("c.md", "fid-c")])
        self.assertEqual(docs.open_documents(), ["c.md"])

    def test_empty_uploads_drop_the_staging_key(self):
        demo._stage_new_uploads([_FakeUpload("notes.md", "fid-1")])
        demo._stage_new_uploads([])
        self.assertNotIn(demo._LAST_UPLOAD_KEY, st.session_state)
        demo._stage_new_uploads([])                        # idempotent
        self.assertNotIn(demo._LAST_UPLOAD_KEY, st.session_state)

    def test_fallback_identity_without_file_id(self):
        """Old Streamlits without file_id at least skip in-session duplicates."""
        up = _FakeUpload("notes.md", None)
        demo._stage_new_uploads([up])
        demo._stage_new_uploads([up])
        self.assertEqual(docs.open_documents(), ["notes.md"])


if __name__ == "__main__":
    unittest.main()
