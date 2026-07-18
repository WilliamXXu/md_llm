"""Tests for saving an md_llm chat conversation to chat_save_dir.

A saved chat is a plain ``<docstem>__chat_<UTC>.md`` file — no sidecar metadata,
no transcript linkage (md_llm has no notion of transcripts). The save helper's
pure-I/O part (``_write_chat_md``) and the markdown renderer
(``_render_chat_as_markdown``) are exercised directly; the Streamlit-facing
wrapper is too thin to test without a script context.

Ported from transcriber_system's test_chat_save.py, adapted to md_llm's
plain-.md save behavior.
"""

import os
import tempfile
import unittest

import streamlit as st

from md_llm import chat, core
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


if __name__ == "__main__":
    unittest.main()
