"""Standalone demo: a one-file Streamlit app over any directory of markdown.

Run with::

    streamlit run src/md_llm/demo.py

The sidebar has a native Streamlit ``st.file_uploader``: clicking it pops up
the browser's OS-level file dialog (Finder on macOS, Explorer on Windows, …),
and the chosen ``.md`` / ``.txt`` file is staged for the Reader tab. The chat
tab talks about whatever is open in the Reader.

Why not ``tkinter.filedialog``? Streamlit runs the script on a worker thread,
but macOS forbids instantiating ``NSWindow`` off the main thread — so a Tk
dialog aborts the whole app with ``NSInternalInconsistencyException``.
``st.file_uploader`` is the supported, cross-platform way to trigger the OS
file dialog from inside Streamlit: it returns the file's *bytes*, so we
materialize them into a stable per-user working dir and let the existing
path-based reader/chat pipeline read them from there.

This file is also the integration reference: a host app does the same
``init(Core(...))`` + two ``st.tabs`` with ``key=TABS_KEY`` then calls
``render_reader()`` / ``render_chat()``.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

import md_llm

# Per-user working dir: uploaded files land in ``uploads/``, saved chats in
# ``uploads/_chats/`` — mirroring the original demo's per-directory layout but
# rooted at a stable spot the user can find afterwards. Created lazily.
_WORK_DIR = Path.home() / ".md_llm"
_UPLOADS_DIR = _WORK_DIR / "uploads"
_CHATS_DIR = _UPLOADS_DIR / "_chats"
_SETTINGS_PATH = _WORK_DIR / "_md_llm_settings.json"


def _ensure_work_dirs():
    """Create the working directories used by the demo (idempotent)."""
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    _CHATS_DIR.mkdir(parents=True, exist_ok=True)


def _install_core():
    """Point md_llm at the working dir.

    The Reader's path-safety guard resolves the staged relpath against
    ``core.base_dir`` and only opens files inside ``core.markdown_dirs``, so we
    need a Core registered even before any file is uploaded (otherwise the
    first render raises). The same Core serves uploaded files: their basename
    lives directly under ``_UPLOADS_DIR`` (= ``base_dir``).
    """
    md_llm.init(md_llm.Core(
        base_dir=str(_UPLOADS_DIR),
        markdown_dirs=(str(_UPLOADS_DIR),),
        chat_save_dir=str(_CHATS_DIR),
        settings_path=str(_SETTINGS_PATH),
    ))


def main():
    st.set_page_config(page_title="md_llm demo", layout="wide", page_icon="📖")
    _ensure_work_dirs()
    _install_core()

    with st.sidebar:
        st.subheader("Open file")
        # Native Streamlit picker: clicking the widget opens the browser's
        # OS-level file dialog. type= restricts the accept filter to .md/.txt
        # in the dialog itself, so the user can't pick anything else.
        uploaded = st.file_uploader(
            "Choose a markdown or text file",
            type=["md", "txt"],
            label_visibility="collapsed",
            help="Opens your OS file dialog. The file is read into a local "
                 "working directory so the Reader can open it.",
        )
        if uploaded is not None:
            # Materialize the upload to disk: the Reader/chat pipeline is
            # path-based (it resolves relpaths against core.base_dir), so we
            # need a real file there. Same-name uploads overwrite, which is
            # the least-surprising behavior for re-picking a file.
            dest = _UPLOADS_DIR / uploaded.name
            try:
                with open(dest, "wb") as f:
                    f.write(uploaded.getvalue())
            except OSError as e:
                st.error(f"Could not stage file: {e}")
                st.stop()
            md_llm.open_in_reader(uploaded.name)
            st.caption(f"Open: `{uploaded.name}`")
        elif st.session_state.get("_reader_target"):
            st.caption(f"Open: `{st.session_state['_reader_target']}`")
        st.caption(
            f"Uploads staged at `{_UPLOADS_DIR}`. Saved chats go to "
            f"`{_CHATS_DIR}`."
        )

    tabs = st.tabs(
        [md_llm.READER_TAB_LABEL, md_llm.CHAT_TAB_LABEL],
        key=md_llm.TABS_KEY,
    )
    with tabs[0]:
        md_llm.render_reader()
    with tabs[1]:
        md_llm.render_chat()


if __name__ == "__main__":
    main()
