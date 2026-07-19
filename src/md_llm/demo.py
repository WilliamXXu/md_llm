"""Standalone demo: a one-file Streamlit app over any directory of markdown.

Run with::

    streamlit run -m md_llm.demo

The sidebar holds a directory picker; the two tabs render the md_llm reader and
chat against any ``.md`` / ``.txt`` file in that directory. This is also the
integration reference: a host app does the same ``init(Core(...))`` + two
``st.tabs`` with ``key=TABS_KEY`` then calls ``render_reader()`` / ``render_chat()``.
"""

from __future__ import annotations

import os

import streamlit as st

import md_llm


def _list_documents(root):
    """Collect .md/.txt files under ``root``, top-level files first.

    Returns the list of *relpaths* (relative to ``root``) the selectbox will
    show. Files directly in ``root`` come first (bare names, sorted); then each
    subfolder's files are listed in turn, shown as ``subdir/file`` so the nested
    path is visible in the dropdown. Hidden files/dirs (leading dot) and the
    chat-save subdir (``_chats``) are skipped. Relpaths are what
    ``open_in_reader`` stages, so the reader resolves them against ``base_dir``
    (= ``root``) correctly whether the file is at the top or nested.
    """
    top_level = sorted(
        name for name in os.listdir(root)
        if not name.startswith(".")
        and os.path.isfile(os.path.join(root, name))
        and name.endswith((".md", ".txt"))
    )

    nested = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs and the chat-save subdir in-place so os.walk doesn't
        # descend into them.
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d != "_chats"
        )
        # Only collect from subfolders; the top level is handled above.
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            continue
        for name in sorted(filenames):
            if name.startswith(".") or not name.endswith((".md", ".txt")):
                continue
            rel = os.path.join(rel_dir, name)
            # Normalize for the host OS (so "sub/foo.md" on display everywhere).
            nested.append(rel.replace(os.sep, "/"))

    return top_level + nested


def main():
    st.set_page_config(page_title="md_llm demo", layout="wide")

    with st.sidebar:
        st.subheader("Directory")
        default_dir = os.path.expanduser("~")
        root = st.text_input("Document directory", value=default_dir)
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            st.error("Not a directory.")
            st.stop()
        files = _list_documents(root)
        if not files:
            st.info("No .md / .txt files in this directory.")
            st.stop()
        chosen = st.selectbox(
            "Open in reader", files,
            on_change=lambda: md_llm.open_in_reader(st.session_state.get("_demo_pick")),
            key="_demo_pick",
        )
        # Initial open (without a change event) on first load.
        if "_reader_target" not in st.session_state and chosen:
            md_llm.open_in_reader(chosen)
        st.caption(
            "Saved chats go to a `_chats/` subdirectory of the chosen dir. "
            "Settings persist to `_md_llm_settings.json` there too."
        )

    # Re-inject the host's facts whenever the directory changes. init() itself
    # is idempotent (just overwrites the registered core), but it is NOT called
    # again after the first run, so without this block the Core would keep
    # pointing at the first directory ever picked — and every staged relpath
    # would be resolved against that stale base_dir, so files from a newly
    # chosen directory would silently fail to open.
    if st.session_state.get("_demo_core_root") != root:
        md_llm.init(md_llm.Core(
            base_dir=root,
            markdown_dirs=(root,),
            chat_save_dir=os.path.join(root, "_chats"),
            settings_path=os.path.join(root, "_md_llm_settings.json"),
        ))
        # Drop any file staged against the old directory: its relpath is now
        # ambiguous (a same-named file may exist in the new dir) or stale.
        st.session_state.pop("_reader_target", None)
        st.session_state["_demo_core_root"] = root

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
