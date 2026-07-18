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
        files = sorted(
            f for f in os.listdir(root)
            if f.endswith((".md", ".txt")) and os.path.isfile(os.path.join(root, f))
        )
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

    # Inject the host's facts once per process (idempotent).
    try:
        md_llm.get_core()
    except RuntimeError:
        md_llm.init(md_llm.Core(
            base_dir=root,
            markdown_dirs=(root,),
            chat_save_dir=os.path.join(root, "_chats"),
            settings_path=os.path.join(root, "_md_llm_settings.json"),
        ))

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
