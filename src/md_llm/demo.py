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
``init(Core(...))`` + a widget keyed ``key=TABS_KEY`` (here a pair of sidebar
buttons; a top-of-page ``st.tabs`` works too) to select the active view, then
calls ``render_reader()`` / ``render_chat()``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import md_llm

# Per-user working dir: uploaded files land in ``uploads/``, saved chats in
# ``uploads/_chats/`` — mirroring the original demo's per-directory layout but
# rooted at a stable spot the user can find afterwards. Created lazily.
_WORK_DIR = Path.home() / ".md_llm"
_UPLOADS_DIR = _WORK_DIR / "uploads"
_CHATS_DIR = _UPLOADS_DIR / "_chats"
_SETTINGS_PATH = _WORK_DIR / "_md_llm_settings.json"
_LAST_UPLOAD_KEY = "_demo_last_uploaded_name"


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


def _preserve_reader_scroll():
    """Remember the Reader's scroll position across view switches.

    Streamlit tears down a panel's DOM when it isn't the active view, so going
    Reader -> chat -> Reader resets the document to the top. This injects a tiny
    same-origin iframe script (via ``components.html``, the only Streamlit escape
    hatch for running JS) that, while the Reader is mounted, continuously
    persists the page scroller's scrollTop to sessionStorage — keyed per document
    so opening a different file starts at the top — and restores it on mount.
    The interval lives inside the iframe, so it stops automatically when the
    Reader is torn down; the saved value therefore always reflects the Reader's
    last position, never the chat's.
    """
    doc_name = st.session_state.get("_reader_target", "")
    key = json.dumps("mdllm_reader_scroll::" + doc_name)
    components.html(
        f"""
        <script>
        (function () {{
          try {{
            var d = window.parent.document;
            // Find the nearest actually-scrollable ancestor of this iframe
            // (robust across Streamlit versions — no hard-coded test ids).
            var s = null, n = window.frameElement;
            while (n && n !== d.body) {{
              var oy = window.parent.getComputedStyle(n).overflowY;
              if ((oy === 'auto' || oy === 'scroll' || oy === 'overlay') &&
                  n.scrollHeight - n.clientHeight > 4) {{ s = n; break; }}
              n = n.parentElement;
            }}
            if (!s) {{ s = d.scrollingElement || d.body; }}

            var K = {key};
            function saveNow() {{
              try {{ sessionStorage.setItem(K, String(s.scrollTop)); }} catch (e) {{}}
            }}
            // Restore the last position once the markdown has laid out.
            var v = parseInt(sessionStorage.getItem(K) || '0', 10) || 0;
            function ap() {{ try {{ s.scrollTop = v; }} catch (e) {{}} }}
            requestAnimationFrame(ap);
            setTimeout(ap, 50);
            setTimeout(ap, 200);
            setTimeout(ap, 600);

            // Persist while mounted; also capture at teardown so a switch away
            // records the exact last position (interval may lag up to 400ms).
            setInterval(saveNow, 400);
            window.addEventListener('pagehide', saveNow);
            window.addEventListener('beforeunload', saveNow);
          }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
    )


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
            # Stage + open ONLY on a changed upload: st.file_uploader keeps
            # returning the uploaded file on every rerun, so re-running this
            # block each time would re-call open_in_reader() and force the view
            # back to Reader every rerun (making the chat unreachable while a
            # file is open, since the active view is st.session_state[TABS_KEY]).
            if st.session_state.get(_LAST_UPLOAD_KEY) != uploaded.name:
                dest = _UPLOADS_DIR / uploaded.name
                try:
                    with open(dest, "wb") as f:
                        f.write(uploaded.getvalue())
                except OSError as e:
                    st.error(f"Could not stage file: {e}")
                    st.stop()
                md_llm.open_in_reader(uploaded.name)
                st.session_state[_LAST_UPLOAD_KEY] = uploaded.name
            st.caption(f"Open: `{uploaded.name}`")
        else:
            st.session_state.pop(_LAST_UPLOAD_KEY, None)
            if st.session_state.get("_reader_target"):
                st.caption(f"Open: `{st.session_state['_reader_target']}`")
        st.caption(
            f"Uploads staged at `{_UPLOADS_DIR}`. Saved chats go to "
            f"`{_CHATS_DIR}`."
        )

        # View switcher: two buttons under the file picker. The active view is
        # st.session_state[TABS_KEY] (also driven by open_in_reader() and the
        # Reader's "Send to chat"), so clicking a button just writes that key
        # and reruns. The CSS enlarges + centers the sidebar button labels.
        st.divider()
        st.markdown(
            "<style>"
            "[data-testid=\"stSidebar\"] button{"
            "font-size:1.18rem!important;font-weight:600!important}"
            "[data-testid=\"stSidebar\"] button *{font-size:inherit!important}"
            "</style>",
            unsafe_allow_html=True,
        )
        _cur = st.session_state.get(md_llm.TABS_KEY, md_llm.READER_TAB_LABEL)
        _nav1, _nav2 = st.columns(2)
        if _nav1.button(
            md_llm.READER_TAB_LABEL, use_container_width=True,
            type="primary" if _cur == md_llm.READER_TAB_LABEL else "secondary",
        ):
            st.session_state[md_llm.TABS_KEY] = md_llm.READER_TAB_LABEL
            st.rerun()
        if _nav2.button(
            md_llm.CHAT_TAB_LABEL, use_container_width=True,
            type="primary" if _cur == md_llm.CHAT_TAB_LABEL else "secondary",
        ):
            st.session_state[md_llm.TABS_KEY] = md_llm.CHAT_TAB_LABEL
            st.rerun()

    if st.session_state.get(md_llm.TABS_KEY, md_llm.READER_TAB_LABEL) \
            == md_llm.READER_TAB_LABEL:
        md_llm.render_reader()
        _preserve_reader_scroll()
    else:
        md_llm.render_chat()


if __name__ == "__main__":
    main()
