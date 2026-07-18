"""Reader panel: open any document (markdown/text) as a clean, full-text view.

A host stages a file by calling :func:`open_in_reader` (which records the target
in session state and switches the host's active tab here). The panel renders the
file, offers copy-to-clipboard, lets the user quote a passage to send to the
chat, and shows a read-only summary of the current chat config.

Path safety: the staged relpath is resolved against ``core.base_dir`` and the
resulting absolute path must sit inside one of ``core.markdown_dirs``; anything
that escapes via ``..`` is rejected before being read.

Shared session-state keys (the integration contract the host honors):
  - ``_reader_target``  — the relpath to display (written by open_in_reader).
  - ``_reader_quote``   — a passage staged for the next chat question.
  - ``TABS_KEY`` / ``READER_TAB_LABEL`` / ``CHAT_TAB_LABEL`` — tab switching.
"""

from __future__ import annotations

import base64
import os

import streamlit as st
import streamlit.components.v1 as components

from . import llm as _llm
from .controls import _current_llm_model
from .core import get_core
from .state import _display_name_for_filepath, _human_size, _read_text

# Session-state key holding the reader target (a relpath against core.base_dir).
_READER_TARGET = "_reader_target"

# A passage staged in the Reader to attach to the next chat question. Shared by
# string literal (the chat panel reads "_reader_quote" too) rather than an
# import, to keep reader↔chat decoupled.
_READER_QUOTE = "_reader_quote"

# The text_area widget key for the quote box (Reader-internal).
_READER_QUOTE_AREA = "_reader_quote_area"

# The st.tabs() key in the host app — writing its session-state value switches
# the active tab. Exported so the host uses this exact key.
TABS_KEY = "_app_tabs"
READER_TAB_LABEL = "Reader"
CHAT_TAB_LABEL = "LLM chat"


def open_in_reader(relpath):
    """Record `relpath` as the reader target and jump to the Reader tab.

    Streamlit tabs are widgets (keyed), so assigning the tab's session-state
    value moves the active tab — no new browser tab, no link navigation.
    """
    if relpath:
        st.session_state[_READER_TARGET] = relpath
    st.session_state[TABS_KEY] = READER_TAB_LABEL


def _resolve_reader_target(rel):
    """Resolve the relpath to a safe absolute path, or None.

    Only paths that land inside one of ``core.markdown_dirs`` are accepted, so a
    crafted value can never read outside the host's own data dirs. Returns None
    (and surfaces an error) when the target is rejected or missing.
    """
    if not rel:
        return None
    base = os.path.abspath(get_core().base_dir)
    target = os.path.abspath(os.path.join(base, rel))
    allowed = tuple(os.path.abspath(d) for d in get_core().markdown_dirs)
    inside = any(
        os.path.commonpath([target, root]) == root for root in allowed
    )
    if not inside:
        st.error("Refusing to open a path outside the configured document dirs.")
        return None
    return target


def _copy_text_button(text, label="Copy"):
    """A labelled copy-to-clipboard button, base64-encoded for safe transport."""
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    components.html(
        f"""
        <style>
            #copyBtn {{
                background: rgb(240, 242, 246);
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.4rem;
                padding: 0.4rem 0.9rem;
                font-size: 0.9rem;
                color: rgb(49, 51, 63);
                cursor: pointer;
            }}
            #copyBtn:hover {{ border-color: rgba(49, 51, 63, 0.4); }}
            #copyBtn.copied {{ background: rgb(212, 237, 218); }}
        </style>
        <button id="copyBtn">{label}</button>
        <script>
            (function () {{
                const btn = document.getElementById('copyBtn');
                const bytes = Uint8Array.from(atob("{b64}"), c => c.charCodeAt(0));
                const text = new TextDecoder('utf-8').decode(bytes);

                async function copyText() {{
                    try {{
                        if (navigator.clipboard && window.isSecureContext) {{
                            await navigator.clipboard.writeText(text);
                            return true;
                        }}
                    }} catch (e) {{}}
                    const ta = document.createElement('textarea');
                    ta.value = text;
                    ta.style.position = 'fixed';
                    ta.style.left = '-9999px';
                    document.body.appendChild(ta);
                    ta.focus();
                    ta.select();
                    let ok = false;
                    try {{ ok = document.execCommand('copy'); }} catch (e) {{}}
                    document.body.removeChild(ta);
                    return ok;
                }}

                btn.addEventListener('click', async () => {{
                    const ok = await copyText();
                    if (ok) {{
                        const original = btn.textContent;
                        btn.textContent = 'Copied!';
                        btn.classList.add('copied');
                        setTimeout(() => {{
                            btn.textContent = original;
                            btn.classList.remove('copied');
                        }}, 1500);
                    }}
                }});
            }})();
        </script>
        """,
        height=48,
    )


def _render_chat_config_summary():
    """A read-only summary of the LLM chat config (no editable widgets here).

    The editable controls live in the chat panel (the only place the chat_*
    widget keys are instantiated). We can't render the same widgets here without
    a duplicate-key collision (every tab mounts its widgets on every run), so
    the Reader just reports the current values and offers a jump button.
    """
    provider = st.session_state.get("chat_llm_provider", "OpenRouter")
    model = _current_llm_model(prefix="chat_") or _llm.OPENROUTER_DEFAULT_MODEL
    if provider == "OpenRouter":
        key_status = (
            "set" if st.session_state.get("chat_llm_or_api_key")
            or os.environ.get("OPENROUTER_API_KEY")
            else "missing"
        )
    else:
        key_status = "n/a"
    st.caption(
        f"Provider: **{provider}**  ·  Model: `{model}`  ·  API key: {key_status}"
    )


def render_reader():
    """Render the Reader panel: show the file targeted by ``open_in_reader``."""
    st.subheader("Reader")
    rel = st.session_state.get(_READER_TARGET)
    target = _resolve_reader_target(rel)

    if not target or not os.path.isfile(target):
        st.caption(
            "_Nothing open. Pick a document to read here (call "
            "`md_llm.open_in_reader(relpath)` from your app)._"
        )
        if rel:
            st.button("Close", on_click=_close_reader)
        return

    text = _read_text(target)
    try:
        size = _human_size(os.path.getsize(target))
    except OSError:
        size = "?"
    # Generic vocab: .md is authored markdown, anything else is shown as text.
    kind = "Markdown" if target.endswith(".md") else "Text"
    st.caption(
        f"{kind}: `{_display_name_for_filepath(target)}`  ·  {size}  ·  "
        f"`{os.path.abspath(target)}`"
    )
    col_copy, col_clear = st.columns([1, 1])
    with col_copy:
        _copy_text_button(text)
    with col_clear:
        st.button("Clear", on_click=_close_reader)

    # .md is authored content → render as markdown; anything else → code block.
    if target.endswith(".md"):
        st.markdown(text)
    else:
        st.code(text, language="text")

    # --- Quote a passage into the chat ---------------------------------
    # The content above is read-only DOM: Streamlit never sees the browser's
    # text selection. So the user selects text, copies it (⌘C / Ctrl-C), pastes
    # it here, and "Send to chat" stages it for the next question in the chat
    # panel — alongside the full document, which is always sent as context.
    st.divider()
    with st.expander("Quote a passage for the LLM chat", expanded=False):
        st.caption(
            "_Select text above, copy it, paste it here, then **Send to chat**. "
            "The quote is attached to your next question in the chat — the full "
            "document is still sent as context too._"
        )
        st.text_area(
            "Quote for chat", value=st.session_state.get(_READER_QUOTE, ""),
            height=120, key=_READER_QUOTE_AREA,
            placeholder="Paste the passage you want to ask about…",
        )
        col_send, col_clear = st.columns([1, 1])
        if col_send.button(
            "Send to chat", type="primary",
            help="Stage this quote for the next chat question, then switch to "
                 "the LLM chat tab.",
        ):
            quote = (st.session_state.get(_READER_QUOTE_AREA) or "").strip()
            if quote:
                st.session_state[_READER_QUOTE] = quote
                st.session_state[TABS_KEY] = CHAT_TAB_LABEL
                st.rerun()
            else:
                st.warning("Paste a passage into the box first.")
        if col_clear.button(
            "Clear", help="Drop the staged quote so it is no longer attached.",
        ):
            st.session_state.pop(_READER_QUOTE, None)
            st.session_state[_READER_QUOTE_AREA] = ""
            st.rerun()

    # A compact read-only summary of the current chat config + a jump button.
    st.divider()
    with st.expander("LLM chat — about this document", expanded=False):
        _render_chat_config_summary()
        if st.button("Open chat", help="Switch to the LLM chat tab to "
                     "converse about this document."):
            st.session_state[TABS_KEY] = CHAT_TAB_LABEL
            st.rerun()


def _close_reader():
    """Drop the reader target (wired as the "Clear" button's on_click callback).

    Also clears any staged quote, so a passage quoted from one document can't
    leak into a chat about another.
    """
    st.session_state.pop(_READER_TARGET, None)
    st.session_state.pop(_READER_QUOTE, None)
    st.session_state[_READER_QUOTE_AREA] = ""
