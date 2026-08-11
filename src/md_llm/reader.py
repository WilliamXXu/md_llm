"""Reader panel: open any document (markdown/text) as a clean, full-text view.

A host stages a file by calling :func:`open_in_reader` (which records the target
in session state and switches the host's active tab here). With ``keep_open=True``
the file joins the open-documents registry and becomes the active one — each
open document gets its own Reader view and an independent LLM chat (see
:mod:`md_llm.docs`). The panel renders the file, offers copy-to-clipboard, lets
the user quote a passage to send to the chat, and shows a read-only summary of
the current chat config.

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
import json
import os
import re

import streamlit as st
import streamlit.components.v1 as components

from . import docs
from . import llm as _llm
from .controls import _current_llm_model
from .core import get_core
from .state import (
    _BODY_FONT_SIZE_CSS,
    _display_name_for_filepath,
    _human_size,
    _read_text,
)

# Session-state key holding the reader target (a relpath against core.base_dir).
_READER_TARGET = "_reader_target"

# A passage staged in the Reader to attach to the next chat question. Shared by
# string literal (the chat panel reads "_reader_quote" too) rather than an
# import, to keep reader↔chat decoupled.
_READER_QUOTE = "_reader_quote"

# A heading the sidebar table of contents wants to jump to, staged as the
# DOM-matching signature "H<level>|<normalized title>" (see _inject_toc_jump).
_TOC_JUMP = "_reader_toc_jump"

# Collapsible ToC expansion state: (open target relpath, set of open node ids).
# Stored alongside the target so switching documents resets the tree.
_TOC_OPEN = "_reader_toc_open"
_TOC_OPEN_CTX = "_reader_toc_open_ctx"

# CSS that turns the flat ToC button list into a hierarchy: level-1 rows read
# as section headers (bold + accent bar), deeper rows step inward as a tree
# gutter. Keyed rows are `_reader_toc_l<level>_<i>` and levels > 5 share the
# l5 depth so the panel never collapses.
_TOC_CSS = """
[class*="st-key-_reader_toc_area"] h3 {
    font-size: 0.95rem !important;
    margin: 0 0 0.25rem 0 !important;
}
[class*="st-key-_reader_toc_area"] [data-testid="stVerticalBlock"] {
    gap: 0.1rem !important;
}
[class*="st-key-_reader_toc_area"] button {
    min-height: 0 !important;
    height: auto !important;
    line-height: 1.2 !important;
    padding: 0.1rem 0.4rem !important;
    margin-top: 0.05rem !important;
    margin-bottom: 0.05rem !important;
}
[class*="st-key-_reader_toc_l1"] button {
    font-weight: 700;
    font-size: 0.78em !important;
    border-left: 3px solid rgb(214, 40, 40);
}
[class*="st-key-_reader_toc_l1"] button > div { padding-left: 0.45rem; }
[class*="st-key-_reader_toc_l2"] button { margin-left: 0.7rem; font-size: 0.72em !important; }
[class*="st-key-_reader_toc_l3"] button { margin-left: 1.4rem; }
[class*="st-key-_reader_toc_l4"] button { margin-left: 2.1rem; }
[class*="st-key-_reader_toc_l5"] button { margin-left: 2.1rem; }
[class*="st-key-_reader_toc_l2"] button,
[class*="st-key-_reader_toc_l3"] button,
[class*="st-key-_reader_toc_l4"] button,
[class*="st-key-_reader_toc_l5"] button {
    border: 1px solid transparent;
    border-left: 2px solid rgba(49, 51, 63, 0.16);
}
[class*="st-key-_reader_toc_l3"] button,
[class*="st-key-_reader_toc_l4"] button,
[class*="st-key-_reader_toc_l5"] button { font-size: 0.68em !important; }
"""


def _toc_row_key(level, index):
    """Stable sidebar-ToC button key: encodes the heading level for tree CSS."""
    return f"_reader_toc_l{min(level, 5)}_{index}"


def _toc_depths(entries):
    """Map each entry's markdown level to a tree depth, re-rooted at the
    document's topmost level (so a doc whose body starts at ``##`` still gets
    real section headers). Not capped: nesting is decided by the full relative
    depth (row keys apply the visual depth cap separately)."""
    if not entries:
        return []
    base = min(level for level, _ in entries)
    return [level - base + 1 for level, _ in entries]


def _toc_tree(entries):
    """Build the heading tree from flat ``(level, title)`` entries.

    Each node is ``{"id", "depth", "level", "title", "children": [...]}``
    where ``id`` is the entry's index (stable across reruns, so it can key
    the open/closed state) and ``children`` are the nodes nested directly
    beneath it. Returns the list of top-level roots.
    """
    depths = _toc_depths(entries)
    roots, stack = [], []
    for i, ((level, title), depth) in enumerate(zip(entries, depths)):
        node = {"id": i, "depth": depth, "level": level, "title": title, "children": []}
        while stack and stack[-1]["depth"] >= depth:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def _toc_auto_open(roots):
    """ids to start expanded on a fresh document: the single root when the
    document has exactly one top-level heading (usually the title), so its
    sections are visible without an extra click. Empty for multi-root docs."""
    if len(roots) == 1:
        return {roots[0]["id"]}
    return set()

# The text_area widget key for the quote box (Reader-internal).
_READER_QUOTE_AREA = "_reader_quote_area"


def _reader_quote_key():
    """Session key of the quote staged for the ACTIVE chat session.

    Scoped per document AND per chat session (see :mod:`md_llm.docs`) so a
    passage quoted from one document can never attach to another's
    conversation, and a quote always lands in the chat session that was active
    when it was staged. Falls back to the legacy bare key in single-document
    mode with a single session.
    """
    doc = docs.active_document()
    return docs.chat_key(_READER_QUOTE, docs.active_chat(doc), doc)


def _reader_quote_area_key():
    """Session key of the quote textarea widget for the ACTIVE chat session."""
    doc = docs.active_document()
    return docs.chat_key(_READER_QUOTE_AREA, docs.active_chat(doc), doc)


# The st.tabs() key in the host app — writing its session-state value switches
# the active tab. Exported so the host uses this exact key.
TABS_KEY = "_app_tabs"
READER_TAB_LABEL = "Reader"
CHAT_TAB_LABEL = "LLM chat"


def open_in_reader(relpath, keep_open=False):
    """Record `relpath` as the reader target and jump to the Reader tab.

    Streamlit tabs are widgets (keyed), so assigning the tab's session-state
    value moves the active tab — no new browser tab, no link navigation.

    With ``keep_open=True`` the document is opened in multi-document mode
    (see :mod:`md_llm.docs`): it joins the registry of open documents and
    becomes the active one, each document keeping its own independent LLM
    chat. The default keeps today's single-document behaviour — any previously
    open documents are dropped and the session returns to the legacy keys.
    """
    if relpath:
        st.session_state[_READER_TARGET] = relpath
    if keep_open:
        docs.add_document(relpath)
    else:
        docs.reset_documents()
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


# Markdown constructs stripped when converting a heading line to its plain
# display text / DOM-matching signature (order matters: links first, then tags,
# then span markers).
_MD_AUTOLINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_MD_SPAN_MARK = re.compile(r"[`*_~]")


def _normalize_heading(raw):
    """Markdown heading text -> plain text for display + DOM matching.

    ``**bold**``, ```code```, ``[link](url)``, ``<b>x</b>`` all collapse to
    their visible text, and whitespace runs collapse to single spaces — the
    same normalization the jump script applies on the rendered DOM side.
    """
    s = _MD_AUTOLINK.sub(r"\1", raw)
    s = _HTML_TAG.sub("", s)
    s = _MD_SPAN_MARK.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _toc_entries(text):
    """Parse ATX headings (``#``..``######``) out of markdown text.

    Fenced code blocks are skipped, so a ``# fake heading`` inside a ``` … ```
    block is not treated as a heading. Returns ``[(level, title)]`` where
    level is 1..6 and title is the ``_normalize_heading``-ed plain text
    (trailing closing ``#``s of Setext-style lines are stripped).
    """
    entries = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if not m:
            continue
        title = _normalize_heading(re.sub(r"\s+#+\s*$", "", m.group(2)).strip())
        if title:
            entries.append((len(m.group(1)), title))
    return entries


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


def render_toc():
    """Click-expandable table of contents for the document open in the Reader.

    Meant for a host's left-side panel / sidebar, rendered next to the Reader.
    Parses the opened markdown's ATX headings into a tree; only the top-level
    headings are shown, and clicking one reveals its children (and jumps the
    Reader there) — a second click folds it back up. A document with a single
    top-level heading (a title) starts expanded so its sections are visible.
    Deeper rows step inward under their parent; clicking a leaf jumps to it.
    Clicking any row also switches to the Reader tab if another view is active.
    No-op for text files (no headings) and when nothing is open.
    """
    rel = st.session_state.get(_READER_TARGET)
    target = _resolve_reader_target(rel)
    if not target or not target.endswith(".md"):
        return

    with st.container(key="_reader_toc_area"):
        st.subheader("Contents")
        st.markdown(f"<style>{_TOC_CSS}</style>", unsafe_allow_html=True)
        entries = _toc_entries(_read_text(target))
        if not entries:
            st.caption("_No headings in this document._")
            return
        # A pathological doc with thousands of headings would flood the panel.
        entries = entries[:200]
        roots = _toc_tree(entries)

        # Expansion state is tied to the open document: switching files drops it
        # and starts a fresh tree (single-root docs get their root pre-opened).
        ctx = st.session_state.get(_TOC_OPEN_CTX)
        if ctx and ctx[0] == target:
            open_ids = ctx[1]
        else:
            open_ids = set()
            st.session_state[_TOC_OPEN_CTX] = (target, open_ids)
        if not open_ids:
            open_ids.update(_toc_auto_open(roots))

        _render_toc_nodes(roots, open_ids)


def _render_toc_nodes(nodes, open_ids):
    """Recursively render a ToC row per node; expanded parents render their
    children (each click also jumps the Reader to the node's heading)."""
    for node in nodes:
        node_id = node["id"]
        has_children = bool(node["children"])
        opened = node_id in open_ids
        caret = ("▾ " if opened else "▸ ") if has_children else ""
        title = node["title"]
        if st.button(
            f"{caret}{title}",
            key=_toc_row_key(node["depth"], node_id),
            use_container_width=True,
            help=f"Jump to “{title}”",
        ):
            if has_children:
                if opened:
                    open_ids.discard(node_id)
                else:
                    open_ids.add(node_id)
            # Raw markdown level in the signature: the jump script matches
            # the rendered DOM's actual heading element (h3 stays h3).
            st.session_state[_TOC_JUMP] = f"H{node['level']}|{title}"
            st.session_state[TABS_KEY] = READER_TAB_LABEL
            st.rerun()
        if has_children and opened:
            _render_toc_nodes(node["children"], open_ids)


def _inject_toc_jump(sig):
    """Scroll the rendered document to the heading described by ``sig``.

    The Reader's content is plain render-only DOM — Streamlit knows nothing
    about the headings inside it — so a tiny same-origin iframe script (the
    same escape hatch ``demo._preserve_reader_scroll`` uses to remember the
    TOC location) finds the heading element in the main scroller and places
    it just below the top edge, with a brief highlight so the reader sees
    where they landed. ``sig`` is ``"H<level>|<normalized title>"``, produced
    by :func:`render_toc`.
    """
    payload = json.dumps({"sig": sig}).replace("</", "<\\/")
    components.html(
        f"""
        <script>
        (function () {{
          try {{
            var d = window.parent.document;
            var P = {payload};
            // Claim a "recent jump" on the shared parent document so the
            // demo's scroll-restore script (which re-establishes the last
            // position for ~7.5 s after each mount) defers to us instead of
            // fighting the jump. No storage-key coupling: just a timestamp.
            try {{
              d['__mdllm_recent_jump'] = String(Date.now());
            }} catch (e) {{}}
            var sels = [
              '[data-testid="stMain"]',
              '[data-testid="stMainViewContainer"]',
              'section.main',
            ];
            var scroller = null;
            for (var i = 0; i < sels.length; i++) {{
              if (scroller) break;
              var el = d.querySelector(sels[i]);
              if (el) scroller = el;
            }}
            if (!scroller) return;
            var want = P.sig, sep = want.indexOf('|');
            var lvl = parseInt(want.slice(1, sep), 10);
            var title = want.slice(sep + 1);
            // Mirror reader._normalize_heading: links, tags, span markers,
            // whitespace runs — so the source heading matches what the
            // markdown renderer actually produced in the DOM.
            function norm(t) {{
              return t.replace(/\\[([^\\]]*)\\]\\([^)]*\\)/g, '$1')
                      .replace(/<[^>]+>/g, '')
                      .replace(/[`*_~]/g, '')
                      .replace(/\\s+/g, ' ').trim();
            }}
            // Retry briefly: Streamlit re-renders content in fits and
            // starts, so the heading may not exist in the DOM yet.
            var tries = 0;
            (function poll() {{
              if (tries > 120) return;
              tries++;
              var heads = scroller.querySelectorAll('h1,h2,h3,h4,h5,h6');
              for (var i = 0; i < heads.length; i++) {{
                var h = heads[i];
                if (parseInt(h.tagName.charAt(1), 10) !== lvl) continue;
                if (norm(h.textContent || '') !== title) continue;
                var rel = h.getBoundingClientRect().top
                          - scroller.getBoundingClientRect().top;
                scroller.scrollTop += rel - 24;
                h.style.transition = 'background-color 1s';
                h.style.backgroundColor = 'rgba(214, 40, 40, 0.14)';
                setTimeout(function () {{
                  h.style.backgroundColor = '';
                }}, 1800);
                return;
              }}
              setTimeout(poll, 50);
            }})();
          }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
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

    # Permanently bump body-text + code font size. Scoped to Streamlit's
    # markdown/code containers so widget labels, buttons, and headers keep their
    # default sizes; chat-message bodies use the same containers, so the chat
    # panel is covered too.
    st.markdown(_BODY_FONT_SIZE_CSS, unsafe_allow_html=True)

    # .md is authored content → render as markdown; anything else → code block.
    # unsafe_allow_html lets HTML inline markup like <b>bold</b> / <strong>…
    # render alongside standard **bold** markdown — without it Streamlit strips
    # the tags and shows their inner text unstyled.
    if target.endswith(".md"):
        st.markdown(text, unsafe_allow_html=True)
    else:
        st.code(text, language="text")

    # A sidebar-table-of-contents jump, if one was staged: scroll to the
    # heading, then drop the staged target so the next rerun doesn't re-jump.
    jump = st.session_state.get(_TOC_JUMP)
    if jump:
        _inject_toc_jump(jump)
        st.session_state.pop(_TOC_JUMP, None)

    # --- Quote a passage into the chat ---------------------------------
    # The content above is read-only DOM: Streamlit never sees the browser's
    # text selection. So the user selects text, copies it (⌘C / Ctrl-C), pastes
    # it here, and "Send to chat" stages it for the next question in the chat
    # panel — alongside the full document, which is always sent as context.
    st.divider()
    with st.expander("Quote a passage for the LLM chat", expanded=False):
        _sess_label = docs.chat_session_label(
            docs.active_document(), docs.active_chat(docs.active_document())
        )
        st.caption(
            "_Select text above, copy it, paste it here, then **Send to chat**. "
            f"The quote is attached to your next question in "
            f"**{_sess_label}** — the full document is still sent as context "
            "too._"
        )
        st.text_area(
            "Quote for chat",
            value=st.session_state.get(_reader_quote_key(), ""),
            height=120, key=_reader_quote_area_key(),
            placeholder="Paste the passage you want to ask about…",
        )
        col_send, col_clear = st.columns([1, 1])
        if col_send.button(
            "Send to chat", type="primary",
            help="Stage this quote for the next chat question, then switch to "
                 "the LLM chat tab.",
        ):
            quote = (st.session_state.get(_reader_quote_area_key()) or "").strip()
            if quote:
                st.session_state[_reader_quote_key()] = quote
                st.session_state[TABS_KEY] = CHAT_TAB_LABEL
                st.rerun()
            else:
                st.warning("Paste a passage into the box first.")
        if col_clear.button(
            "Clear", help="Drop the staged quote so it is no longer attached.",
        ):
            st.session_state.pop(_reader_quote_key(), None)
            st.session_state[_reader_quote_area_key()] = ""
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
    """Drop the active document (wired as the "Clear" button's on_click).

    In multi-document mode the document is removed from the registry — its
    chat state is dropped and the next open document becomes active, so a
    passage quoted from one document can't leak into a chat about another. In
    single-document mode the reader target (and any staged quote) is cleared.
    """
    doc = docs.active_document()
    if doc:
        docs.remove_document(doc)
    else:
        st.session_state.pop(_READER_TARGET, None)
        # In single-document mode, clear the staged quotes of every chat
        # session of this (now closed) document.
        for sid in docs.chat_sessions(""):
            st.session_state.pop(docs.chat_key(_READER_QUOTE, sid, ""), None)
            st.session_state.pop(docs.chat_key(_READER_QUOTE_AREA, sid, ""), None)
