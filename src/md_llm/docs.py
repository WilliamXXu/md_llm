"""Optional multi-document session: an ordered registry of open documents,
each with its own independent LLM chat.

By default md_llm is single-document: one Reader target, one chat
conversation, all under the legacy bare session-state keys (``_reader_target``,
``_chat_messages``, …). A host opts into multi-document mode by opening files
with ``open_in_reader(relpath, keep_open=True)`` (or calling
:func:`add_document` directly). From then on:

  * every open document keeps its own conversation (``_chat_messages``),
    background stream task (``_chat_bg_task``), last chat error, and staged
    Reader quote — namespaced via :func:`doc_key` as
    ``<base>__doc__<relpath>`` so nothing leaks across documents;
  * a file can be open at most once — :func:`add_document` refuses a second
    copy of an already-open file (a different relpath resolving to it, e.g.
    ``./notes.md`` vs ``notes.md``), activates the existing document, and
    pops a warning dialog (:func:`_warn_already_open`);
  * the **active** document decides what the Reader shows and which
    conversation the chat tab operates on;
  * :func:`render_doc_selector` / :func:`render_doc_buttons` render the
    sidebar picker (an active-document selectbox + Close button, or one
    switch/Close button pair per open document) that hosts mount next to
    their file picker.

On top of the document registry sits a per-document **chat-session registry**
(:func:`chat_sessions` / :func:`add_chat` / :func:`remove_chat`): any document
can have several independent chat sessions ("tabs"), each with its own
conversation, background stream task, last error, and staged Reader quote.
Sessions are keyed via :func:`chat_key` as ``<base>__chat__<id>__doc__<relpath>``
(the first session deliberately keeps the document's legacy keys, so existing
single-chat sessions are untouched). The session registry is session-memory
only and is dropped with its document.

The registry is session-memory only. Closing the last open document — or
calling ``open_in_reader(relpath)`` without ``keep_open`` — returns the
session to single-document mode and its legacy keys, so hosts that stage files
one at a time keep today's exact behaviour.

Decoupling: this module knows nothing about reader/chat internals. It shares
the reader's ``_reader_target`` key by literal string (the same string-literal
convention ``_reader_quote`` already uses between reader.py and chat.py), and
it cleans up per-document keys by their ``__doc__<relpath>`` suffix rather than
naming them individually.
"""

from __future__ import annotations

import os

import streamlit as st

from .core import get_core
from .state import _display_name_for_filepath

# Session-state keys (session-memory only; nothing is persisted).
_OPEN_DOCS = "_md_llm_open_docs"    # ordered dict: relpath -> None
_ACTIVE_DOC = "_md_llm_active_doc"  # relpath of the active document
_DOC_SELECT = "_md_llm_doc_sel"     # sidebar selectbox widget key
# Non-widget mirror of the document the selectbox last rendered (see
# render_doc_selector — it tells a user click apart from an external change).
_DOC_SELECT_LAST = "_md_llm_doc_sel_last"
# Flag set when the permanent "(no document)" placeholder is the active
# context. While set, active_document() returns None (so the chat falls back
# to its legacy bare keys — an independent, context-free conversation) and the
# Reader target is cleared. Survives having real documents open alongside.
_NO_DOC_ACTIVE = "_md_llm_no_doc_active"

# Separator between a base session key and the document relpath in
# per-document keys (e.g. ``_chat_messages__doc__notes.md``). Chosen so a
# namespaced key can never collide with a legacy key and is easy to spot in
# the session-state browser.
_DOC_KEY_SEP = "__doc__"

# Separator between a base session key and a chat-session id in per-session
# keys (e.g. ``_chat_messages__chat__2__doc__notes.md``). Placed BEFORE the
# document suffix, so every session key still ends with ``__doc__<relpath>``
# and closing a document (``_drop_doc_keys``) sweeps all of its sessions.
# Session 1 of a document deliberately carries NO ``__chat__`` segment — its
# keys are exactly the document-scoped legacy keys, so a pre-existing
# single-chat session is untouched by this feature.
_CHAT_KEY_SEP = "__chat__"

# Per-document chat-session registry (session-memory only, dropped with the
# document). ``_md_llm_chat_sessions__doc__<relpath>`` holds an ordered
# ``{sid: label}`` dict; ``_md_llm_chat_active__doc__<relpath>`` holds the id
# of the active session. Session ids are ints starting at 1.
_CHAT_SESSIONS = "_md_llm_chat_sessions"
_CHAT_ACTIVE = "_md_llm_chat_active"


def doc_key(base_key, doc_id):
    """Session-state key for one document-scoped value.

    ``doc_id`` None/"" (single-document mode) maps to the bare ``base_key``,
    so pre-multi-doc sessions keep their exact legacy keys; otherwise
    ``<base_key>__doc__<doc_id>`` isolates one document's conversation, staged
    quote, and stream task from every other open document's.
    """
    if not doc_id:
        return base_key
    return f"{base_key}{_DOC_KEY_SEP}{doc_id}"


def chat_key(base_key, chat_id, doc_id):
    """Session-state key for one chat session of one document.

    ``chat_id`` 1 (or falsy) maps to the document-scoped key — exactly the
    legacy keys a single-chat session used before multi-chat support existed —
    and to the bare ``base_key`` in single-document mode. Additional sessions
    add a ``__chat__<id>`` segment between the base and the (optional)
    ``__doc__<relpath>`` suffix, so every session's conversation, stream task,
    staged quote, and last error are isolated from every other session's.
    """
    if not chat_id or chat_id == 1:
        return doc_key(base_key, doc_id)
    key = f"{base_key}{_CHAT_KEY_SEP}{chat_id}"
    if doc_id:
        key += f"{_DOC_KEY_SEP}{doc_id}"
    return key


def open_documents():
    """Ordered relpaths of the open documents ([] in single-document mode)."""
    reg = st.session_state.get(_OPEN_DOCS)
    return list(reg) if isinstance(reg, dict) else []


def is_multi():
    """True when at least one document is open (multi-document mode)."""
    return bool(open_documents())


def is_no_doc_active():
    """True when the permanent "(no document)" placeholder is the active context.

    While True the chat runs without any markdown context (its conversation
    lives under the legacy bare keys) even if real documents are open.
    """
    return bool(st.session_state.get(_NO_DOC_ACTIVE))


def activate_no_document():
    """Switch to the "(no document)" context.

    Sets the no-doc flag (so :func:`active_document` returns None and the chat
    uses its context-free bare-key conversation), clears the Reader target, and
    jumps to the LLM chat view (there is nothing to read). Leaves the document
    registry untouched, so switching back to a real document is one click away.
    """
    st.session_state[_NO_DOC_ACTIVE] = True
    st.session_state.pop("_reader_target", None)


def _doc_identity(rel):
    """Hashable identity of the file behind ``rel`` (duplicate detection).

    An existing file is identified by its ``(st_dev, st_ino)`` pair, so the
    same file reached through any relpath (``./notes.md``, an absolute path,
    a symlink, …) collides while different files never do. A missing file
    falls back to its normalized absolute path; without an injected Core
    (unit tests) the raw string is normalized instead.
    """
    try:
        base = os.path.abspath(get_core().base_dir)
    except RuntimeError:
        return ("raw", os.path.normpath(rel))
    full = os.path.abspath(os.path.join(base, rel))
    try:
        st_ = os.stat(full)
    except OSError:
        return ("path", os.path.realpath(full))
    return ("ino", st_.st_dev, st_.st_ino)


def _find_duplicate_document(rel):
    """Relpath of an already-open document that IS the file behind ``rel``.

    None when ``rel`` isn't already open — including the exact-match case
    (``rel`` itself in the registry), which is :func:`add_document`'s
    documented idempotent re-open, not a duplicate.
    """
    ident = _doc_identity(rel)
    for other in open_documents():
        if other != rel and _doc_identity(other) == ident:
            return other
    return None


def add_document(rel):
    """Register ``rel`` as an open document and make it the active one.

    Returns the doc id (``rel``, or "" when falsy). Idempotent: re-opening an
    already-open file keeps its position and its existing conversation. A
    DIFFERENT relpath resolving to an already-open file (``./notes.md`` vs
    ``notes.md``, an absolute path, a symlink) is refused: no second copy is
    registered — the existing document is activated and the warning dialog
    :func:`_warn_already_open` explains why.
    """
    if not rel:
        return ""
    st.session_state.pop(_NO_DOC_ACTIVE, None)  # leaving no-doc context
    dup = _find_duplicate_document(rel)
    if dup is not None:
        set_active_document(dup)
        _warn_already_open(dup)
        return dup
    reg = st.session_state.get(_OPEN_DOCS)
    if not isinstance(reg, dict):
        reg = {}
        st.session_state[_OPEN_DOCS] = reg
    reg.setdefault(rel, None)
    set_active_document(rel)
    return rel


def active_document():
    """Relpath of the active document, or None in single-document mode.

    Returns None when the "(no document)" placeholder is active
    (:func:`is_no_doc_active`), even if real documents are open — the chat then
    runs without markdown context under its legacy bare keys.

    Self-healing: if the stored active doc is no longer registered (closed
    elsewhere), falls back to the first still-open document.
    """
    if st.session_state.get(_NO_DOC_ACTIVE):
        return None
    reg = st.session_state.get(_OPEN_DOCS)
    if not isinstance(reg, dict) or not reg:
        return None
    rel = st.session_state.get(_ACTIVE_DOC)
    if rel in reg:
        return rel
    rel = next(iter(reg))
    st.session_state[_ACTIVE_DOC] = rel
    return rel


def set_active_document(rel):
    """Make ``rel`` the active document (the Reader and chat tab follow it).

    Also mirrors it into the reader's ``_reader_target`` key (by literal
    string — the same decoupling convention ``_reader_quote`` uses) so the
    Reader always shows the active document. Clears the "(no document)" flag
    so :func:`active_document` reports the real document.
    """
    st.session_state.pop(_NO_DOC_ACTIVE, None)
    rel = rel or ""
    st.session_state[_ACTIVE_DOC] = rel
    if rel:
        st.session_state["_reader_target"] = rel
    else:
        st.session_state.pop("_reader_target", None)


def remove_document(rel):
    """Close ``rel``: drop its per-document state and activate a fallback.

    With documents remaining, the next one becomes active (and the Reader
    follows). Closing the last one returns the session to single-document mode
    and clears the reader target.
    """
    reg = st.session_state.get(_OPEN_DOCS)
    if not isinstance(reg, dict):
        return
    reg.pop(rel, None)
    _drop_doc_keys(rel)
    if not reg:
        st.session_state.pop(_OPEN_DOCS, None)
        st.session_state.pop(_ACTIVE_DOC, None)
        st.session_state.pop(_DOC_SELECT, None)
        st.session_state.pop(_DOC_SELECT_LAST, None)
        st.session_state.pop(_NO_DOC_ACTIVE, None)
        st.session_state.pop("_reader_target", None)
        st.session_state.pop("_reader_quote", None)
        return
    # Don't disturb the "(no document)" context if it was active — the user
    # may be closing documents while chatting without context.
    if not st.session_state.get(_NO_DOC_ACTIVE) \
            and st.session_state.get(_ACTIVE_DOC) == rel:
        set_active_document(next(iter(reg)))


def reset_documents():
    """Leave multi-document mode: drop the registry, active doc, and selector.

    Used by ``open_in_reader(relpath)`` (without ``keep_open``) so a host that
    stages files one at a time keeps today's exact single-document behaviour.
    """
    st.session_state.pop(_OPEN_DOCS, None)
    st.session_state.pop(_ACTIVE_DOC, None)
    st.session_state.pop(_DOC_SELECT, None)
    st.session_state.pop(_DOC_SELECT_LAST, None)
    st.session_state.pop(_NO_DOC_ACTIVE, None)


# ---------------------------------------------------------------------------
# Chat sessions: several independent conversations per document
# ---------------------------------------------------------------------------
#
# A document's chat tab can hold any number of sessions ("tabs"): each session
# has its own conversation history, background stream task, last error, and
# staged Reader quote, keyed via chat_key(). Session ids are ints; id 1 keeps
# the document's legacy keys so existing single-chat sessions survive. The
# registry is session-memory only and is dropped together with the document
# (its keys end with ``__doc__<relpath>``, so remove_document sweeps them).


def _chat_sessions_key(doc_id):
    """Session key of the chat-session registry for ``doc_id``."""
    return doc_key(_CHAT_SESSIONS, doc_id)


def _chat_active_key(doc_id):
    """Session key of the active chat-session id for ``doc_id``."""
    return doc_key(_CHAT_ACTIVE, doc_id)


def _session_registry(doc_id, create=False):
    """The ``{sid: label}`` registry for ``doc_id``, or {} when unset.

    With ``create=True`` an absent registry is seeded as the default
    single-session one ({1: "Chat 1"}).
    """
    reg = st.session_state.get(_chat_sessions_key(doc_id))
    if not isinstance(reg, dict) or not reg:
        if create:
            reg = {1: "Chat 1"}
            st.session_state[_chat_sessions_key(doc_id)] = reg
        else:
            return {}
    return reg


def chat_sessions(doc_id):
    """Ordered chat-session ids for ``doc_id`` ([1] by default).

    A document always has at least one session — the default one, whose keys
    are the document's legacy keys. Further sessions are created with
    :func:`add_chat`.
    """
    return list(_session_registry(doc_id)) or [1]


def chat_session_label(doc_id, chat_id):
    """Display label for one chat session (e.g. "Chat 2")."""
    return _session_registry(doc_id).get(chat_id) or f"Chat {chat_id}"


def active_chat(doc_id):
    """Id of the active chat session for ``doc_id``; falls back to the first.

    Self-healing like :func:`active_document`: when the stored id is no longer
    registered (the session was closed elsewhere), the first remaining session
    becomes active.
    """
    sids = chat_sessions(doc_id)
    cur = st.session_state.get(_chat_active_key(doc_id))
    if cur in sids:
        return cur
    cur = sids[0]
    st.session_state[_chat_active_key(doc_id)] = cur
    return cur


def set_active_chat(chat_id, doc_id):
    """Make ``chat_id`` the active chat session for ``doc_id``."""
    st.session_state[_chat_active_key(doc_id)] = chat_id


def add_chat(doc_id):
    """Open a new chat session for ``doc_id`` and make it active.

    Returns the new session id. Sessions are numbered 1, 2, 3, …; the first
    keeps the document's legacy keys (see :func:`chat_key`), later ones live
    under ``__chat__<id>`` keys so they never collide.
    """
    reg = _session_registry(doc_id, create=True)
    new_id = max(reg) + 1
    reg[new_id] = f"Chat {new_id}"
    st.session_state[_chat_sessions_key(doc_id)] = reg
    set_active_chat(new_id, doc_id)
    return new_id


def remove_chat(chat_id, doc_id):
    """Close a chat session for ``doc_id`` and drop its state.

    The last remaining session can't be closed (a document always has one);
    otherwise the session's conversation, stream task, staged quote, and last
    error are dropped, and — if it was active — the first remaining session
    becomes active.
    """
    reg = _session_registry(doc_id)
    if chat_id not in reg or len(reg) <= 1:
        return
    reg.pop(chat_id)
    st.session_state[_chat_sessions_key(doc_id)] = reg
    _drop_chat_keys(chat_id, doc_id)
    if st.session_state.get(_chat_active_key(doc_id)) == chat_id:
        set_active_chat(next(iter(reg)), doc_id)


# The chat-session state bases that get session-scoped (the conversation, its
# background stream task, the transient last error, and the Reader's staged
# quote + quote textarea). Session 1's keys are exactly these document-scoped
# / bare keys, so closing session 1 drops them by name in single-document mode
# (where no ``__doc__`` suffix exists to match on).
_CHAT_SCOPED_BASES = (
    "_chat_messages",
    "_chat_bg_task",
    "_chat_last_error",
    "_reader_quote",
    "_reader_quote_area",
)


def _drop_chat_keys(chat_id, doc_id):
    """Remove every session-state key belonging to one chat session.

    For sessions 2+ the keys share the ``__chat__<id>`` segment, so they are
    matched by suffix. Session 1 uses the document-scoped (legacy) keys, so
    they are matched by the ``__doc__<relpath>`` suffix — or, in
    single-document mode, dropped by name. Two key families are never dropped
    here: the ``_md_llm_*`` management keys (the session registry and the
    chat-panel selector mirrors must survive — other sessions of the document
    depend on them), and — when dropping session 1 — the other sessions'
    ``__chat__`` keys, which share the same ``__doc__`` suffix.
    """
    if chat_id == 1:
        if not doc_id:
            for base in _CHAT_SCOPED_BASES:
                st.session_state.pop(base, None)
            return
        suffix = f"{_DOC_KEY_SEP}{doc_id}"
    else:
        suffix = f"{_CHAT_KEY_SEP}{chat_id}"
        if doc_id:
            suffix += f"{_DOC_KEY_SEP}{doc_id}"
    for k in [
        k for k in st.session_state
        if isinstance(k, str) and k.endswith(suffix)
        and not k.startswith("_md_llm_")
        and ("__chat__" not in k if chat_id == 1 else True)
    ]:
        st.session_state.pop(k, None)


def _drop_doc_keys(rel):
    """Remove every session key namespaced to ``rel`` (``<base>__doc__<rel>``).

    A closed document must not resurrect its old conversation when re-opened.
    Matching by the ``__doc__<rel>`` suffix covers all per-document keys (chat
    messages, stream task, staged quote, quote box) without this module naming
    each one individually.
    """
    suffix = f"{_DOC_KEY_SEP}{rel}"
    for k in [
        k for k in st.session_state
        if isinstance(k, str) and k.endswith(suffix)
    ]:
        st.session_state.pop(k, None)


# --- sidebar picker ---------------------------------------------------------

def _resolve_doc_path(rel):
    """Absolute path for a staged relpath (or the path itself if absolute)."""
    if not rel:
        return None
    full = rel if os.path.isabs(rel) else os.path.join(get_core().base_dir, rel)
    return os.path.abspath(full)


def _doc_display_name(rel):
    """Display name for one open document (sidecar title or filename stem).

    The registry stores relpaths, so resolve against ``core.base_dir`` first —
    the sidecar-title lookup in ``_display_name_for_filepath`` only works on a
    real path. Files that no longer exist are flagged so a stale entry is
    obvious.
    """
    full = _resolve_doc_path(rel)
    name = _display_name_for_filepath(full or rel)
    if full and not os.path.isfile(full):
        return f"(missing) {name}"
    return name


def doc_chat_has_messages(rel):
    """True when any chat session of ``rel`` holds at least one message.

    Used by :func:`close_document` to warn before a close wipes a
    conversation. The message base key is shared with chat.py by literal
    string (the same convention ``_reader_quote`` uses).
    """
    for sid in chat_sessions(rel):
        if st.session_state.get(chat_key("_chat_messages", sid, rel)):
            return True
    return False


@st.dialog("Document already open")
def _warn_already_open(rel):
    """Modal warning that a second copy of an open file was refused."""
    st.warning(
        f"**{_doc_display_name(rel)}** is already open — a document can't be "
        "opened twice. The existing copy is now active."
    )
    if st.button("OK", type="primary"):
        st.rerun()


@st.dialog("Non-empty LLM chat")
def _confirm_close_document(rel):
    """Modal asking to proceed with closing a non-empty chat document."""
    st.warning(
        f"Do you want to proceed with a non-empty LLM chat?\n\n"
        f"Closing **{_doc_display_name(rel)}** removes its conversation "
        "from the session."
    )
    col_proceed, col_cancel = st.columns(2)
    if col_proceed.button("Close anyway", type="primary"):
        remove_document(rel)
        st.rerun()
    if col_cancel.button("Cancel"):
        st.rerun()


def close_document(rel):
    """Close ``rel`` from a button click, guarding a non-empty chat.

    With any chat session of the document holding messages, a confirmation
    dialog (:func:`_confirm_close_document`) runs first and the document is
    only closed when the user proceeds; otherwise the document closes
    immediately, exactly like ``remove_document`` + rerun before.
    """
    if doc_chat_has_messages(rel):
        _confirm_close_document(rel)
        return
    remove_document(rel)
    st.rerun()


def render_doc_selector():
    """Sidebar picker over the open documents; no-op in single-document mode.

    A selectbox (labelled with each document's display name) chooses the
    active document — the Reader and the chat tab follow it — and a Close
    button removes the selected document and its independent chat. Meant for
    the host's sidebar, next to its file picker; the standalone demo mounts it
    there.
    """
    docs_ = open_documents()
    if not docs_:
        return
    cur = active_document()

    # Keep the selectbox in sync with the active document when it changed
    # WITHOUT this selectbox being the cause (a fresh upload, the Reader's
    # close button, …). The distinction matters because a keyed selectbox
    # restores its own widget value, and a frontend click is applied to
    # session_state BEFORE this code runs — blindly forcing the widget value
    # to ``cur`` would swallow the user's click. The last-rendered mirror
    # tells the two apart:
    #   * user click  -> the widget value changed, ``cur`` did not
    #   * external    -> ``cur`` moved, the widget value did not
    # So only sync when ``cur`` moved. This pre-mount session-state write is
    # also the widget's default-value mechanism — the selectbox below passes
    # no ``index=``, because combining ``index=`` with a session-state set
    # trips Streamlit's "created with a default value but also had its value
    # set via the Session State API" policy warning.
    if cur != st.session_state.get(_DOC_SELECT_LAST):
        st.session_state[_DOC_SELECT] = cur

    st.caption("Open documents — the active one is shown below:")
    sel = st.selectbox(
        "Open documents",
        docs_,
        format_func=_doc_display_name,
        key=_DOC_SELECT,
        label_visibility="collapsed",
    )
    st.session_state[_DOC_SELECT_LAST] = cur
    if sel != cur:
        set_active_document(sel)
        st.rerun()

    if st.button("Close active"):
        close_document(sel)


def render_doc_buttons():
    """Open-document switch buttons with a permanent "(no document)" entry.

    An alternative to the sidebar selectbox (:func:`render_doc_selector`) for
    hosts that prefer buttons. The first row is always the "(no document)"
    placeholder — it can't be closed (its Close button is disabled) and
    clicking it switches to a context-free LLM chat (no markdown sent to the
    model). It is highlighted when active, stays put when documents are
    opened, and carries its own independent conversation.

    One button per open document follows — the active one highlighted — each
    switching the Reader and the chat tab to that document and jumping to the
    Reader view, with its own Close button. The sidebar selectbox can still be
    mounted alongside.
    """
    no_doc = is_no_doc_active()
    docs_ = open_documents()
    cur = active_document()

    # --- Permanent "(no document)" placeholder (always first, never closeable)
    col_switch, col_close = st.columns([5, 1])
    if col_switch.button(
        "(no document, direct LLM chat)",
        key="_md_llm_doc_btn_none",
        type="primary" if no_doc or not cur else "secondary",
        use_container_width=True,
    ):
        activate_no_document()
        st.session_state["_app_tabs"] = "LLM chat"
        st.rerun()
    col_close.button(
        "✕",
        key="_md_llm_doc_close_none",
        disabled=True,
    )

    # --- One switch + close row per open document -------------------------
    for rel in docs_:
        col_switch, col_close = st.columns([5, 1])
        if col_switch.button(
            _doc_display_name(rel),
            key=f"_md_llm_doc_btn_{rel}",
            type="primary" if rel == cur and not no_doc else "secondary",
            use_container_width=True,
        ):
            set_active_document(rel)
            # Literal strings, same convention as the "_reader_target" write
            # above: switching the active view to the Reader tab.
            st.session_state["_app_tabs"] = "Reader"
            st.rerun()
        if col_close.button(
            "✕",
            key=f"_md_llm_doc_close_{rel}",
        ):
            close_document(rel)
