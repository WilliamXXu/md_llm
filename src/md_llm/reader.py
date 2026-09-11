"""Reader panel: open any document (markdown/text) as a clean, full-text view.

A host stages a file by calling :func:`open_in_reader` (which records the target
in session state and switches the host's active tab here). With ``keep_open=True``
the file joins the open-documents registry and becomes the active one — each
open document gets its own Reader view and an independent LLM chat (see
:mod:`md_llm.docs`). The panel renders the file, offers copy-to-clipboard and
a one-click **⚡ Summarize** quick action (left of Copy: opens a new "Summary"
tab in the LLM chat and sends the whole document there with an editable
summary prompt).

In-place editing is gated behind a **safety lock** (the "🔒 Edit lock" toggle in
the action row): the lock is ON by default — every document starts read-only —
and turning it off keeps the document fully rendered and makes it editable in
place: every rendered block (heading, paragraph, list, table, quote, fence…)
grows a hover "✎" handle; clicking one swaps just that block for a small
source editor (Done / Cancel) while the rest of the document stays rendered
around it, and committing splices the block's source back into the draft. A
collapsed "Raw source" expander keeps the whole-file textarea for bulk
restructuring; text files keep a plain split editor. Saving writes the draft
through the same path guard that gates opening, atomically
(:func:`md_llm.state._write_text`), and confirms first when the file changed
on disk since the draft was opened. The draft is per-document session state;
closing a document that still differs from disk asks before discarding it.
Note for the bundled app: files staged by the macOS Finder droplet / file
uploader are working copies under its uploads dir, so editing there applies to
the staged copy — the caption above the content always names the exact file a
save would overwrite.

Path safety: the staged relpath is resolved against ``core.base_dir`` and the
resulting absolute path must sit inside one of ``core.markdown_dirs``; anything
that escapes via ``..`` is rejected before being read (or written).

Shared session-state keys (the integration contract the host honors):
  - ``_reader_target``  — the relpath to display (written by open_in_reader).
  - ``TABS_KEY`` / ``READER_TAB_LABEL`` / ``CHAT_TAB_LABEL`` — tab switching.
"""

from __future__ import annotations

import base64
import json
import os
import re

import streamlit as st
import streamlit.components.v1 as components
from markdown_it import MarkdownIt

from . import docs
from .core import get_core
from .state import (
    _BODY_FONT_SIZE_CSS,
    _display_name_for_filepath,
    _escape_currency_dollars,
    _human_size,
    _read_text,
    _write_text,
)

# Session-state key holding the reader target (a relpath against core.base_dir).
_READER_TARGET = "_reader_target"

# --- in-place editing (safety-locked; see the module docstring) --------------
#
# All keys are per-document via docs.doc_key (absent = the legacy bare key in
# single-document mode), so a closed document's draft is swept together with
# its chat state by docs._drop_doc_keys' __doc__-suffix match. Widget keys are
# pruned by Streamlit on Reader↔chat view switches; their non-widget mirrors
# survive, and each widget re-seeds from its mirror on remount — the same
# trick as the quick-prompt editor below.
#
# The lock itself is ON by default: the unlocked flag's mirror key is ABSENT
# for a locked document, so a fresh session (or a fresh document) can never
# start editable.
_EDIT_UNLOCKED = "_reader_edit_unlocked"      # non-widget mirror (absent = locked)
_EDIT_LOCK_TOGGLE = "_reader_edit_lock"       # st.toggle widget key
_EDIT_AREA = "_reader_edit_area"              # whole-file st.text_area widget key
_EDIT_DRAFT = "_reader_edit_draft"            # non-widget mirror of the draft
_EDIT_BASE_MTIME = "_reader_edit_base_mtime"  # mtime the draft was seeded from
_EDIT_PENDING = "_reader_edit_pending"        # draft stashed for the conflict dialog
_EDIT_BLOCK_INDEX = "_reader_edit_block_index"  # doc-scoped: block open in the
                                                # in-place editor (absent = none)
_EDIT_BLOCK_AREA = "_reader_edit_block_area"    # doc-scoped: the in-place block
                                                # editor's st.text_area widget key
_EDIT_AREA_GEN = "_reader_edit_area_gen"        # doc-scoped: raw textarea key
                                                # generation (see _bump_raw_area)

# Above this many characters the editor shows a "may be slow" notice (the
# text_area widget degrades on very large documents).
_EDIT_SIZE_WARN = 400_000

# CSS for the in-place block editor. Each rendered block sits in a keyed
# container (`_reader_blk_<i>`) made position:relative; its "✎" handle
# (`_reader_blk_btn_<i>`) is absolutely positioned in the block's top-right
# corner and invisible until the block is hovered/focused, so the document
# reads exactly like the locked view until the reader reaches for a block.
# The blocks area also tightens Streamlit's inter-element gap so stacked
# blocks keep the whole-document rhythm.
_BLOCKS_CSS = """
[class*="st-key-_reader_blocks_area"] [data-testid="stVerticalBlock"] {
    gap: 0.3rem !important;
}
[class*="st-key-_reader_blk_"] { position: relative; }
[class*="st-key-_reader_blk_btn_"] {
    position: absolute !important;
    top: 0.05rem;
    right: 0;
    z-index: 20;
    opacity: 0;
    transition: opacity 0.12s ease-in-out;
}
[class*="st-key-_reader_blk_btn_"] button {
    min-height: 1.35rem !important;
    height: 1.35rem !important;
    padding: 0 0.45rem !important;
    margin: 0 !important;
    font-size: 0.7rem !important;
    line-height: 1.2 !important;
}
[class*="st-key-_reader_blk_btn_"] button > div { padding: 0 !important; }
[class*="st-key-_reader_blk_"]:hover [class*="st-key-_reader_blk_btn_"],
[class*="st-key-_reader_blk_"]:focus-within [class*="st-key-_reader_blk_btn_"] {
    opacity: 1;
}
"""

# The ⚡ Summarize quick action's prompt, staged for the NEXT chat turn of the
# ACTIVE document + chat session. Shared by string literal (the chat panel
# reads "_reader_quick_prompt" too) rather than an import, to keep reader↔chat
# decoupled. The chat panel pops it and sends it through the normal chat
# pipeline (see chat._send_staged_quick_prompt).
_READER_QUICK_PROMPT = "_reader_quick_prompt"

# The ⚡ Summarize quick action's default prompt. The user can edit it in the
# Reader's "Quick summarize prompt" expander; this constant is the factory
# default the editor seeds and its "Reset to default" restores.
QUICK_SUMMARY_PROMPT = (
    "你是一个摘要助手，请概括给定文本。规则：1. 若原文为英文，则用英文输出；"
    "否则一律用简体中文。2. 平衡原文的行文顺序和你的结构化输出，考虑使用bullet "
    "points/表格/executive summary 3. 保持客观，不得添加原文没有的信息"
)

# The quick-prompt editor's widget key. Streamlit prunes unmounted widget keys
# on Reader↔chat view switches, so edits are mirrored into _QUICK_PROMPT_SAVED
# (an ordinary, non-pruned session key) by an on_change callback; on remount
# the editor re-seeds from the mirror — the same trick as the chat panel's
# control snapshot (_chat_controls_snapshot).
_QUICK_PROMPT_EDIT = "_reader_quick_prompt_edit"
_QUICK_PROMPT_SAVED = "_reader_quick_prompt_saved"

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

def _reader_quick_prompt_key():
    """Session key of the ⚡ Summarize prompt staged for the ACTIVE session.

    Scoped per document AND per chat session, so a prompt staged from one
    document's Reader can never fire into another document's conversation
    (chat.py resolves the same key the same way). The ⚡ button stages into the
    chat session it just opened for the summary (docs.add_chat activates it),
    so reader and chat always agree on the target.
    """
    doc = docs.active_document()
    return docs.chat_key(_READER_QUICK_PROMPT, docs.active_chat(doc), doc)


def _save_quick_prompt_edit():
    """on_change for the quick-prompt editor: mirror the edit into the mirror key.

    The mirror (_QUICK_PROMPT_SAVED) is a non-widget key, so it survives the
    view switches that prune the textarea itself; the ⚡ button reads it.
    """
    st.session_state[_QUICK_PROMPT_SAVED] = (
        st.session_state.get(_QUICK_PROMPT_EDIT, "")
    )


def _current_quick_prompt():
    """The prompt the ⚡ Summarize button will send.

    The edited copy when there is one, else the factory default. Stripped;
    empty means "cleared by the user — refuse to send".
    """
    return (
        st.session_state.get(_QUICK_PROMPT_SAVED) or QUICK_SUMMARY_PROMPT
    ).strip()


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


def _current_edit_doc():
    """The doc id the editor keys are namespaced under: the active document.

    Deliberately ``docs.active_document()`` — NOT the ``_READER_TARGET``
    mirror: in single-document mode the registry is empty and every scoped
    feature (chat messages, staged quick prompt) uses its legacy BARE key,
    so the editor must too. Keying by the target's (always-truthy) relpath
    would mint ``__doc__``-suffixed keys that the single-document close path
    never cleans up.
    """
    return docs.active_document() or ""


def _edit_unlocked(rel):
    """True when the safety lock of ``rel`` is off (in-place editing enabled).

    Absent mirror key = locked, which is the default for every fresh session
    and every newly opened document.
    """
    return bool(st.session_state.get(docs.doc_key(_EDIT_UNLOCKED, rel)))


def _mirror_edit_lock():
    """on_change for the lock toggle: mirror "unlocked" into the non-widget key.

    Runs before the script on toggle flips, so the same rerun already renders
    the editor (unlock) or the read-only view (re-lock). Re-locking keeps the
    draft: unlocking again resumes where the editing stopped.
    """
    rel = _current_edit_doc()
    if st.session_state.get(docs.doc_key(_EDIT_LOCK_TOGGLE, rel)):
        st.session_state.pop(docs.doc_key(_EDIT_UNLOCKED, rel), None)
    else:
        st.session_state[docs.doc_key(_EDIT_UNLOCKED, rel)] = True


def _raw_area_key(rel):
    """Session key of the whole-file textarea's CURRENT generation.

    The generation is part of the widget key (before the ``__doc__`` suffix,
    so the close-time sweep still matches) — see :func:`_bump_raw_area` for
    why it exists.
    """
    gen = st.session_state.get(docs.doc_key(_EDIT_AREA_GEN, rel), 0)
    return docs.doc_key(f"{_EDIT_AREA}__g{gen}", rel)


def _bump_raw_area(rel):
    """Rotate the whole-file textarea's widget key after a programmatic reset.

    Streamlit widgets belong to the browser once mounted: a server-side
    re-seed of a mounted widget's key does not reach the client, and the
    client's next widget-state snapshot silently reverts the server value —
    replaying the OLD whole-file text and, through the on_change mirror,
    clobbering the block edits in the draft. Rotating the key forces a
    remount, so the textarea comes back displaying the fresh draft and the
    client can never replay a stale copy.
    """
    old_key = _raw_area_key(rel)
    key = docs.doc_key(_EDIT_AREA_GEN, rel)
    st.session_state[key] = st.session_state.get(key, 0) + 1
    st.session_state.pop(old_key, None)


def _mirror_edit_draft():
    """on_change for the whole-file textarea: mirror the edit into the draft key.

    Streamlit fires widget callbacks only after the widget's new value is in
    session state, so the mirror is always at least as fresh as the last
    committed edit — which is what the dirty checks (that run in widget order,
    possibly BEFORE the textarea re-instantiates) rely on. A whole-file edit
    also closes any open block editor: the raw text just replaced the draft
    that editor was splicing into.
    """
    rel = _current_edit_doc()
    st.session_state[docs.doc_key(_EDIT_DRAFT, rel)] = st.session_state.get(
        _raw_area_key(rel), ""
    )
    st.session_state.pop(docs.doc_key(_EDIT_BLOCK_INDEX, rel), None)
    st.session_state.pop(docs.doc_key(_EDIT_BLOCK_AREA, rel), None)


def _doc_edit_dirty(rel):
    """True when ``rel`` holds an unsaved editor draft that differs from disk.

    ``rel`` is the doc id the draft key is namespaced under (``""`` in
    single-document mode); the file it is compared against is always whatever
    the Reader is displaying (``_READER_TARGET``). Absent draft = clean (the
    draft mirror only appears on a committed edit and is popped again by a
    save or revert). A missing/unreadable file counts as dirty: the draft is
    then the only copy of that text.
    """
    draft = st.session_state.get(docs.doc_key(_EDIT_DRAFT, rel))
    if draft is None:
        return False
    target = _resolve_reader_target(st.session_state.get(_READER_TARGET))
    if not target or not os.path.isfile(target):
        return True
    return _read_text(target) != draft


def _pop_edit_state(rel):
    """Drop every editor key of ``rel`` (draft, widget values, mtime, lock)."""
    for base in (
        _EDIT_DRAFT, _EDIT_AREA, _EDIT_AREA_GEN, _EDIT_BASE_MTIME,
        _EDIT_UNLOCKED, _EDIT_BLOCK_INDEX, _EDIT_BLOCK_AREA,
    ):
        st.session_state.pop(docs.doc_key(base, rel), None)
    _pop_raw_area_generations(rel)


def _pop_raw_area_generations(rel):
    """Drop every generation of the whole-file textarea's widget key.

    The key carries a generation (``_reader_edit_area__g<N>``), so name-based
    cleanup sweeps them by prefix; multi-document closes get this for free
    from the ``__doc__`` suffix sweep.
    """
    suffix = docs.doc_key(_EDIT_AREA, rel)
    for k in [
        k for k in st.session_state
        if isinstance(k, str) and k.startswith(f"{suffix}__g")
    ]:
        st.session_state.pop(k, None)


def _write_doc_edit(rel, target, draft):
    """Commit ``draft`` to ``target``; True (and clean up) on success.

    A successful save makes the file the single source of truth again: the
    draft mirror is dropped (so dirty checks go quiet) and the recorded base
    mtime moves to the just-written file, so the next save doesn't mistake its
    own write for an outside change.
    """
    if not _write_text(target, draft):
        st.error(f"Could not write `{target}` — check the file's permissions.")
        return False
    st.session_state.pop(docs.doc_key(_EDIT_DRAFT, rel), None)
    st.session_state.pop(_EDIT_PENDING, None)
    try:
        st.session_state[docs.doc_key(_EDIT_BASE_MTIME, rel)] = os.path.getmtime(
            target
        )
    except OSError:
        st.session_state.pop(docs.doc_key(_EDIT_BASE_MTIME, rel), None)
    st.toast("Saved to disk", icon="💾")
    return True


def _save_doc_edit():
    """💾 Save handler: write the editor draft to the file on disk.

    The draft is the whole-file textarea's value when the raw editor is
    mounted (the freshest copy — its pending edit lands in session state with
    the click), else the block editor's draft mirror. When the file changed
    on disk since editing started — and the content really differs, since the
    macOS launcher re-copies dropped files (bumping mtime) without changing
    them — a confirmation dialog stands between the draft and the outside
    changes.
    """
    rel = _current_edit_doc()
    # The file to write is whatever the Reader is displaying (_READER_TARGET);
    # ``rel`` (the active-document id) only names the editor's session keys
    # and is empty in single-document mode.
    target = _resolve_reader_target(st.session_state.get(_READER_TARGET))
    if not target or not os.path.isfile(target):
        st.error("The file no longer exists on disk — cannot save.")
        return
    draft = st.session_state.get(_raw_area_key(rel))
    if draft is None:
        draft = st.session_state.get(docs.doc_key(_EDIT_DRAFT, rel))
    if draft is None:
        return
    base_mtime = st.session_state.get(docs.doc_key(_EDIT_BASE_MTIME, rel))
    try:
        cur_mtime = os.path.getmtime(target)
    except OSError:
        cur_mtime = None
    if (
        base_mtime is not None
        and cur_mtime is not None
        and cur_mtime != base_mtime
        and _read_text(target) != draft
    ):
        st.session_state[_EDIT_PENDING] = draft
        _confirm_save_over_changed_file(rel, target)
        return
    _write_doc_edit(rel, target, draft)


@st.dialog("File changed on disk")
def _confirm_save_over_changed_file(rel, target):
    """Modal for saving a draft over a file that changed since it was opened."""
    st.warning(
        f"**{_display_name_for_filepath(target)}** was modified outside the "
        "editor since you started editing. Saving overwrites those changes "
        "with your draft."
    )
    col_over, col_reload, col_cancel = st.columns(3)
    if col_over.button("Overwrite", type="primary"):
        draft = st.session_state.get(_EDIT_PENDING)
        if draft is not None and _write_doc_edit(rel, target, draft):
            st.rerun()
    if col_reload.button("Reload from disk"):
        for base in (_EDIT_DRAFT, _EDIT_BASE_MTIME,
                     _EDIT_BLOCK_INDEX, _EDIT_BLOCK_AREA):
            st.session_state.pop(docs.doc_key(base, rel), None)
        _bump_raw_area(rel)
        st.session_state.pop(_EDIT_PENDING, None)
        st.rerun()
    if col_cancel.button("Cancel"):
        st.session_state.pop(_EDIT_PENDING, None)
        st.rerun()


@st.dialog("Unsaved edits")
def _confirm_close_with_edits(rel):
    """Modal asking to proceed with closing a document holding unsaved edits."""
    shown = st.session_state.get(_READER_TARGET) or rel
    st.warning(
        f"**{_display_name_for_filepath(shown)}** has unsaved editor edits. "
        "Closing the document discards them."
    )
    col_proceed, col_cancel = st.columns(2)
    if col_proceed.button("Discard and close", type="primary"):
        _close_reader()
        st.rerun()
    if col_cancel.button("Cancel"):
        st.rerun()


def _close_reader_clicked():
    """Clear/Close button handler: guard unsaved editor edits first.

    The dirty check runs even when the doc id is empty (single-document
    mode) — the draft then lives under the legacy bare key, which
    ``_doc_edit_dirty`` resolves the same way.
    """
    rel = _current_edit_doc()
    if _doc_edit_dirty(rel):
        _confirm_close_with_edits(rel)
        return
    _close_reader()


def _render_edit_lock(rel):
    """The safety lock toggle: ON (default) = read-only, OFF = editable.

    The toggle's widget value is seeded pre-mount from the non-widget mirror
    so unlocking survives the view switches that prune the widget; the
    on_change callback mirrors every flip back (see _mirror_edit_lock).
    """
    toggle_key = docs.doc_key(_EDIT_LOCK_TOGGLE, rel)
    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = not _edit_unlocked(rel)
    st.toggle(
        "🔒 Edit lock",
        key=toggle_key,
        on_change=_mirror_edit_lock,
        help=(
            "On by default: the document is read-only. Turn off to edit the "
            "file in place — 💾 Save overwrites the file on disk at the path "
            "shown in the caption above."
        ),
    )


# --- block splitting (the in-place editor's source of block boundaries) -----

_MD_PARSER = None


def _md_parser():
    """The block splitter's parser, created once per session.

    CommonMark plus the table/strikethrough rulesets — the constructs
    ``st.markdown`` renders — so block boundaries never cut through a
    construct the renderer draws as one piece.
    """
    global _MD_PARSER
    if _MD_PARSER is None:
        _MD_PARSER = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    return _MD_PARSER


def _md_blocks(text):
    """Split markdown into its top-level blocks for the in-place editor.

    Returns ``(blocks, refs_src)``. Blocks are ``(start, end, source)`` with
    half-open line ranges into ``text``: a whole list/table/quote/fence is one
    block, and the blank lines between blocks belong to no block — so splicing
    a replacement over a block's lines can never eat a separator. ``refs_src``
    is the source of every link-reference definition; the block renderer
    prepends it to each block so per-block rendering still resolves
    ``[text][ref]`` links (definitions themselves render nothing).
    """
    lines = text.splitlines(keepends=True)
    env = {}
    ranges = []
    last = None
    for tok in _md_parser().parse(text, env):
        if tok.level or tok.map is None or tok.type == "inline":
            continue
        if tok.map != last:
            ranges.append((tok.map[0], tok.map[1]))
            last = tok.map
    blocks = [
        (s, e, "".join(lines[s:e])) for s, e in ranges if s < len(lines)
    ]
    refs_src = "".join(
        "".join(lines[ref["map"][0]:ref["map"][1]])
        for ref in env.get("references", {}).values()
        if ref.get("map") and ref["map"][0] < len(lines)
    )
    return blocks, refs_src


def _current_draft(rel, text):
    """The text the unlocked view renders and edits: the unsaved draft if one
    exists (so committed block edits are visible), else ``text`` from disk."""
    draft = st.session_state.get(docs.doc_key(_EDIT_DRAFT, rel))
    return text if draft is None else draft


# --- in-place block editing --------------------------------------------------

def _open_block_edit(index):
    """✎ handle handler: open the in-place editor for block ``index``.

    The block textarea's key is shared by every block of the document (only
    one editor is open at a time) and popped here so the editor re-seeds from
    the freshly selected block instead of a previous block's text.
    """
    rel = _current_edit_doc()
    st.session_state[docs.doc_key(_EDIT_BLOCK_INDEX, rel)] = index
    st.session_state.pop(docs.doc_key(_EDIT_BLOCK_AREA, rel), None)
    st.rerun()


def _commit_block_edit():
    """Commit the open block editor: splice its source back into the draft.

    Runs on Done and on the textarea's own commit (blur / ⌘+Enter), so
    clicking anywhere else lands the edit first. The block's line range is
    looked up in the text the block view currently renders — the draft, else
    the file — the replacement is spliced over exactly those lines, and the
    editor closes (an unchanged block just closes). The whole-file textarea
    is dropped on any real change so it re-seeds from the updated draft
    instead of holding a stale pre-splice copy.
    """
    rel = _current_edit_doc()
    idx_key = docs.doc_key(_EDIT_BLOCK_INDEX, rel)
    area_key = docs.doc_key(_EDIT_BLOCK_AREA, rel)
    index = st.session_state.get(idx_key)
    new_src = st.session_state.get(area_key)
    st.session_state.pop(idx_key, None)
    st.session_state.pop(area_key, None)
    if index is None or new_src is None:
        return
    draft = st.session_state.get(docs.doc_key(_EDIT_DRAFT, rel))
    if draft is None:
        target = _resolve_reader_target(st.session_state.get(_READER_TARGET))
        draft = _read_text(target) if target else ""
    blocks, _refs = _md_blocks(draft)
    if index >= len(blocks):
        return
    start, end, old_src = blocks[index]
    if new_src != old_src:
        lines = draft.splitlines(keepends=True)
        # Keep the block's separators: a final line without a newline would
        # fuse with whatever follows, and blocks that absorb their trailing
        # blank line (lists do) must hand it back if the replacement lacks it.
        missing_tail = len(old_src) - len(old_src.rstrip("\n")) - (
            len(new_src) - len(new_src.rstrip("\n"))
        )
        if missing_tail > 0 and end < len(lines):
            new_src += "\n" * missing_tail
        new_lines = new_src.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n") and end < len(lines):
            new_lines[-1] += "\n"
        draft = "".join(lines[:start] + new_lines + lines[end:])
        st.session_state[docs.doc_key(_EDIT_DRAFT, rel)] = draft
        _bump_raw_area(rel)
    st.rerun()


def _cancel_block_edit():
    """Close the in-place block editor without splicing anything.

    Edits already committed elsewhere stay; only the open editor's
    uncommitted textarea content is discarded.
    """
    rel = _current_edit_doc()
    for base in (_EDIT_BLOCK_INDEX, _EDIT_BLOCK_AREA):
        st.session_state.pop(docs.doc_key(base, rel), None)
    st.rerun()


def _render_block_edit_row(rel, src):
    """The in-place editor that replaces one rendered block: a textarea sized
    to the block plus Done / Cancel. Commits go through :func:`_commit_block_edit`
    (the textarea's on_change), so blur and ⌘+Enter land the edit too."""
    area_key = docs.doc_key(_EDIT_BLOCK_AREA, rel)
    if area_key not in st.session_state:
        st.session_state[area_key] = src
    n_lines = max(1, len(src.rstrip("\n").splitlines()))
    st.text_area(
        "Block source",
        key=area_key,
        height=min(max(76, n_lines * 26 + 26), 620),
        label_visibility="collapsed",
        on_change=_commit_block_edit,
    )
    col_done, col_cancel, _spare = st.columns([1, 1, 4])
    with col_done:
        if st.button("Done", type="primary", key="_reader_blk_done_btn"):
            _commit_block_edit()
    with col_cancel:
        if st.button("Cancel", key="_reader_blk_cancel_btn"):
            _cancel_block_edit()


def _render_blocks(text, rel):
    """The unlocked .md view: the document fully rendered, editable in place.

    Every top-level block is one unit — hovering it reveals a "✎" handle,
    clicking swaps just that block for a small source editor, and committing
    splices the block's source back into the draft (see
    :func:`_commit_block_edit`). Blocks are re-parsed from the current text
    on every run, so committed edits can never desync the indices.
    """
    draft = _current_draft(rel, text)
    blocks, refs_src = _md_blocks(draft)
    idx_key = docs.doc_key(_EDIT_BLOCK_INDEX, rel)
    editing = st.session_state.get(idx_key)
    if editing is not None and not 0 <= editing < len(blocks):
        st.session_state.pop(idx_key, None)
        editing = None
    st.markdown(f"<style>{_BLOCKS_CSS}</style>", unsafe_allow_html=True)
    if not blocks:
        st.caption("_Nothing to edit block-by-block — use Raw source below._")
    with st.container(key="_reader_blocks_area"):
        for i, (_s, _e, src) in enumerate(blocks):
            with st.container(key=f"_reader_blk_{i}"):
                if editing == i:
                    _render_block_edit_row(rel, src)
                    continue
                if st.button("✎", key=f"_reader_blk_btn_{i}"):
                    _open_block_edit(i)
                st.markdown(
                    _escape_currency_dollars(refs_src + src),
                    unsafe_allow_html=True,
                )


def _render_raw_editor(rel, text, in_expander, height=400):
    """The whole-file source textarea.

    For markdown it lives in a collapsed expander (the block editor is the
    primary interface; the raw textarea covers bulk restructuring); text
    files render it plainly as their only editor. Seeded pre-mount from the
    draft mirror (an edit in progress from before a view switch) or from the
    file — never with ``value=``, which would trip Streamlit's
    default-value-vs-session-state policy once the key exists. The key carries
    a generation (:func:`_bump_raw_area`) so block edits can swap the mounted
    textarea over to the fresh draft instead of being reverted by the client.
    """
    area_key = _raw_area_key(rel)
    if area_key not in st.session_state:
        st.session_state[area_key] = st.session_state.get(
            docs.doc_key(_EDIT_DRAFT, rel), text
        )

    def textarea():
        st.text_area(
            "Raw source",
            key=area_key,
            height=height,
            label_visibility="collapsed",
            on_change=_mirror_edit_draft,
        )

    if in_expander:
        with st.expander("Raw source", expanded=False):
            st.caption(
                "_Whole-file source — commits on click-outside / ⌘+Enter; "
                "committing closes any open block editor._"
            )
            textarea()
    else:
        textarea()


def _revert_doc_edit(rel):
    """Revert handler: drop the draft and every editor, re-seed from disk."""
    for base in (_EDIT_DRAFT, _EDIT_BASE_MTIME,
                 _EDIT_BLOCK_INDEX, _EDIT_BLOCK_AREA):
        st.session_state.pop(docs.doc_key(base, rel), None)
    _bump_raw_area(rel)
    st.rerun()


def _render_editor(target, text):
    """The unlocked editor: the document stays rendered, edits happen in place.

    Markdown files get the block editor (:func:`_render_blocks`) plus the
    collapsed raw-source expander; text files keep a split editor (the raw
    textarea beside a code-block preview). Both feed the same draft mirror,
    so the dirty checks and 💾 Save work identically. The base mtime is
    recorded when editing starts so the save can detect outside changes.
    """
    rel = _current_edit_doc()
    if target.endswith(".md"):
        st.caption(
            "**Editing unlocked** — hover a block and click its ✎ handle to "
            "edit it right in the document. Changes stay in this browser "
            "until you 💾 Save (which overwrites the file at the path shown "
            "above); ⚡ Summarize always reads the saved file."
        )
    else:
        st.caption(
            "**Editing unlocked** — changes stay in this browser until you "
            "save. 💾 Save overwrites the file on disk at the path shown "
            "above; ⚡ Summarize always reads the saved file."
        )
    if len(text) > _EDIT_SIZE_WARN:
        st.warning(
            "This document is very large — the editor may feel slow."
        )
    mtime_key = docs.doc_key(_EDIT_BASE_MTIME, rel)
    if mtime_key not in st.session_state:
        try:
            st.session_state[mtime_key] = os.path.getmtime(target)
        except OSError:
            pass
    col_save, col_revert, _spare = st.columns([1, 1, 4])
    with col_save:
        if st.button("💾 Save", type="primary", key="_reader_edit_save_btn"):
            _save_doc_edit()
    with col_revert:
        if st.button("Revert", key="_reader_edit_revert_btn"):
            _revert_doc_edit(rel)
    if target.endswith(".md"):
        _render_blocks(text, rel)
        _render_raw_editor(rel, text, in_expander=True)
    else:
        col_src, col_view = st.columns([1, 1], gap="medium")
        with col_src:
            st.caption("_Source — commits on click-outside / ⌘+Enter._")
            _render_raw_editor(rel, text, in_expander=False, height=560)
        with col_view:
            st.caption("_Preview — the file as the Reader renders it._")
            st.code(_current_draft(rel, text), language="text")


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
    same escape hatch ``app._preserve_reader_scroll`` uses to remember the
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
            // app's scroll-restore script (which re-establishes the last
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
    # --- Quick action + copy + clear ------------------------------------
    # ⚡ Summarize (the quick action, left of Copy) opens a NEW "Summary" chat
    # session tab for the active document (docs.add_chat activates it), stages
    # the editable summary prompt into that session, and jumps to the LLM chat,
    # which sends it through the normal chat pipeline with the panel's existing
    # provider/model settings — the full document always goes along as context.
    col_quick, col_copy, col_clear, col_lock = st.columns([1, 1, 1, 1])
    with col_quick:
        if st.button(
            "⚡ Summarize",
            type="primary",
            key="_reader_quick_summarize_btn",
            help=(
                "Quick action: open a new 'Summary' tab in the LLM chat and "
                "send this document there with the summarize prompt (the "
                "chat panel's existing provider/model settings are used "
                "as-is). Edit the prompt in the 'Quick summarize prompt' "
                "expander below."
            ),
        ):
            prompt = _current_quick_prompt()
            if not prompt:
                st.warning(
                    "The quick summarize prompt is empty — edit it in the "
                    "'Quick summarize prompt' expander below (or reset it) "
                    "and try again."
                )
            else:
                # A dedicated session per request: summaries never mix with an
                # ongoing conversation. add_chat activates the new session, so
                # the staged key below — and the chat panel's own key
                # resolution — both point at it.
                docs.add_chat(docs.active_document(), label="Summary")
                st.session_state[_reader_quick_prompt_key()] = prompt
                st.session_state[TABS_KEY] = CHAT_TAB_LABEL
                st.rerun()
    with col_copy:
        _copy_text_button(text)
    with col_clear:
        if st.button("Clear", key="_reader_close_doc_btn"):
            _close_reader_clicked()
    with col_lock:
        _render_edit_lock(_current_edit_doc())

    # --- Quick summarize prompt (the ⚡ button's payload) ----------------
    # Sits at the top of the panel, next to the button that uses it. The
    # textarea is the modifiable copy of the factory default: it is seeded
    # pre-mount from the non-widget mirror (never with value=, which would
    # trip Streamlit's default-value-vs-session-state policy once the key
    # exists) and every edit is mirrored back via on_change, so the prompt
    # survives Reader↔chat view switches.
    with st.expander("Quick summarize prompt", expanded=False):
        st.caption(
            "_**⚡ Summarize** opens a new **Summary** tab in the LLM chat and "
            "sends this document there using this prompt — the full document "
            "is attached as context, and the chat panel's provider/model "
            "settings are used as-is. Edits stick for this browser session._"
        )
        if _QUICK_PROMPT_EDIT not in st.session_state:
            st.session_state[_QUICK_PROMPT_EDIT] = st.session_state.get(
                _QUICK_PROMPT_SAVED, QUICK_SUMMARY_PROMPT
            )
        st.text_area(
            "Prompt",
            key=_QUICK_PROMPT_EDIT,
            height=150,
            on_change=_save_quick_prompt_edit,
        )
        if st.button("Reset to default", key="_reader_quick_prompt_reset"):
            st.session_state.pop(_QUICK_PROMPT_SAVED, None)
            st.session_state.pop(_QUICK_PROMPT_EDIT, None)
            st.rerun()

    # Permanently bump body-text + code font size. Scoped to Streamlit's
    # markdown/code containers so widget labels, buttons, and headers keep their
    # default sizes; chat-message bodies use the same containers, so the chat
    # panel is covered too.
    st.markdown(_BODY_FONT_SIZE_CSS, unsafe_allow_html=True)

    # .md is authored content → render as markdown; anything else → code block.
    # unsafe_allow_html lets HTML inline markup like <b>bold</b> / <strong>…
    # render alongside standard **bold** markdown — without it Streamlit strips
    # the tags and shows their inner text unstyled.
    #
    # With the safety lock off, the document stays rendered and becomes
    # editable in place: block-level ✎ handles plus a raw-source expander for
    # .md, a split editor for text files; locked, the file is read-only and
    # rendered exactly as before. Lock state is keyed by the active document
    # (see _current_edit_doc), never by the raw target.
    if _edit_unlocked(_current_edit_doc()):
        _render_editor(target, text)
    elif target.endswith(".md"):
        st.markdown(_escape_currency_dollars(text), unsafe_allow_html=True)
    else:
        st.code(text, language="text")

    # A sidebar-table-of-contents jump, if one was staged: scroll to the
    # heading, then drop the staged target so the next rerun doesn't re-jump.
    jump = st.session_state.get(_TOC_JUMP)
    if jump:
        _inject_toc_jump(jump)
        st.session_state.pop(_TOC_JUMP, None)


def _close_reader():
    """Drop the active document (via _close_reader_clicked / the dirty dialog).

    In multi-document mode the document is removed from the registry — its
    chat and editor state (``__doc__``-suffixed keys) is dropped and the next
    open document becomes active, so a staged quick prompt or unsaved draft
    from one document can't leak into another. In single-document mode the
    reader target and the legacy bare quick-prompt / editor keys are cleared.
    """
    doc = docs.active_document()
    if doc:
        docs.remove_document(doc)
    else:
        st.session_state.pop(_READER_TARGET, None)
        # In single-document mode the editor keys are the legacy bare keys —
        # dropped by name (multi-doc copies carry the __doc__ suffix and are
        # swept by remove_document above).
        for base in (
            _EDIT_DRAFT, _EDIT_AREA, _EDIT_AREA_GEN, _EDIT_BASE_MTIME,
            _EDIT_UNLOCKED, _EDIT_BLOCK_INDEX, _EDIT_BLOCK_AREA,
        ):
            st.session_state.pop(base, None)
        _pop_raw_area_generations("")
        # In single-document mode, clear the staged quick prompt (legacy bare
        # key shared by every chat session of this now-closed document).
        for sid in docs.chat_sessions(""):
            st.session_state.pop(
                docs.chat_key(_READER_QUICK_PROMPT, sid, ""), None
            )
