"""LLM chat panel: converse with an LLM about the document open in the Reader,
using Streamlit's native chat UI (``st.chat_message`` / ``st.chat_input``).

There is no separate document picker — the chat always follows whatever the
Reader currently has open (``_reader_target``). Open a document in the Reader,
then ask about it here. The document's full text is sent once as the leading
context turn, so the model has it in mind for the whole conversation.

With several documents open (multi-document mode, see :mod:`md_llm.docs`) the
chat still follows the ACTIVE document, and each document's conversation is
stored under its own namespaced session keys
(``_chat_messages__doc__<relpath>``) — so each open document has a fully
independent chat that keeps streaming in the background while you work on
another.

On top of that, any document can hold **several chat sessions** ("tabs"):
the session buttons at the top of this panel pick the active one, ``+ New``
opens another independent conversation about the same document, and ``✕``
closes the current one. Each session keeps its own conversation, background
stream task, last error, and staged Reader ⚡ Summarize prompt, keyed via
``docs.chat_key`` as ``<base>__chat__<id>__doc__<relpath>`` (session 1 keeps
the document's legacy keys). Provider/model/key controls are shared panel-wide
— sessions differ only in their conversations and streams.

The assistant reply is streamed token-by-token via ``st.write_stream`` (both
OpenRouter's SSE deltas and Ollama's newline-delimited chunks are supported).

The conversation lives in session memory (``_chat_messages``); a **Save
conversation** button writes it as a plain ``<docstem>__chat_<UTC>.md`` file
into the **Save location** directory — a memorized choice (settings key
``llm.chat_save_dir``, editable in the expander under the save buttons) that
falls back to the host's ``core.chat_save_dir``. No sidecar metadata, no
transcript linkage — md_llm has no notion of "transcripts". The
provider/model/key controls live in this panel under the ``chat_`` key
namespace.
"""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components

from . import llm
from . import docs
from . import sandbox
from .autossh import _render_autossh_panel
from .console import log_event
from .controls import (
    _current_llm_model,
    _current_oai_endpoint,
    _current_opencode_variant,
    _oai_registry_entry,
    _remember_oai_endpoint,
    _remember_opencode_model,
    _remember_openrouter_endpoint,
    _remember_openrouter_model,
    _render_llm_controls,
    _save_oai_registry_entry,
)
from .core import get_core
from .state import (
    DEFAULT_LLM_AUTOSSH,
    _BODY_FONT_SIZE_CSS,
    _display_name_for_filepath,
    _escape_currency_dollars,
    _read_text,
)

# Session-state keys (session-memory only — nothing persisted except via Save).
_CHAT_MESSAGES = "_chat_messages"  # list[{"role","content"}]
# Dict describing the in-flight background stream (see _stream_worker): the LLM
# call runs in a daemon thread so it survives tab switches — esp. OpenCode's
# subprocess, whose generator kills the process on close. Keys: text/done/error/source.
_CHAT_BG_TASK = "_chat_bg_task"

# A prompt staged by the Reader's ⚡ Summarize quick action for the NEXT chat
# turn of the ACTIVE document + chat session (the button also opens a new
# "Summary" session tab and switches the view here). Read here by literal
# string (matches reader.py) rather than imported from .reader, to keep
# chat↔reader decoupled. Popped and sent by _send_staged_quick_prompt.
_READER_QUICK_PROMPT = "_reader_quick_prompt"


def _chat_state_key(base):
    """Session-state key for one chat field, scoped to the active session.

    Each open document's ACTIVE chat session owns its own conversation /
    stream task / last error (see :mod:`md_llm.docs`), so chats run
    independently no matter how many documents or sessions exist. Session 1 of
    a document uses its legacy keys, so existing sessions are untouched.
    """
    doc = docs.active_document()
    return docs.chat_key(base, docs.active_chat(doc), doc)


def _staged_quick_prompt_key():
    """Session key of the Reader ⚡ Summarize prompt staged for the ACTIVE session."""
    doc = docs.active_document()
    return docs.chat_key(_READER_QUICK_PROMPT, docs.active_chat(doc), doc)


def _session_sandbox_dir():
    """This chat session's OpenCode sandbox, created empty on first use.

    Memoized per chat session via ``docs.chat_key``, so every turn of one
    session reuses the same directory (files persist across its turns) while
    every other session — even on the same document, running in parallel —
    gets a distinct one. Directories abandoned by closed sessions/app restarts
    are garbage-collected by age when a new sandbox is created; see
    :mod:`md_llm.sandbox` for the isolation and clearing guarantees.
    """
    doc = docs.active_document()
    sid = docs.active_chat(doc)
    key = docs.chat_key("_opencode_sandbox", sid, doc)
    path = st.session_state.get(key)
    if not path:
        stem = os.path.splitext(os.path.basename(doc))[0] if doc else "chat"
        path = sandbox.new_session_sandbox(f"{stem}-s{sid}")
        st.session_state[key] = path
    return path


def _resolve(path):
    """Resolve a relpath (against core.base_dir) or absolute path to a real file.

    Returns the absolute path if it is an existing file, else None.
    """
    if not path:
        return None
    full = path
    if not os.path.isabs(full):
        full = os.path.join(get_core().base_dir, full)
    full = os.path.abspath(full)
    if os.path.isfile(full):
        return full
    return None


def _current_context_path():
    """The document the chat is about: whatever the Reader has open.

    Falls back to a host-staged ``_viewing_transcript`` (kept for host
    compatibility). Returns an absolute path or None.
    """
    for candidate in (
        st.session_state.get("_reader_target"),
        (st.session_state.get("_viewing_transcript") or {}).get("path"),
    ):
        resolved = _resolve(candidate)
        if resolved:
            return resolved
    return None


# ---------------------------------------------------------------------------
# Saving the conversation
# ---------------------------------------------------------------------------

# Settings key (inside the ``llm`` subkey md_llm owns) holding the memorized
# save directory. Absent/empty means "use the host's core.chat_save_dir".
_SAVE_DIR_SETTING_KEY = "chat_save_dir"
# Widget key of the Save-location text input. Starts with neither ``chat_`` nor
# ``_chat_ssh_`` so the control snapshot ignores it: the value is persisted in
# settings (not session memory) and the input re-mounts from there.
_SAVE_DIR_INPUT_KEY = "_chat_save_dir_input"


def _settings_chat_save_dir():
    """The memorized save directory from settings ("" when none is stored)."""
    llm_s = get_core().load_settings().get("llm")
    if not isinstance(llm_s, dict):
        return ""
    val = llm_s.get(_SAVE_DIR_SETTING_KEY)
    return val.strip() if isinstance(val, str) else ""


def _remember_chat_save_dir(path):
    """Persist `path` as the save directory under ``llm.chat_save_dir``.

    An empty `path` clears the memorized choice, falling back to the host's
    ``core.chat_save_dir``.
    """
    settings = get_core().load_settings()
    llm_s = dict(settings.get("llm") or {})
    path = (path or "").strip()
    if path:
        llm_s[_SAVE_DIR_SETTING_KEY] = path
    else:
        llm_s.pop(_SAVE_DIR_SETTING_KEY, None)
    settings["llm"] = llm_s
    get_core().save_settings(settings)


def _chat_save_dir():
    """Where the next save writes: the memorized directory or the host default."""
    return _settings_chat_save_dir() or get_core().chat_save_dir


def _save_dir_problem(path):
    """Why `path` can't serve as the save directory, or None when it can.

    Read-only checks — nothing is created here (the save itself makedirs). A
    missing directory is fine as long as its nearest existing ancestor is a
    writable directory; anything else (empty path, a file in the way, no write
    permission) is reported so it can be fixed before saving.
    """
    raw = (path or "").strip()
    if not raw:
        return "Enter a directory path."
    p = os.path.abspath(os.path.expanduser(raw))
    if os.path.exists(p):
        if not os.path.isdir(p):
            return f"`{p}` is a file, not a directory."
        if not os.access(p, os.W_OK | os.X_OK):
            return f"No write permission in `{p}`."
        return None
    probe = os.path.dirname(p)
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    if not probe:
        return f"`{p}` has no existing parent directory to create it under."
    if not os.path.isdir(probe):
        return f"`{probe}` is a file, so `{p}` cannot be created under it."
    if not os.access(probe, os.W_OK | os.X_OK):
        return (
            f"No write permission in `{probe}` — needed to create `{p}`."
        )
    return None


def _on_save_dir_input_change():
    """Memorize the Save-location input's committed value (widget callback)."""
    _remember_chat_save_dir(st.session_state.get(_SAVE_DIR_INPUT_KEY))


def _slugify_stem(path):
    """Filesystem-safe stem for a saved-chat filename, from a document path."""
    if not path:
        return "chat"
    stem = os.path.splitext(os.path.basename(path))[0]
    # Collapse non alnum/CJK to underscores; a plain doc stem like "my-notes"
    # survives untouched, an opaque uuid stays as-is.
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", stem).strip("_")
    return slug or "chat"


def _render_chat_as_markdown(context_path, provider, model):
    """Render the saved conversation as Markdown.

    Layout: provenance header, then the source document's full text (so the
    saved chat is self-contained), then the Q&A turns verbatim from
    ``_chat_messages``.
    """
    name = _display_name_for_filepath(context_path) if context_path else "(none)"
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Chat — {name}",
        "",
        f"_Saved {when} · Provider: **{provider}** · Model: `{model}`_",
        "",
        "---",
        "",
    ]
    # Embed the source text before the turns so the saved chat stands alone.
    if context_path:
        source_text = _read_text(context_path)
        if source_text.strip():
            lines.append("## Source document")
            lines.append("")
            lines.append(source_text.rstrip())
            lines.append("")
            lines.append("---")
            lines.append("")
    for m in (st.session_state.get(_chat_state_key(_CHAT_MESSAGES)) or []):
        role = m.get("role", "")
        content = (m.get("content") or "").rstrip()
        if role == "user":
            lines.append("**You:**")
            lines.append("")
            lines.append(content)
        elif role == "assistant":
            lines.append("**Assistant:**")
            lines.append("")
            lines.append(content)
        else:
            label = role.capitalize() or "Message"
            lines.append(f"**{label}:**")
            lines.append("")
            lines.append(content)
        lines.append("")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _chat_default_title(source_title, messages):
    """Default display title for a saved chat.

    ``"<source title> — <first user message>"`` — the source document's own
    title followed by the opening question. With no source title, just the
    opening question. Falls back to the source title (or a placeholder) when
    there is no user message.
    """
    first = next(
        ((m.get("content") or "").strip() for m in messages
         if m.get("role") == "user" and (m.get("content") or "").strip()),
        None,
    )
    if first:
        if source_title:
            return f"{source_title} — {first[:80]}"
        return first[:80]
    return source_title or "Chat"


def _write_chat_md(context_path, text, save_dir=None):
    """Write `text` as a ``<docstem>__chat_<UTC>.md`` in `save_dir`.

    Pure I/O (no Streamlit calls) so it is unit-testable. Each saved chat is
    a plain markdown file named after the source document stem plus a UTC
    timestamp (so multiple chats about the same doc don't collide). No sidecar
    metadata is written — md_llm has no transcript-linkage concept.
    `save_dir` defaults to the memorized chat save directory (falling back to
    the host's ``core.chat_save_dir``).
    Returns the absolute path, or None on write failure.
    """
    save_dir = save_dir or _chat_save_dir()
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError:
        return None
    stem = _slugify_stem(context_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # The timestamp is second-granular, so two rapid saves of the same document
    # could otherwise collide. Append -2, -3, … until the name is free (matches
    # how a filesystem "Keep both" copy disambiguates).
    out_path = os.path.join(save_dir, f"{stem}__chat_{ts}.md")
    if os.path.exists(out_path):
        n = 2
        while os.path.exists(
            os.path.join(save_dir, f"{stem}__chat_{ts}-{n}.md")
        ):
            n += 1
        out_path = os.path.join(save_dir, f"{stem}__chat_{ts}-{n}.md")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        return None
    return out_path


def _save_conversation(context_path, provider, model, save_dir=None):
    """Write the conversation to the chat save directory.

    `save_dir` overrides the memorized directory (kept for explicit/test
    callers; the panel saves to :func:`_chat_save_dir`). Every outcome —
    empty conversation, invalid directory, write failure, success — surfaces
    a clear Streamlit message.

    Returns the saved absolute path, or None on failure / when the conversation
    is empty.
    """
    messages = st.session_state.get(_chat_state_key(_CHAT_MESSAGES)) or []
    if not messages:
        st.warning("Nothing to save — the conversation is empty.")
        return None

    target = save_dir or _chat_save_dir()
    problem = _save_dir_problem(target)
    if problem:
        st.error(f"**Save failed** — {problem}")
        return None

    text = _render_chat_as_markdown(context_path, provider, model)
    source_title = (
        _display_name_for_filepath(context_path) if context_path else ""
    )
    _chat_default_title(source_title, messages)  # computed for parity/title hooks
    out_path = _write_chat_md(context_path, text, save_dir=save_dir)
    if out_path is None:
        st.error(
            "**Save failed** — could not write to "
            f"`{os.path.abspath(os.path.expanduser(target))}`. Check that the "
            "directory exists and is writable (see Save location above)."
        )
        return None
    return out_path


# ---------------------------------------------------------------------------
# Building the outgoing message list
# ---------------------------------------------------------------------------

def _send_context_and_turns(context_path):
    """Build the message list to send to the LLM for the current chat.

    The document text becomes a single leading user message (so the model has it
    in context for the whole conversation), followed by an assistant ack, then
    the actual Q&A turns. The leading turn is rebuilt from disk each send, so
    opening a different document in the Reader takes effect on the next send.
    """
    messages = []
    if context_path:
        doc = _read_text(context_path)
        if doc.strip():
            messages.append({
                "role": "user",
                "content": f"Here is the document I want to discuss:\n\n{doc}",
            })
            messages.append({
                "role": "assistant",
                "content": "Got it — I've read the document. "
                           "What would you like to know?",
            })
    turns = list(st.session_state.get(_chat_state_key(_CHAT_MESSAGES)) or [])
    messages.extend(turns)
    return messages


def _turns_to_opencode_prompt(turns):
    """Flatten the chat message list into a single prompt for ``opencode run``.

    ``opencode run`` takes one positional prompt (not a message array), so the
    document-context turn + the Q&A history are rendered as labelled
    ``User:`` / ``Assistant:`` blocks. The system instruction is passed
    separately to :func:`md_llm.llm.opencode_chat_stream`, which prepends it.
    """
    labels = {"user": "User", "assistant": "Assistant", "system": "System"}
    parts = []
    for m in turns:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{labels.get(role, role.capitalize())}:\n{content}")
    return "\n\n".join(parts).strip()


def _safe_stream(gen, holder):
    """Yield from ``gen``, capturing its first exception into ``holder``.

    The streaming generators in ``llm`` raise lazily (from inside the iteration
    that ``st.write_stream`` drives), so a raised error would otherwise crash
    the write_stream call. This wrapper swallows the first exception, records it
    in ``holder["error"]``, and ends the stream cleanly.
    """
    try:
        yield from gen
    except Exception as e:  # noqa: BLE001 — surface any provider error inline
        holder["error"] = str(e)


def _build_stream(context_path, holder):
    """Return (stream_generator, error) for the current chat_* provider/model.

    On a validation failure returns (None, error_message); otherwise returns the
    streaming generator (a ``_safe_stream`` wrapper bound to ``holder``) and None.
    """
    p = "chat_"
    provider = st.session_state.get(f"{p}llm_provider", "OpenRouter")
    model = _current_llm_model(prefix=p)
    instruction = st.session_state.get(f"{p}llm_instruction") or None

    if not model:
        return None, "Pick or type an LLM model first (in the LLM controls)."

    turns = _send_context_and_turns(context_path)
    if provider == "OpenRouter":
        api_key = st.session_state.get(f"{p}llm_or_api_key") or os.environ.get(
            "OPENROUTER_API_KEY", ""
        )
        if not api_key:
            return None, (
                "No OpenRouter API key. Paste one in the LLM controls or set "
                "the OPENROUTER_API_KEY env var."
            )
        endpoint = st.session_state.get(
            f"{p}llm_or_endpoint", llm.OPENROUTER_DEFAULT_ENDPOINT
        )
        # Persist the chosen model + endpoint so they reappear next session,
        # mirroring the OpenCode branch's model memory. The API key stays
        # write-only by design — never persisted to disk.
        _remember_openrouter_model(model)
        _remember_openrouter_endpoint(endpoint)
        gen = llm.openrouter_chat_stream(
            turns, api_key=api_key, model=model, endpoint=endpoint,
            instruction=instruction,
        )
    elif provider == "OpenAI-compatible":
        endpoint = _current_oai_endpoint("chat_")
        if not endpoint:
            return None, (
                "No OpenAI-compatible endpoint selected. Please select an "
                "endpoint from the dropdown in the LLM controls."
            )
        entry = _oai_registry_entry(
            get_core().load_settings().get("llm") or {}, endpoint
        )
        api_key = (
            st.session_state.get(f"{p}llm_oai_api_key")
            or entry["api_key"]
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not api_key:
            return None, (
                "No OpenAI-compatible API key. Paste one in the LLM controls "
                "or set the OPENAI_API_KEY env var."
            )
        # Persist this key + model paired with the endpoint in the shared
        # registry, mirroring what a host's manual / autopilot panels do on run.
        _remember_oai_endpoint(endpoint)
        _save_oai_registry_entry(
            endpoint, last_model=model, api_key=api_key,
            pending_model_key="_pending_chat_oai_model_sel",
            pending_api_key_key="_pending_chat_oai_api_key",
        )
        gen = llm.openai_chat_stream(
            turns, api_key=api_key, model=model, endpoint=endpoint,
            instruction=instruction,
        )
    elif provider == "OpenCode":
        workdir = sandbox.normalize_workdir(
            st.session_state.get(f"{p}llm_opencode_workdir")
        )
        hardened = bool(st.session_state.get(f"{p}llm_opencode_hardened", True))
        if workdir is None:
            # Managed mode: this session's own fresh sandbox (Seatbelt-confined
            # when hardened), never the shared uploads folder.
            workdir = _session_sandbox_dir()
        attach = (st.session_state.get(f"{p}llm_opencode_attach") or "").strip() or None
        agent = (st.session_state.get(f"{p}llm_opencode_agent") or "").strip() or None
        variant = _current_opencode_variant(p)
        # Persist the chosen model so it reappears next session.
        _remember_opencode_model(model)
        prompt = _turns_to_opencode_prompt(turns)
        gen = llm.opencode_chat_stream(
            prompt, model=model, workdir=workdir, attach=attach,
            agent=agent, variant=variant, hardened=hardened,
            instruction=instruction,
        )
    else:
        endpoint = st.session_state.get(
            f"{p}llm_endpoint", llm.DEFAULT_ENDPOINT
        )
        gen = llm.ollama_chat_stream(
            turns, endpoint=endpoint, model=model, instruction=instruction,
        )
    return _safe_stream(gen, holder), None


def _send_staged_quick_prompt(context_path):
    """Send the Reader-staged ⚡ Summarize prompt as the next chat turn.

    The Reader's quick-action button opens a new "Summary" session tab for the
    active document, stages its prompt into that session (scoped per document +
    chat session) and switches the view here; this pops it and runs the exact
    ``st.chat_input`` pipeline — user turn appended to the conversation,
    ``_build_stream`` snapshotting it, the stream handed to a background worker
    thread. Returns the started task dict, or None when nothing was staged or
    validation failed (the error lands in ``_chat_last_error`` and surfaces as
    the transient error bubble, exactly like a failed typed send).

    Must be called AFTER the chat controls have mounted this run: like a typed
    send, it reads the ``chat_*`` provider/model/key values, and the OpenCode
    variant default is only seeded when those controls render.
    """
    prompt = st.session_state.get(_staged_quick_prompt_key())
    if prompt is None:
        return None
    st.session_state.pop(_staged_quick_prompt_key(), None)
    prompt = (prompt or "").strip()
    if not prompt:
        return None

    provider = st.session_state.get("chat_llm_provider", "OpenRouter")
    model = _current_llm_model(prefix="chat_") or "(unknown)"
    chat_src = f"Quick summarize ({provider} · {model})"
    preview = prompt.replace("\n", " ")[:80]
    log_event(f"Chat send → {preview}", level="info", source=chat_src)

    msgs = st.session_state.setdefault(_chat_state_key(_CHAT_MESSAGES), [])
    msgs.append({"role": "user", "content": prompt})
    holder = {}
    stream, verr = _build_stream(context_path, holder)
    if stream is None:
        msgs.pop()  # validation failed: roll back the dangling question
        st.session_state[_chat_state_key("_chat_last_error")] = verr
        log_event(f"Chat failed: {verr}", level="error", source=chat_src)
        return None
    task = {"text": "", "done": False, "error": None, "source": chat_src}
    worker = threading.Thread(
        target=_stream_worker, args=(task, stream, holder), daemon=True,
    )
    st.session_state[_chat_state_key(_CHAT_BG_TASK)] = task
    worker.start()
    return task


def _stream_worker(task, stream, holder):
    """Consume ``stream`` in a background thread, accumulating text into ``task``.

    Detached from Streamlit's render loop so the LLM call keeps running when the
    user switches tabs. This matters most for the OpenCode provider:
    ``opencode_chat_stream`` drives a subprocess and kills it in its generator's
    ``finally``; and ``st.write_stream`` is cancelled by the rerun a tab switch
    triggers, which would close the generator and terminate the agent. Iterating
    here keeps the subprocess alive to completion.

    The chat panel reads ``task`` (``text``/``done``/``error``) on each render
    for live display and finalizes once when ``done`` is set. Must not call any
    ``st.*`` API (not thread-safe).
    """
    buf: list[str] = []
    try:
        for chunk in stream:
            if chunk:
                buf.append(chunk)
                task["text"] = "".join(buf)
    except Exception as e:  # noqa: BLE001 — surface any provider error
        task["error"] = str(e)
    if not task.get("error") and holder.get("error"):
        task["error"] = holder["error"]
    task["text"] = (task.get("text") or "").strip()
    task["done"] = True


@st.fragment(run_every=0.4)
def _stream_partial_reply(task):
    """Live-view an in-flight background stream, re-rendering only this bubble.

    ``run_every`` polls the worker's shared ``task`` dict and re-runs just this
    fragment — never the whole page. (The old full-page ``st.rerun`` poll
    re-rendered the whole history every 0.4 s, and each rerun yanked the
    viewport back to the bottom, making earlier turns unreadable mid-stream;
    a fragment rerun leaves the rest of the page and the user's scroll
    position untouched.)

    Must NOT touch ``st.session_state`` or finalize the task: when the worker
    sets ``done``, trigger a full app rerun and let ``render_chat`` fold the
    finished reply in from the main script — which does so BEFORE rendering
    any element, so the finalized conversation lands in a single frontend
    commit (see ``_finalize_chat_task``).
    """
    if task.get("done"):
        st.rerun(scope="app")
    with st.chat_message("assistant"):
        body = task.get("text") or ""
        if body:
            st.markdown(_escape_currency_dollars(body) + " ▌")
        else:
            st.caption(
                "_working… (running in the background — switching tabs is "
                "safe; the reply will appear here)_"
            )


def _finalize_chat_task(task, doc=None, chat_id=None):
    """Fold a finished background stream's outcome into session state.

    Called at the very TOP of ``render_chat`` — before any element renders —
    so the finished reply joins the history that this same run then draws,
    and the streaming bubble is swapped for the final message in ONE
    frontend commit.

    Why that ordering matters: Streamlit wraps the main view in a sticky
    scroll-to-bottom container whenever an ``st.chat_input`` is mounted, and
    that container re-asserts "scroll to the bottom" on every content-height
    change. The old flow finalized MID-run (after the history loop had
    already drawn) and then called ``st.rerun()``: commit 1 removed the
    streaming bubble (the page shrank by the whole reply), commit 2 re-added
    it as a history message (the page regrew). That shrink→regrow churn was
    exactly what the sticky container rode to yank the viewport back down
    the moment the LLM finished — even though the user had scrolled up to
    read. Finalizing before the first element renders makes completion a
    single, near-height-neutral commit, and dropping the extra ``st.rerun``
    means the page (controls, sidebar, history) is not re-rendered twice.

    ``task`` is the worker dict (``text``/``done``/``error``/``source``);
    the caller pops the ``_CHAT_BG_TASK`` session key afterwards.
    When ``doc``/``chat_id`` are given the task is finalized into that
    document/session's conversation (used by the multi-doc finalizer that
    scans all open documents). When omitted the active document/session is
    used (backwards compatible for single-call sites).
    """
    # Resolve the target keys for this task's document/session.  When the
    # caller supplies doc/chat_id (the multi-doc scan) use those; otherwise
    # fall back to the active document/session (the original single-task
    # path).
    if doc is None and chat_id is None:
        err_key = _chat_state_key("_chat_last_error")
        msg_key = _chat_state_key(_CHAT_MESSAGES)
    else:
        # Normalise: active_document() uses None for bare mode, docs.chat_key
        # treats falsy as bare.
        err_key = docs.chat_key("_chat_last_error", chat_id, doc)
        msg_key = docs.chat_key(_CHAT_MESSAGES, chat_id, doc)
    if task.get("error"):
        st.session_state[err_key] = task["error"]
        log_event(f"Chat failed: {task['error']}", level="error",
                  source=task.get("source", ""))
        return
    reply_text = (task.get("text") or "").strip()
    st.session_state.setdefault(msg_key, []).append({
        "role": "assistant",
        "content": reply_text
        or "_(empty response — nothing came back.)_",
    })
    if reply_text:
        log_event(f"Chat reply ({len(reply_text)} chars)",
                  level="info", source=task.get("source", ""))
    else:
        log_event("Chat reply empty — nothing came back.",
                  level="warn", source=task.get("source", ""))


def _finalize_all_done_tasks():
    """Finalize every completed background stream across all open documents.

    The original single-task finalizer only checked the *active* document's
    ``_CHAT_BG_TASK`` key.  If the user switched documents while a stream was
    running, the completed reply stayed under the original document's key and
    was only finalized when the user switched back — until then ``Save
    conversation`` for that document saw only the user turn (or appeared
    empty) and the assistant reply seemed lost.  Scanning all open documents
    and their chat sessions ensures every finished reply is folded into its
    correct conversation on the very next render, no matter which document is
    currently active.
    """
    # Collect all doc ids that might hold a task: every open document plus
    # the bare (no-document) context.  ``open_documents()`` is [] in single-
    # doc mode, so the bare entry is still checked.
    seen_docs = set(docs.open_documents())
    seen_docs.add(docs.active_document())
    seen_docs.add("")  # bare / single-doc fallback
    seen_docs.add(None)
    for doc in list(seen_docs):
        # Normalize None -> "" for chat_sessions lookup; docs.chat_sessions
        # handles None/"" as bare.
        doc_id = doc or ""
        # In single-doc mode with no registry, chat_sessions("") returns [1];
        # for an open doc it returns its actual sessions.
        try:
            sids = docs.chat_sessions(doc_id)
        except Exception:
            sids = [1]
        for sid in sids:
            key = docs.chat_key(_CHAT_BG_TASK, sid, doc_id)
            task = st.session_state.get(key)
            if task and task.get("done"):
                _finalize_chat_task(task, doc_id, sid)
                st.session_state.pop(key, None)


def _tame_chat_autoscroll():
    """Stop Streamlit's sticky chat scroller from yanking the viewport down.

    While any ``st.chat_input`` is mounted, Streamlit wraps the whole main
    view in a "scroll-to-bottom" container (``useScrollToBottom`` /
    ``useScrollAnimation`` in its frontend): whenever that container's
    content height changes, the container re-asserts "scroll to the
    bottom" — INCLUDING when the user has already scrolled up to read.
    Unsticking is supposed to happen on a *user* scroll event, but the
    hook classifies a scroll event as user-initiated only when the
    container's ``scrollHeight``/``offsetHeight`` are unchanged since the
    last handled event. Every content-height change around the user's
    scroll (the finalize commit swapping the streaming bubble for the
    final message, late layout shifts, the disabled→enabled input
    transition) therefore misclassifies the user's wheel-up as a
    "synthetic" Chrome resize-compensation event and re-arms the
    auto-scroll — the viewport snaps back down the moment the LLM
    finishes, exactly what the user reports. A 1px at-bottom threshold,
    a 100ms scroll-event debouncer and an ignore-window after each
    auto-scroll animation all widen that race window (verified against
    Streamlit 1.58's frontend source and in a live browser).

    The fix is deliberately surgical: the hook scrolls by *assigning*
    ``container.scrollTop = value`` from a rAF loop; the browser's own
    user-driven scrolling never goes through that property setter. So a
    same-origin script (``components.html``, the same escape hatch
    demo._preserve_reader_scroll and reader._inject_toc_jump use) defines
    an own ``scrollTop`` accessor on the container that delegates reads
    (measurements stay truthful) but gates writes on a *follow* flag:

      * follow starts true — entering the chat view lands at the bottom,
        and while the user watches, the streaming reply auto-scrolls.
      * Physical user input that scrolls away from the bottom (wheel up,
        touch drag up, scroll keys — never bare scroll events, which
        Streamlit fires plenty of without the user) drops follow, and the
        hook's yank writes are silently swallowed from then on.
      * Reaching the bottom by user input, or submitting a new question
        (Enter / send button), re-engages follow — submitting jumps to
        the new question as chats do.

    Layout is untouched (the CSS "move the scroller up a level" hack was
    tried and scrolls away the app header).

    Delivery detail, learned the hard way in a live browser: the shim
    must NOT live inside the component iframe's own script. Streamlit
    reloads the st.iframe in place on some full-app reruns — reliably at
    the exact finalize rerun where the user starts scrolling away — and
    the browser silently removes event listeners whose creating iframe
    document was destroyed, which re-opened the yank window at the worst
    moment. So the iframe is only a LOADER: it injects the real script
    as a <script> element into the parent document, where the code, its
    listeners and a MutationObserver live for the rest of the page's
    lifetime (idempotent — repeated loads skip re-injecting). The
    observer re-arms the gate whenever a fresh sticky container appears
    (view switches recreate it); with no chat mounted the shim idles.

    Must be called on EVERY chat render so the loader is present
    whenever the chat view is mounted.
    """
    components.html(
        """
<script>
// Loader: inject the chat-scroll shim into the PARENT document (top
// realm) so it survives this iframe being reloaded/replaced.
(function () {
  try {
    var w = window.parent;
    var d = w.document;
    if (w.__mdllm_chat_scroll_installed) return;
    w.__mdllm_chat_scroll_installed = true;

    var s = d.createElement('script');
    // Template literal: the shim below is plain, readable JS — no
    // string escaping layers (it contains no backticks or ${).
    s.textContent = `
(function () {
  try {
    var d = document;
    var st = d.__mdllm_chat_scroll_state;
    if (!st) { st = d.__mdllm_chat_scroll_state =
               {follow: true, armed: false, el: null}; }

    function find() {
      return d.querySelector(
          '[data-testid="stAppScrollToBottomContainer"]');
    }
    function atBottom(el) {
      return el.scrollHeight - el.scrollTop - el.offsetHeight < 1;
    }
    // Bypass our own gate (used by the submit-jump below).
    function rawSet(el, v) {
      Object.getOwnPropertyDescriptor(Element.prototype, 'scrollTop')
        .set.call(el, v);
    }
    function toBottom(el) {
      rawSet(el, el.scrollHeight - el.offsetHeight);
    }

    // --- Gate scrollTop writes on the container -------------------
    // follow is USER INTENT: reset only on a fresh chat-view mount
    // (new container element) or by the user acting below.
    function arm(el) {
      if (st.el === el && st.armed) return;
      if (st.el !== el) {
        // Fresh container = the chat view (re)mounted: follow anew,
        // and let Streamlit's native first-load scroll-to-bottom pass.
        st.follow = true;
      }
      var desc = Object.getOwnPropertyDescriptor(
          Element.prototype, 'scrollTop');
      Object.defineProperty(el, 'scrollTop', {
        configurable: true,
        get: function () { return desc.get.call(el); },
        set: function (v) {
          if (st.follow) desc.set.call(el, v);
          // else: swallow — Streamlit's sticky-scroll may not move
          // the viewport. Its animation converges in value space and
          // ends on its own; only its writes are dropped.
        },
      });
      st.el = el;
      st.armed = true;
    }

    function check() {
      var el = find();
      if (el && st.el !== el) arm(el);
    }
    check();
    new MutationObserver(check).observe(
        d.body, {childList: true, subtree: true});

    // --- Real-user-input detection --------------------------------
    // Follow must ONLY be dropped by physical input. Scroll *events*
    // alone never count: Streamlit auto-scrolls fire them without any
    // user action, and inferring "the user scrolled" from them is
    // exactly the misclassification this shim exists to correct.
    // Handlers only react to input aimed at the armed chat container,
    // so scrolling another panel (sidebar, a mounted Reader) never
    // changes the chat's follow state.
    function inChat(e) {
      var s = d.__mdllm_chat_scroll_state;
      return !!(s && s.el && s.el.isConnected
                && e.target && e.target.nodeType === 1
                && s.el.contains(e.target));
    }

    function onWheel(e) {
      if (!inChat(e)) return;
      var s = d.__mdllm_chat_scroll_state;
      if (e.deltaY < 0) {
        s.follow = false;              // scrolling up: stop following
      } else if (e.deltaY > 0) {
        var el = s.el;
        setTimeout(function () {        // re-stick only if it lands at
          if (el.isConnected && atBottom(el)) s.follow = true;
        }, 80);
      }
    }

    var touchY = null;
    function onTouchStart(e) {
      if (!inChat(e)) return;
      touchY = e.touches.length ? e.touches[0].clientY : null;
    }
    function onTouchMove(e) {
      if (!inChat(e) || !e.touches.length) return;
      var y = e.touches[0].clientY;
      if (typeof touchY === 'number' && y < touchY - 4) {
        d.__mdllm_chat_scroll_state.follow = false;  // drag up: stop
      }
      touchY = y;
    }

    function onKey(e) {
      if (!inChat(e)) return;
      var s = d.__mdllm_chat_scroll_state;
      var k = e.key;
      var up = (k === 'ArrowUp' || k === 'PageUp' || k === 'Home'
                || (k === ' ' && e.shiftKey));
      var down = (k === 'ArrowDown' || k === 'PageDown' || k === 'End'
                  || (k === ' ' && !e.shiftKey));
      if (up) s.follow = false;
      if (down) {
        var el = s.el;
        setTimeout(function () {
          if (el.isConnected && atBottom(el)) s.follow = true;
        }, 80);
      }

      // Submitting a new question: re-engage and jump to it — the
      // user just typed at the bottom; the question and the working
      // indicator must come into view (the hook only auto-scrolls
      // when IT still considers itself sticky, which is unreliable).
      if (k === 'Enter' && !e.shiftKey) {
        var t = e.target;
        if (t && t.closest
            && t.closest('[data-testid="stChatInputTextArea"]')) {
          s.follow = true;
          if (s.el) toBottom(s.el);
        }
      }
    }

    function onPointer(e) {
      if (!inChat(e)) return;
      var t = e.target;
      if (t && t.closest
          && t.closest('[data-testid="stBottom"] button')) {
        var s = d.__mdllm_chat_scroll_state;
        s.follow = true;
        if (s.el) toBottom(s.el);
      }
    }

    d.addEventListener('wheel', onWheel, {capture: true, passive: true});
    d.addEventListener('touchstart', onTouchStart,
                       {capture: true, passive: true});
    d.addEventListener('touchmove', onTouchMove,
                       {capture: true, passive: true});
    d.addEventListener('keydown', onKey, {capture: true, passive: true});
    d.addEventListener('pointerdown', onPointer,
                       {capture: true, passive: true});
  } catch (e) {}
})();
`;
    (d.head || d.documentElement).appendChild(s);
  } catch (e) {}
})();
</script>
""",
        height=0,
    )


# ---------------------------------------------------------------------------
# Control continuity across view switches
# ---------------------------------------------------------------------------
#
# Streamlit drops a widget's value from session_state once that widget is no
# longer rendered on a run. A host need not mount both panels at once: the
# standalone demo renders only the active view (sidebar buttons + an
# ``if/else``), and a host app may do the same. So going LLM chat -> Reader ->
# LLM chat would otherwise reset every ``chat_*`` control (provider, model,
# endpoint, API key, autossh fields) to its default.
#
# Fix: mirror the chat panel's widget values into an ordinary (non-widget)
# session_state key. Non-widget keys are NOT pruned, so the snapshot survives
# Reader-view runs. It's taken at the end of every chat render (always current
# as of the last chat-view run, which is when any edit happens) and seeded back
# before the controls mount so the widgets pick up the prior values instead of
# their defaults. Session-memory only — nothing hits disk, so the OpenRouter
# key stays write-only and no new secret is persisted.

_PANEL_SNAPSHOT_KEY = "_chat_controls_snapshot"


def _chat_control_keys():
    """Yield the chat panel's widget-bound session_state keys.

    Covers the ``chat_llm_*`` controls (provider/model/endpoint/key) and the
    ``_chat_ssh_*`` autossh fields. Internal non-widget keys (``_chat_messages``,
    ``_chat_bg_task``, the tracked ``_chat_autossh_proc`` Popen, button keys)
    are intentionally excluded — they're either already non-prunable or hold
    objects that must not be snapshotted.
    """
    for k in st.session_state:
        if k.startswith("chat_") or k.startswith("_chat_ssh_"):
            yield k


def _snapshot_chat_controls():
    """Copy the chat panel's current widget values into a non-widget key.

    Called at the end of every chat render so the snapshot always reflects the
    latest selections the user made (each edit triggers a chat-view rerun, which
    re-runs this). The mirrored dict lives under a non-widget key, so it
    survives runs where the chat panel is not rendered.
    """
    snap = {k: st.session_state[k] for k in _chat_control_keys()}
    if snap:
        st.session_state[_PANEL_SNAPSHOT_KEY] = snap


def _restore_chat_controls():
    """Seed chat panel widget keys from the snapshot when they're absent.

    Runs before the controls mount so each widget picks up its prior value
    instead of its default. Only fills keys missing from session_state — a
    value already set this run (e.g. by an on_change callback or the pending-
    selection block below) is left untouched.
    """
    snap = st.session_state.get(_PANEL_SNAPSHOT_KEY)
    if not snap:
        return
    for k, v in snap.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_chat():
    """Render the LLM chat panel: controls, the chat UI, and the streaming reply.

    The chat always targets whatever document is open in the Reader — open one
    there, then ask about it here. There is no separate dropdown.
    """
    # --- Finalize finished background tasks BEFORE anything renders ----
    # The reply must join _CHAT_MESSAGES ahead of the history loop below so
    # this same run draws the complete conversation — one frontend commit,
    # no intermediate bubble-removed state, no second rerun. See
    # _finalize_chat_task for why that single-commit property is what keeps
    # Streamlit's sticky scroll-to-bottom container from yanking the
    # viewport down the moment the LLM finishes.
    #
    # Finalize *all* done tasks across every open document/session, not just
    # the active one: if the user switched documents while a stream was
    # running, the original document's completed reply would otherwise stay
    # dangling under its own key until the user switched back, making
    # ``Save conversation`` for that document appear empty.  Scanning all
    # docs ensures every finished reply lands in its correct conversation
    # immediately.
    _finalize_all_done_tasks()
    _task = st.session_state.get(_chat_state_key(_CHAT_BG_TASK))
    if _task and _task.get("done"):
        # Defensive: if the active task somehow still shows done (e.g. a
        # race where it completed after the scan), finalize it now.
        _finalize_chat_task(_task)
        st.session_state.pop(_chat_state_key(_CHAT_BG_TASK), None)
        _task = None
    # If the active task is still running, keep it for the streaming bubble;
    # if it was just finalized above, _task is now None so the history loop
    # draws the complete conversation.

    st.subheader("LLM chat")

    # --- Chat sessions for this document -------------------------------
    # A document can hold several independent chat sessions ("tabs"): each has
    # its own conversation, stream task, staged ⚡ Summarize prompt, and last
    # error (see docs.chat_key). Session buttons pick the active one (the
    # active session is highlighted); "+ New" opens another independent
    # conversation about the same document; "Close" drops the current one
    # (never the last remaining session).
    _doc = docs.active_document()
    sessions = docs.chat_sessions(_doc)
    cur = docs.active_chat(_doc)

    # Session switch buttons — one per session, active one highlighted.
    # Rendered only when there is more than one session (nothing to switch
    # between otherwise); "+ New" below creates additional sessions.
    if len(sessions) > 1:
        sess_cols = st.columns(len(sessions))
        for i, sid in enumerate(sessions):
            if sess_cols[i].button(
                docs.chat_session_label(_doc, sid),
                key=f"_md_llm_chat_btn_{sid}_{_doc}",
                type="primary" if sid == cur else "secondary",
                use_container_width=True,
            ):
                if sid != cur:
                    docs.set_active_chat(sid, _doc)
                    st.rerun()

    # "+ New" opens another independent conversation; "Close" drops the
    # current session (never the last remaining one).
    col_new, col_close, _ = st.columns([1, 1, 3])
    if col_new.button("+ New"):
        docs.add_chat(_doc)
        st.rerun()
    if col_close.button(
        "Close",
        disabled=len(sessions) <= 1,
    ):
        docs.remove_chat(cur, _doc)
        st.rerun()

    # Restore the chat panel's controls from the in-memory snapshot before any
    # widget mounts, so a Reader -> chat round-trip keeps the user's provider /
    # model / endpoint / key selections (Streamlit otherwise prunes unmounted
    # widget state). No-op on the first render and when the chat view never left.
    _restore_chat_controls()

    # Apply pending selections from prior chat interactions, but only if the
    # endpoint hasn't changed (API keys are paired per endpoint).
    pending_chat_oai_endpoint = st.session_state.pop("_pending_chat_oai_endpoint", None)
    current_chat_oai_endpoint = _current_oai_endpoint("chat_")
    if pending_chat_oai_endpoint and pending_chat_oai_endpoint == current_chat_oai_endpoint:
        pending_chat_oai_model = st.session_state.pop("_pending_chat_oai_model_sel", None)
        if pending_chat_oai_model is not None:
            st.session_state["chat_llm_oai_model_sel"] = pending_chat_oai_model
        pending_chat_oai_key = st.session_state.pop("_pending_chat_oai_api_key", None)
        if pending_chat_oai_key is not None:
            st.session_state["chat_llm_oai_api_key"] = pending_chat_oai_key
    else:
        st.session_state.pop("_pending_chat_oai_model_sel", None)
        st.session_state.pop("_pending_chat_oai_api_key", None)

    context_path = _current_context_path()
    if context_path:
        st.caption(
            "Discussing the document open in the Reader: "
            f"**{_display_name_for_filepath(context_path)}**. Open a different "
            "one there to switch context; its text is sent once as the leading "
            "turn, then your questions build on it."
        )
    else:
        st.caption(
            "_Nothing open in the Reader — your messages go straight to the LLM "
            "with no document context. Open a document there to discuss it._"
        )
    if docs.is_multi():
        st.caption(
            f"**{len(docs.open_documents())} documents open** — each keeps its "
            "own independent chat. Switch the active document in the sidebar; "
            "this panel always shows the active document's conversation."
        )
    st.caption(
        f"Showing **{docs.chat_session_label(_doc, cur)}** — {len(sessions)} "
        "independent chat session(s) for this document. Each session keeps "
        "its own conversation and background stream."
    )
    st.caption(
        "The provider/model/key live in the LLM controls expander below. The "
        "conversation lives in memory for this session only."
    )

    # --- Controls (the only place chat_* widgets are instantiated) ------
    # No "Instruction / prompt" field here: in a chat the prompt comes from the
    # chat box, and a fixed instruction would otherwise hijack the system
    # message. _build_stream passes instruction=None for the chat panel.
    with st.expander("LLM controls", expanded=False):
        _render_llm_controls(prefix="chat_", show_instruction=False)
        # Ollama-only remote tunnel, under chat_-namespaced keys.
        if st.session_state.get("chat_llm_provider") == "Ollama":
            _render_autossh_panel(
                prefix="chat_", default=DEFAULT_LLM_AUTOSSH,
                title="Remote tunnel to Ollama (autossh)",
            )
    # Snapshot the control values now that every chat_* / _chat_ssh_* widget has
    # mounted with its current value. The mirrored dict lives under a non-widget
    # key, so it survives the Reader-view runs that prune the widget keys.
    _snapshot_chat_controls()

    # --- Quick action staged in the Reader (⚡ Summarize) ----------------
    # The Reader's quick-action button stages its prompt and switches here;
    # send it through the normal pipeline — but only when this session has no
    # stream running. While one does, the staged prompt waits: the very rerun
    # that finalizes the running reply clears _task at the top of this
    # function, so that same run picks the queued prompt up and sends it.
    if not _task:
        _started = _send_staged_quick_prompt(context_path)
        if _started:
            _task = _started
    elif st.session_state.get(_staged_quick_prompt_key()):
        st.info(
            "A **⚡ Summarize** prompt from the Reader is staged — it sends "
            "automatically when the current reply finishes."
        )
        if st.button("Drop staged prompt", key="_chat_drop_quick_prompt"):
            st.session_state.pop(_staged_quick_prompt_key(), None)
            st.rerun()

    col_save, col_clear, _ = st.columns([1, 1, 2])
    if col_save.button("Save conversation"):
        provider = st.session_state.get("chat_llm_provider", "OpenRouter")
        model = _current_llm_model(prefix="chat_") or "(none)"
        saved = _save_conversation(context_path, provider, model)
        if saved:
            # No st.rerun() here: a rerun discards the current run's output,
            # so the success message (like any st.error/warning from
            # _save_conversation) would never reach the screen. The message
            # simply stays up until the next interaction.
            st.success(f"Conversation saved to `{saved}`")
    if col_clear.button("Clear conversation"):
        st.session_state.pop(_chat_state_key(_CHAT_MESSAGES), None)
        st.session_state.pop(_chat_state_key(_CHAT_BG_TASK), None)
        st.rerun()

    # --- Save location (memorized; default = the host's chat_save_dir) ---
    # Rendered every run, even collapsed: the on_change callback persists each
    # committed edit to settings, and the inline validation keeps the shown
    # target honest before the user clicks Save.
    with st.expander("Save location", expanded=False):
        st.caption(
            "Saved chats are written here as `<docstem>__chat_<UTC>.md`. "
            f"Host default: `{get_core().chat_save_dir}`."
        )
        st.text_input(
            "Directory",
            value=_chat_save_dir(),
            key=_SAVE_DIR_INPUT_KEY,
            on_change=_on_save_dir_input_change,
            help=(
                "Edit and press Enter to change where conversations are "
                "saved — your choice is remembered across sessions. Clear "
                "the field to fall back to the host default."
            ),
        )
        problem = _save_dir_problem(_chat_save_dir())
        if problem:
            st.error(problem)
        else:
            st.caption(
                f"Ready — the next save writes to "
                f"`{os.path.abspath(os.path.expanduser(_chat_save_dir()))}`."
            )

    st.divider()

    # Permanently bump body-text + code font size (scoped to Streamlit's
    # markdown/code containers; chat-message bodies use the same containers).
    st.markdown(_BODY_FONT_SIZE_CSS, unsafe_allow_html=True)

    # Gate Streamlit's sticky scroll-to-bottom container so it follows the
    # stream while the user is at the bottom but can never yank the viewport
    # back down after they scroll away (see _tame_chat_autoscroll). Mounted
    # on every chat render so it re-arms whenever the chat view remounts.
    _tame_chat_autoscroll()

    # --- Chat history --------------------------------------------------
    messages = st.session_state.get(_chat_state_key(_CHAT_MESSAGES)) or []
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(_escape_currency_dollars(m["content"]))

    # Surface the last failed call as a transient bubble (not stored).
    err = st.session_state.pop(_chat_state_key("_chat_last_error"), None)
    if err:
        with st.chat_message("assistant"):
            st.error(f"LLM call failed: {err}")

    # --- Background streaming task (survives tab switches) -------------
    # The LLM call runs in a daemon thread (_stream_worker), so switching to
    # the Reader mid-reply no longer cancels it — the call (and OpenCode's
    # subprocess) keeps running at the back end. A finished task was already
    # folded in at the top of this run (see _finalize_chat_task); only the
    # still-running case remains here.
    if _task:
        # Still running: the fragment renders/polls the partial reply. The
        # work continues in the background thread regardless of which tab
        # is active.
        _stream_partial_reply(_task)

        # The chat input stays mounted while a reply streams (disabled).
        # Streamlit wraps the whole main view in a sticky scroll-to-bottom
        # container whenever an st.chat_input is present, and every fresh
        # mount of that container force-scrolls the page to the bottom — so
        # unmounting the input mid-stream (the old early `return`) yanked
        # the viewport down at the remount when the stream finished. The
        # input's element id does NOT depend on `disabled`, so keeping it
        # mounted lets the container — and the user's scroll position —
        # survive the whole stream→done transition untouched.
        st.chat_input("Ask about this document…", disabled=True)
        return

    # --- Input ---------------------------------------------------------
    prompt = st.chat_input("Ask about this document…")
    if prompt:
        # Append the user turn BEFORE building the stream — _build_stream
        # snapshots the conversation at call time into the generator, so the
        # new question must already be in _CHAT_MESSAGES for the model to see.
        provider = st.session_state.get("chat_llm_provider", "OpenRouter")
        model = _current_llm_model(prefix="chat_") or "(unknown)"
        chat_src = f"LLM chat ({provider} · {model})"
        preview = prompt.strip().replace("\n", " ")[:80]
        log_event(f"Chat send → {preview}", level="info", source=chat_src)

        msgs = st.session_state.setdefault(_chat_state_key(_CHAT_MESSAGES), [])
        msgs.append({"role": "user", "content": prompt})
        holder = {}
        stream, verr = _build_stream(context_path, holder)
        if stream is None:
            msgs.pop()  # validation failed: roll back the dangling question
            st.session_state[_chat_state_key("_chat_last_error")] = verr
            log_event(f"Chat failed: {verr}", level="error", source=chat_src)
        else:
            # Hand the stream to a background thread so the call (and any
            # subprocess it drives, e.g. `opencode run`) keeps running when the
            # user switches tabs or switches to another open document's chat.
            # st.write_stream would be cancelled by the tab-switch rerun, and
            # the stream's generator kills its subprocess on close. The
            # polling fragment above drains `task` for live display.
            task = {"text": "", "done": False, "error": None, "source": chat_src}
            worker = threading.Thread(
                target=_stream_worker, args=(task, stream, holder),
                daemon=True,
            )
            st.session_state[_chat_state_key(_CHAT_BG_TASK)] = task
            worker.start()
        st.rerun()
