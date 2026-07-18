"""LLM chat panel: converse with an LLM about the document open in the Reader,
using Streamlit's native chat UI (``st.chat_message`` / ``st.chat_input``).

There is no separate document picker — the chat always follows whatever the
Reader currently has open (``_reader_target``). Open a document in the Reader,
then ask about it here. The document's full text is sent once as the leading
context turn, so the model has it in mind for the whole conversation.

The assistant reply is streamed token-by-token via ``st.write_stream`` (both
OpenRouter's SSE deltas and Ollama's newline-delimited chunks are supported).

The conversation lives in session memory (``_chat_messages``); a **Save
conversation** button writes it to ``core.chat_save_dir`` as a plain
``<docstem>__chat_<UTC>.md`` file. No sidecar metadata, no transcript linkage —
md_llm has no notion of "transcripts". The provider/model/key controls live in
this panel under the ``chat_`` key namespace.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import streamlit as st

from . import llm
from .autossh import _render_autossh_panel
from .console import log_event
from .controls import (
    _current_llm_model,
    _current_oai_endpoint,
    _oai_registry_entry,
    _remember_oai_endpoint,
    _render_llm_controls,
    _save_oai_registry_entry,
)
from .core import get_core
from .state import DEFAULT_LLM_AUTOSSH, _display_name_for_filepath, _read_text

# Session-state keys (session-memory only — nothing persisted except via Save).
_CHAT_MESSAGES = "_chat_messages"  # list[{"role","content"}]

# A passage staged in the Reader ("Send to chat") to attach to the NEXT chat
# question. Read here by literal string (matches reader.py) rather than imported
# from .reader, to keep chat↔reader decoupled. Cleared once attached.
_READER_QUOTE = "_reader_quote"


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
    for m in (st.session_state.get(_CHAT_MESSAGES) or []):
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


def _write_chat_md(context_path, text):
    """Write `text` as a ``<docstem>__chat_<UTC>.md`` in chat_save_dir.

    Pure I/O (no Streamlit calls) so it is unit-testable. Each saved chat is
    a plain markdown file named after the source document stem plus a UTC
    timestamp (so multiple chats about the same doc don't collide). No sidecar
    metadata is written — md_llm has no transcript-linkage concept.
    Returns the absolute path, or None on write failure.
    """
    save_dir = get_core().chat_save_dir
    os.makedirs(save_dir, exist_ok=True)
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


def _save_conversation(context_path, provider, model):
    """Write the conversation to chat_save_dir.

    Returns the saved absolute path, or None on failure / when the conversation
    is empty.
    """
    messages = st.session_state.get(_CHAT_MESSAGES) or []
    if not messages:
        st.warning("Nothing to save — the conversation is empty.")
        return None

    text = _render_chat_as_markdown(context_path, provider, model)
    source_title = (
        _display_name_for_filepath(context_path) if context_path else ""
    )
    _chat_default_title(source_title, messages)  # computed for parity/title hooks
    out_path = _write_chat_md(context_path, text)
    if out_path is None:
        st.error("Could not save conversation (write failed).")
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

    A quote staged in the Reader ("Send to chat") is prepended to the final user
    turn.
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
    turns = list(st.session_state.get(_CHAT_MESSAGES) or [])
    _attach_quote_to_last_turn(turns)
    messages.extend(turns)
    return messages


def _attach_quote_to_last_turn(turns):
    """Prepend any staged Reader quote to the last user turn, then clear it.

    Operates on a shallow copy of the turn list (the caller owns the live
    session-state list).
    """
    quote = st.session_state.get(_READER_QUOTE)
    if not quote:
        return
    if not turns or turns[-1].get("role") != "user":
        return
    last = dict(turns[-1])
    last["content"] = (
        f"I want to focus on this passage from the document:\n\n"
        f"> {quote}\n\n{last['content']}"
    )
    turns[-1] = last
    st.session_state.pop(_READER_QUOTE, None)


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
    else:
        endpoint = st.session_state.get(
            f"{p}llm_endpoint", llm.DEFAULT_ENDPOINT
        )
        gen = llm.ollama_chat_stream(
            turns, endpoint=endpoint, model=model, instruction=instruction,
        )
    return _safe_stream(gen, holder), None


def render_chat():
    """Render the LLM chat panel: controls, the chat UI, and the streaming reply.

    The chat always targets whatever document is open in the Reader — open one
    there, then ask about it here. There is no separate dropdown.
    """
    st.subheader("LLM chat")

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
    st.caption(
        "The provider/model/key live in the LLM controls expander below. The "
        "conversation lives in memory for this session only."
    )

    # Show a staged Reader quote so the user knows their next question carries
    # it as focused context (alongside the full document).
    staged_quote = st.session_state.get(_READER_QUOTE)
    if staged_quote:
        st.info(
            f"**Quote attached to your next question** (alongside the full "
            f"document):\n\n> {staged_quote}"
        )
        if st.button("Drop quote", help="Don't attach the staged passage to "
                     "the next question after all."):
            st.session_state.pop(_READER_QUOTE, None)
            st.rerun()

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

    col_save, col_clear, _ = st.columns([1, 1, 2])
    if col_save.button(
        "Save conversation", help="Save this chat as a .md in the configured "
        "save dir, with the source document embedded for context.",
    ):
        provider = st.session_state.get("chat_llm_provider", "OpenRouter")
        model = _current_llm_model(prefix="chat_") or "(none)"
        saved = _save_conversation(context_path, provider, model)
        if saved:
            st.success(f"Saved: `{os.path.basename(saved)}`")
            st.rerun()
    if col_clear.button("Clear conversation", help="Reset the chat history."):
        st.session_state.pop(_CHAT_MESSAGES, None)
        st.rerun()

    st.divider()

    # --- Chat history --------------------------------------------------
    messages = st.session_state.get(_CHAT_MESSAGES) or []
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Surface the last failed call as a transient bubble (not stored).
    err = st.session_state.pop("_chat_last_error", None)
    if err:
        with st.chat_message("assistant"):
            st.error(f"LLM call failed: {err}")

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

        msgs = st.session_state.setdefault(_CHAT_MESSAGES, [])
        msgs.append({"role": "user", "content": prompt})
        holder = {}
        stream, verr = _build_stream(context_path, holder)
        if stream is None:
            msgs.pop()  # validation failed: roll back the dangling question
            st.session_state["_chat_last_error"] = verr
            log_event(f"Chat failed: {verr}", level="error", source=chat_src)
        else:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                reply = st.write_stream(stream)
            if holder.get("error"):
                st.session_state["_chat_last_error"] = holder["error"]
                log_event(
                    f"Chat failed: {holder['error']}",
                    level="error", source=chat_src,
                )
            else:
                reply_text = (reply or "").strip()
                if reply_text:
                    log_event(
                        f"Chat reply ({len(reply_text)} chars)",
                        level="info", source=chat_src,
                    )
                else:
                    log_event(
                        "Chat reply empty — nothing came back.",
                        level="warn", source=chat_src,
                    )
                msgs.append({
                    "role": "assistant",
                    "content": reply_text
                    or "_(empty response — nothing came back.)",
                })
        st.rerun()
