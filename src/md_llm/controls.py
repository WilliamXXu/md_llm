"""LLM provider / model / endpoint controls + the per-endpoint OAI registry.

Split out of the host's monolithic ``ui.llm_panel``: the bits the reader and
chat need (provider radio, model dropdown, API-key field, the OpenAI-compatible
per-endpoint registry) — with NONE of the transcript-specific machinery (batch
worker, autopilot, LLM-output grid). Every persistence call goes through the
injected Core (:func:`md_llm.core.get_core`) instead of a host ``tl`` module, so
this module is host-agnostic.

Five providers, toggled by a radio:
  - **Ollama**: a local server; models auto-discovered via /api/tags.
  - **OpenRouter**: a hosted API; API key defaults to OPENROUTER_API_KEY.
    The Model dropdown auto-populates with the catalog's current free models
    (``GET /models`` — public, fetched once per session, Refresh to re-fetch).
    The last-used model and endpoint URL are memorized in settings (the API
    key stays write-only — session memory, never persisted).
  - **OpenAI-compatible**: a generic OpenAI Chat Completions API. Models AND the
    API key are remembered PER endpoint URL (the ``oai_endpoints`` registry), so
    switching endpoints restores the matching model list + key.
  - **OpenCode**: the open source coding AGENT, invoked as a subprocess
    (``opencode run --format json --auto``). No API key (auth is out-of-band
    via ``opencode auth login`` / env); instead it exposes a working directory,
    an optional ``--attach`` server URL, and an optional agent. Models — and
    each model's reasoning-effort variants — come from one
    ``opencode models --verbose`` call (cached per session with a Refresh
    button); the variant dropdown defaults to the model's highest effort.
  - **Cline**: the Cline coding AGENT CLI, also invoked as a subprocess
    (``cline --json "prompt"``, tools auto-approved). No API key (auth is
    out-of-band via ``cline auth``); it exposes a working directory and a
    reasoning-effort ``--thinking`` level (a closed set advertised by the CLI,
    so no discovery subprocess is needed). The CLI has no model-list command,
    but Cline's provider API exposes a public catalog whose zero-cost models
    carry a ``:free`` suffix — the Model dropdown auto-populates with those
    (fetched once per session, Refresh to re-fetch) on top of remembered
    history, with an "(default)" option that uses Cline's own configured
    model.

The controls are prefix-namespaced (``prefix`` arg) so several panels can each
keep independent values without their Streamlit widget keys colliding — the chat
panel renders them under ``prefix="chat_"``.
"""

from __future__ import annotations

import os

import streamlit as st

from . import docs
from . import llm
from . import sandbox
from .core import get_core


# --- helpers: read the active provider/model -------------------------------

def _current_oai_endpoint(prefix=""):
    """Return the actual OpenAI-compatible endpoint URL to use.

    ``prefix`` selects which set of widget keys to read. Returns the custom
    endpoint if "(other — type below)" is selected, otherwise the dropdown
    selection. Empty string when no endpoint is selected.
    """
    p = prefix
    endpoint_key = f"{p}llm_oai_endpoint"
    endpoint_custom_key = f"{p}llm_oai_endpoint_custom"

    dropdown_value = st.session_state.get(endpoint_key, "")
    if dropdown_value == "(other — type below)":
        return st.session_state.get(endpoint_custom_key, "").strip()
    return (dropdown_value or "").strip()


def _current_llm_model(prefix=""):
    """Return the model for whichever LLM provider is currently selected.

    ``prefix`` selects which set of widget keys to read.
    """
    p = prefix
    provider = st.session_state.get(f"{p}llm_provider", "OpenRouter")
    if provider == "OpenRouter":
        sel = st.session_state.get(f"{p}llm_or_model_sel")
        if sel and sel != "(other — type below)":
            return sel.strip()
        return st.session_state.get(
            f"{p}llm_or_model", llm.OPENROUTER_DEFAULT_MODEL
        ).strip()
    if provider == "OpenAI-compatible":
        sel = st.session_state.get(f"{p}llm_oai_model_sel")
        if sel and sel != "(other — type below)":
            return sel.strip()
        return st.session_state.get(f"{p}llm_oai_model", "").strip()
    if provider == "OpenCode":
        sel = st.session_state.get(f"{p}llm_opencode_model_sel")
        if sel and sel != "(other — type below)":
            return sel.strip()
        return st.session_state.get(f"{p}llm_opencode_model", "").strip()
    if provider == "Cline":
        sel = st.session_state.get(f"{p}llm_cline_model_sel")
        if sel and sel not in ("(other — type below)", CLINE_DEFAULT_MODEL_LABEL):
            return sel.strip()
        return st.session_state.get(f"{p}llm_cline_model", "").strip()
    sel = st.session_state.get(f"{p}llm_model_sel")
    if sel == "(other — type below)":
        return st.session_state.get(f"{p}llm_model_custom", "").strip()
    return sel or ""


# --- model / instruction history (OpenRouter) ------------------------------

def _model_history(saved_llm, key):
    """Read a remembered-model list from settings under ``key``.

    Stored as a list, most-recent-first. Returns a fresh de-duplicated copy;
    never returns None.
    """
    models = saved_llm.get(key) or []
    if not isinstance(models, list):
        return []
    seen = set()
    out = []
    for m in models:
        if isinstance(m, str) and m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _remember_model(model, key, sel_key, pending_key):
    """Promote ``model`` to the front of the model history at settings ``key``.

    Records the last-used selection in ``sel_key`` and stages it in
    ``pending_key`` so the next render applies it to the selectbox before that
    widget is instantiated (mutating a widget key post-instantiation raises).
    """
    model = (model or "").strip()
    if not model:
        return
    settings = get_core().load_settings()
    llm_s = dict(settings.get("llm") or {})
    models = [m for m in _model_history(llm_s, key) if m != model]
    llm_s[key] = [model] + models
    llm_s[sel_key] = model
    settings["llm"] = llm_s
    get_core().save_settings(settings)
    st.session_state[pending_key] = model


def _openrouter_model_history(saved_llm):
    """Read the OpenRouter model history (stored under ``llm_or_models``)."""
    return _model_history(saved_llm, "llm_or_models")


def _remember_openrouter_model(model):
    """Promote ``model`` in the OpenRouter model history on disk."""
    _remember_model(
        model, "llm_or_models", "llm_or_model_sel", "_pending_or_model_sel",
    )


def _remember_openrouter_endpoint(endpoint):
    """Persist the last-used OpenRouter endpoint URL to settings.

    Stored under ``llm_or_endpoint``; the endpoint text input mounts with that
    key's value (falling back to ``llm.OPENROUTER_DEFAULT_ENDPOINT``), so the
    memorized choice reopens next session. Empty values are ignored. Called on
    every OpenRouter send — the sibling of ``_remember_oai_endpoint`` for the
    OpenAI-compatible provider and ``_remember_opencode_model`` for OpenCode.
    """
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return
    settings = get_core().load_settings()
    llm_s = dict(settings.get("llm") or {})
    llm_s["llm_or_endpoint"] = endpoint
    settings["llm"] = llm_s
    get_core().save_settings(settings)


def _seed_openrouter_last_model(saved_llm, prefix=""):
    """Seed the model selectbox with the memorized last-used OpenRouter model.

    Copies the ``llm_or_model_sel`` settings key (written by
    :func:`_remember_openrouter_model` on every send) into the selectbox's
    session key so a fresh mount reopens on the memorized choice instead of
    the factory default — mirroring the OpenCode branch's last-model seeding.
    Only fills an ABSENT key: a selection already made this run (e.g. restored
    from the chat panel's control snapshot) always wins. Returns the seeded
    model ("" when nothing was seeded); the caller must ensure the returned
    value is among the selectbox's options — a stale memorized model is
    prepended there, exactly like a live selection that left the option list.
    """
    key = f"{prefix}llm_or_model_sel"
    if st.session_state.get(key):
        return ""
    saved = saved_llm.get("llm_or_model_sel", "")
    if saved:
        st.session_state[key] = saved
    return saved


# Session-state key of OpenRouter's per-session free-model catalog cache (see
# :func:`_openrouter_cached_models`). Lives outside the chat panel's snapshot
# prefixes on purpose: it's a cache, not a user choice.
_OPENROUTER_MODELS_CACHE_KEY = "_openrouter_models_cache"


def _openrouter_cached_models(endpoint=llm.OPENROUTER_DEFAULT_ENDPOINT):
    """Return OpenRouter's free-model catalog, fetched once per session.

    ``GET /models`` is a remote HTTPS call (~1 MB catalog), so the result is
    cached in session_state — mirroring the OpenCode discovery cache — and
    refreshed on demand via the Refresh button in the OpenRouter controls; a
    fresh browser session re-fetches automatically, so the list tracks the
    live catalog without paying a round trip on every rerun. [] means the
    fetch failed (offline, endpoint down) and the dropdown degrades to the
    factory default + remembered history until a Refresh succeeds.
    """
    cache = st.session_state.get(_OPENROUTER_MODELS_CACHE_KEY)
    if cache is None:
        cache = llm.list_openrouter_models(endpoint)
        st.session_state[_OPENROUTER_MODELS_CACHE_KEY] = cache
    return list(cache)


def _openrouter_dropdown_options(saved_llm, discovered):
    """Build the OpenRouter Model dropdown's option list.

    Order: the factory default first (it stays the default selection when
    nothing is memorized), then the remembered model history, then the
    discovered free-model catalog (see :func:`_openrouter_cached_models`), and
    the manual-entry escape hatch last. De-duplicated, order-preserving.
    """
    models = _openrouter_model_history(saved_llm)
    if not any(m == llm.OPENROUTER_DEFAULT_MODEL for m in models):
        models = [llm.OPENROUTER_DEFAULT_MODEL] + models
    merged: list[str] = []
    for m in list(models) + list(discovered or []):
        if isinstance(m, str) and m and m not in merged:
            merged.append(m)
    return merged + ["(other — type below)"]


def _instruction_history(saved_llm):
    """Read the list of previously used instructions/prompts from settings."""
    items = saved_llm.get("llm_instruction_history") or []
    if not isinstance(items, list):
        return []
    seen = set()
    out = []
    for it in items:
        if isinstance(it, str) and it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _remember_instruction(instruction):
    """Promote `instruction` to the front of the prompt history on disk.

    Caps the list at the most recent 10 entries.
    """
    instruction = (instruction or "").strip()
    if not instruction:
        return
    settings = get_core().load_settings()
    llm_s = dict(settings.get("llm") or {})
    items = [i for i in _instruction_history(llm_s) if i != instruction]
    items = [instruction] + items
    llm_s["llm_instruction_history"] = items[:10]
    settings["llm"] = llm_s
    get_core().save_settings(settings)


# --- OpenAI-compatible per-endpoint registry --------------------------------
#
# Models AND the API key are remembered PER endpoint URL. The registry lives
# under the ``oai_endpoints`` settings key as a map keyed by the (normalized)
# endpoint base URL:
#
#   "oai_endpoints": {
#       "https://api.groq.com/openai/v1": {
#           "models": ["qwen/qwen3-32b", "gpt-4o-mini"],  # most-recent-first
#           "last_model": "qwen/qwen3-32b",
#           "api_key": "gsk_..."
#       },
#       ...
#   }

_OAI_REGISTRY_KEY = "oai_endpoints"

# Sentinel for "argument not passed" (None is a valid value for some fields).
_UNSET = object()


def _normalize_oai_endpoint(endpoint):
    """Normalize an endpoint URL for use as a registry key (strip trailing /)."""
    return (endpoint or "").strip().rstrip("/")


def _oai_registry(saved_llm):
    """Return the oai_endpoints registry dict (a fresh copy, never None)."""
    reg = (saved_llm or {}).get(_OAI_REGISTRY_KEY) or {}
    return dict(reg) if isinstance(reg, dict) else {}


def _oai_registry_entry(saved_llm, endpoint):
    """Return one endpoint's registry entry as a fresh dict (never None).

    Always returns a dict with ``models`` (list), ``last_model`` (str) and
    ``api_key`` (str) keys, defaulting to empty — callers never need to guard.
    """
    reg = _oai_registry(saved_llm)
    entry = reg.get(_normalize_oai_endpoint(endpoint)) or {}
    if not isinstance(entry, dict):
        return {"models": [], "last_model": "", "api_key": ""}
    return {
        "models": list(entry.get("models") or []),
        "last_model": entry.get("last_model") or "",
        "api_key": entry.get("api_key") or "",
    }


def _save_oai_registry_entry(endpoint, *, models=None, last_model=None,
                             api_key=_UNSET, pending_model_key=None,
                             pending_api_key_key=None):
    """Update one endpoint's registry entry on disk, merging into stored state.

    ``models`` replaces the model list if given; ``last_model`` sets the
    last-used model and is also promoted to the front of ``models`` (de-duped);
    ``api_key`` sets the key if given (pass "" to clear); ``api_key`` defaults
    to a sentinel so it's left untouched when unspecified. ``pending_*`` keys
    stage values for the next render (so a freshly-remembered selection lands
    before the widget instantiates).
    """
    endpoint = _normalize_oai_endpoint(endpoint)
    if not endpoint:
        return
    settings = get_core().load_settings()
    llm_s = dict(settings.get("llm") or {})
    reg = _oai_registry(llm_s)
    entry = reg.get(endpoint) or {}
    if not isinstance(entry, dict):
        entry = {}
    entry = dict(entry)

    if last_model:
        existing = [m for m in (entry.get("models") or []) if m != last_model]
        entry["models"] = [last_model] + existing
        entry["last_model"] = last_model
    elif models is not None:
        entry["models"] = list(models)

    if api_key is not _UNSET:
        entry["api_key"] = api_key or ""

    reg[endpoint] = entry
    llm_s[_OAI_REGISTRY_KEY] = reg
    settings["llm"] = llm_s
    get_core().save_settings(settings)

    # Stage values for the next render; also stage the endpoint itself so a
    # caller can verify it hasn't changed before applying.
    pending_endpoint_key = (
        pending_model_key.replace("_model_sel", "_endpoint")
        if pending_model_key else None
    )
    if pending_endpoint_key:
        st.session_state[pending_endpoint_key] = endpoint
    if last_model and pending_model_key:
        st.session_state[pending_model_key] = last_model
    if api_key is not _UNSET and pending_api_key_key:
        st.session_state[pending_api_key_key] = api_key or ""


def _oai_known_endpoints(saved_llm):
    """Return (list of configured endpoint URLs, last-used endpoint)."""
    reg = _oai_registry(saved_llm)
    endpoints = list(reg.keys())
    last_used = saved_llm.get("llm_oai_last_endpoint", "")
    return endpoints, last_used


def _remember_oai_endpoint(endpoint):
    """Mark ``endpoint`` as the most recently used OpenAI-compatible endpoint."""
    endpoint = _normalize_oai_endpoint(endpoint)
    if not endpoint:
        return
    settings = get_core().load_settings()
    llm_s = dict(settings.get("llm") or {})
    llm_s["llm_oai_last_endpoint"] = endpoint
    settings["llm"] = llm_s
    get_core().save_settings(settings)


# --- control widgets --------------------------------------------------------

# --- OpenCode (coding agent, subprocess path) -------------------------------
#
# OpenCode is not an LLM API — it's an agent CLI (`opencode run`). So its
# controls look different from the HTTP providers: there's no API key (auth is
# done out-of-band via `opencode auth login` / env), and there's a working
# directory the agent operates in. Models — and each model's reasoning-effort
# variants — come from one `opencode models --verbose` call (cached per session
# + a Refresh button, since it's a subprocess call); the variant dropdown shows
# the selected model's real variants and defaults to the highest effort.

def _opencode_cached_details():
    """Return opencode's per-model metadata, memoized in session_state.

    ``opencode models --verbose`` is a subprocess call, so the result is cached
    in session_state and refreshed on demand via the Refresh button in
    :func:`_render_opencode_controls`. {} means discovery is unavailable (old
    opencode, binary missing) and the UI falls back to static presets.
    """
    cache = st.session_state.get("_opencode_model_details_cache")
    if cache is None:
        cache = llm.list_opencode_model_details()
        st.session_state["_opencode_model_details_cache"] = cache
    return cache


def _opencode_cached_models():
    """Return opencode's model list, memoized in session_state per session.

    Derived from the ``opencode models --verbose`` metadata cache when that
    succeeds (one subprocess call feeds both the model dropdown and the
    per-model variant options); falls back to the plain ``opencode models``
    table when verbose discovery fails (older opencode).
    """
    details = _opencode_cached_details()
    if details:
        return list(details.keys())
    cache = st.session_state.get("_opencode_models_cache")
    if cache is None:
        cache = llm.list_opencode_models()
        st.session_state["_opencode_models_cache"] = cache
    return list(cache)


def _remember_opencode_model(model):
    """Persist ``model`` as the most recently used opencode model on disk."""
    model = (model or "").strip()
    if not model:
        return
    settings = get_core().load_settings()
    llm_s = dict(settings.get("llm") or {})
    existing = [m for m in (llm_s.get("llm_opencode_models") or []) if m != model]
    llm_s["llm_opencode_models"] = [model] + existing
    llm_s["llm_opencode_last_model"] = model
    settings["llm"] = llm_s
    get_core().save_settings(settings)


def _current_opencode_variant(prefix=""):
    """Return the resolved opencode model variant, or None to omit ``--variant``.

    ``prefix`` selects which set of widget keys to read. Returns the custom
    value when "(other — type below)" is selected, the dropdown selection when a
    concrete variant is picked, and None for "(none)" / unset. The dropdown's
    default (highest effort for the selected model) is seeded when the widget
    renders, so an untouched panel resolves to that default here.
    """
    p = prefix
    sel = st.session_state.get(f"{p}llm_opencode_variant_sel")
    if sel == "(other — type below)":
        return (st.session_state.get(f"{p}llm_opencode_variant") or "").strip() or None
    if sel and sel != "(none)":
        return sel
    return None


def _selected_opencode_model(prefix):
    """The model id the OpenCode panel currently targets ('' when unresolved).

    Reads the model dropdown's key (instantiated by the time the variant
    controls render), or the custom-name input when "(other — type below)" is
    selected.
    """
    sel = st.session_state.get(f"{prefix}llm_opencode_model_sel")
    if sel == "(other — type below)":
        return (st.session_state.get(f"{prefix}llm_opencode_model") or "").strip()
    return sel or ""


def _opencode_variant_options(details, model_id):
    """Compute the Model variant dropdown's (options, default selection).

    With per-model variants discovered via ``opencode models --verbose``, the
    options are that model's real variants sorted least → most effort and the
    default is the highest one ("default to highest effort"). A model known to
    have no variants gets only the escape hatches — passing ``--variant`` to a
    variant-less model errors at runtime. When discovery is unavailable at all
    ({} details), falls back to the static presets in
    :data:`llm.OPENCODE_VARIANTS`.
    """
    if details:
        variants = llm.opencode_variants_for(details, model_id)
        if variants:
            return (
                ["(none)"] + llm.order_opencode_variants(variants)
                + ["(other — type below)"],
                llm.highest_opencode_variant(variants),
            )
        if model_id:
            return ["(none)", "(other — type below)"], "(none)"
    return (
        ["(none)"] + list(llm.OPENCODE_VARIANTS) + ["(other — type below)"],
        "(none)",
    )


# The OpenCode "clear sandbox" button key. It is suffixed per panel
# (``f"_opencode_clear_sandbox{p.rstrip('_')}"``) so the manual (prefix="") and
# chat (prefix="chat_") panels don't emit the same auto/button ID. The leading
# "_" (and the fact it never starts with "chat_") keeps it outside the chat_*
# snapshot, which would otherwise re-inject a BUTTON key and raise
# StreamlitValueAssignmentNotAllowedError. This constant is the chat-panel
# variant, used by tests to assert the snapshot-exclusion contract.
OPENCODE_CLEAR_SANDBOX_KEY = "_opencode_clear_sandboxchat"


def _render_opencode_controls(prefix, saved_llm):
    """Render the OpenCode provider's model / variant / sandbox / agent controls.

    Models are merged from ``opencode models`` (cached) and the user's
    previously-used list; each model's reasoning-effort variants come from the
    same ``opencode models --verbose`` metadata, and the variant dropdown
    defaults to the model's highest effort. By default the agent runs in a
    hardened per-chat sandbox (:mod:`md_llm.sandbox`): a fresh, Seatbelt-
    confined directory that no other session can see. The user may override
    the workdir to point at a real project; the confinement toggle still
    applies.
    """
    p = prefix

    discovered = _opencode_cached_models()
    history = [
        m for m in (saved_llm.get("llm_opencode_models") or [])
        if isinstance(m, str) and m
    ]
    last_model = saved_llm.get("llm_opencode_last_model", "")

    # Merge history (most-recent-first) + discovered, de-duplicated.
    merged: list[str] = []
    for m in list(history) + discovered:
        if m and m not in merged:
            merged.append(m)
    if last_model and last_model not in merged:
        merged = [last_model] + merged

    options = merged + ["(other — type below)"]
    sel = st.session_state.get(f"{p}llm_opencode_model_sel")
    if not sel and last_model and last_model in options:
        st.session_state[f"{p}llm_opencode_model_sel"] = last_model
    if sel and sel != "(other — type below)" and sel not in options:
        options = [sel] + options

    scol1, scol2 = st.columns([4, 1])
    scol1.selectbox(
        "Model (provider/model)",
        options,
        key=f"{p}llm_opencode_model_sel",
    )
    if scol2.button("Refresh", key=f"_opencode_refresh{p.rstrip('_')}"):
        st.session_state.pop("_opencode_models_cache", None)
        st.session_state.pop("_opencode_model_details_cache", None)
        st.rerun()
    if st.session_state.get(f"{p}llm_opencode_model_sel") == "(other — type below)":
        st.text_input(
            "Custom model name",
            value=saved_llm.get(f"{p}llm_opencode_model", ""),
            key=f"{p}llm_opencode_model",
        )

    # Variant options follow the selected model (the dropdown key is populated
    # by the selectbox above, custom names included). Seed the default — the
    # model's highest-effort variant — on first render, and reset a stale
    # choice the current model no longer offers.
    variant_options, variant_default = _opencode_variant_options(
        _opencode_cached_details(), _selected_opencode_model(p)
    )
    vkey = f"{p}llm_opencode_variant_sel"
    if st.session_state.get(vkey) not in variant_options:
        st.session_state[vkey] = variant_default
    st.selectbox(
        "Model variant",
        variant_options,
        key=vkey,
        help=(
            "Reasoning effort passed to `opencode run --variant`. Options are "
            "the selected model's own variants (via `opencode models "
            "--verbose`); the highest one is preselected. \"(none)\" omits "
            "the flag and uses opencode's per-model default."
        ),
    )
    if st.session_state.get(f"{p}llm_opencode_variant_sel") == "(other — type below)":
        st.text_input(
            "Custom variant",
            value=saved_llm.get(f"{p}llm_opencode_variant", ""),
            key=f"{p}llm_opencode_variant",
        )

    st.checkbox(
        "Hardened sandbox (Seatbelt)",
        value=saved_llm.get(f"{p}llm_opencode_hardened", True),
        key=f"{p}llm_opencode_hardened",
        help=(
            "Run the agent under a macOS Seatbelt profile: writes are confined "
            "to the working directory plus scratch space, and reads of this "
            "app's data folder (uploads, chats, settings) and of credential "
            "stores (~/.ssh, ~/.gnupg, ...) are blocked. Network stays open "
            "for the model API."
        ),
    )
    st.text_input(
        "Working directory (optional override)",
        value=saved_llm.get(f"{p}llm_opencode_workdir", ""),
        key=f"{p}llm_opencode_workdir",
        placeholder="fresh per-chat sandbox (recommended)",
        help=(
            "Leave empty to give each chat session its own fresh sandbox "
            "directory — cleared before use, garbage-collected after. Enter a "
            "path to pin a real project directory instead; it is never wiped "
            "automatically. The legacy default (.opencode-sandbox inside the "
            "data folder) now also means managed mode."
        ),
    )
    if st.button("Clear this chat's sandbox", key=f"_opencode_clear_sandbox{p.rstrip('_')}"):
        doc = docs.active_document()
        sb_key = docs.chat_key("_opencode_sandbox", docs.active_chat(doc), doc)
        path = st.session_state.pop(sb_key, None)
        if path and sandbox.clear_sandbox(path):
            st.toast("Sandbox directory deleted.", icon="🧹")
        else:
            st.caption("No active sandbox yet — it is created on first send.")
    st.text_input(
        "Attach to server (optional)",
        value=saved_llm.get(f"{p}llm_opencode_attach", ""),
        key=f"{p}llm_opencode_attach",
        placeholder="e.g. http://localhost:4096",
    )
    st.text_input(
        "Agent (optional)",
        value=saved_llm.get(f"{p}llm_opencode_agent", ""),
        key=f"{p}llm_opencode_agent",
    )
    st.caption(
        "_OpenCode runs as a full agent with `--auto` (tools auto-approved in "
        "the working directory). Authenticate models out-of-band via "
        "`opencode auth login` or provider env vars._"
    )


# --- Cline (coding agent CLI, subprocess path) -------------------------------
#
# Cline mirrors the OpenCode controls where the concepts match (hardened
# sandbox, working directory, per-chat sandbox clearing) and diverges where
# the CLI does: there is no `cline models` discovery subprocess, so the model
# is a remembered-history selectbox with an explicit "(default …)" option that
# omits --model (Cline then uses whatever `cline auth` configured), and the
# reasoning-effort knob is --thinking — a closed set the CLI advertises, so
# the dropdown is static instead of discovered per model.

# Selectbox option meaning "pass no --model; use the model Cline was
# configured with via `cline auth`". Resolves to "" in _current_llm_model.
CLINE_DEFAULT_MODEL_LABEL = "(default — configured via cline auth)"

# The Cline "clear sandbox" button key — suffixed per panel exactly like
# OPENCODE_CLEAR_SANDBOX_KEY, and likewise never captured by the chat_* snapshot.
CLINE_CLEAR_SANDBOX_KEY = "_cline_clear_sandboxchat"

# Session-state key of Cline's per-session free-model catalog cache (see
# :func:`_cline_cached_models`). Lives outside the chat panel's snapshot
# prefixes on purpose: it's a cache, not a user choice.
_CLINE_MODELS_CACHE_KEY = "_cline_models_cache"


def _cline_cached_models(endpoint=llm.CLINE_API_ENDPOINT):
    """Return Cline's free-model catalog, fetched once per session.

    ``GET {api}/models`` is a remote HTTPS call, so the result is cached in
    session_state — mirroring the OpenRouter and OpenCode discovery caches —
    and refreshed on demand via the Refresh button in the Cline controls; a
    fresh browser session re-fetches automatically. [] means the fetch failed
    (offline, endpoint down) and the dropdown degrades to the "(default)"
    option + remembered history until a Refresh succeeds.
    """
    cache = st.session_state.get(_CLINE_MODELS_CACHE_KEY)
    if cache is None:
        cache = llm.list_cline_models(endpoint)
        st.session_state[_CLINE_MODELS_CACHE_KEY] = cache
    return list(cache)


def _current_cline_thinking(prefix=""):
    """Return the resolved cline ``--thinking`` level, or None to omit it.

    ``prefix`` selects which set of widget keys to read. "(provider default)"
    (the dropdown default) resolves to None — the flag is omitted and Cline's
    provider default applies; any concrete level passes through unchanged.
    """
    sel = st.session_state.get(f"{prefix}llm_cline_thinking_sel")
    if sel and sel != "(provider default)":
        return sel
    return None


def _remember_cline_model(model):
    """Persist ``model`` as the most recently used cline model on disk."""
    model = (model or "").strip()
    if not model:
        return
    settings = get_core().load_settings()
    llm_s = dict(settings.get("llm") or {})
    existing = [m for m in (llm_s.get("llm_cline_models") or []) if m != model]
    llm_s["llm_cline_models"] = [model] + existing
    llm_s["llm_cline_last_model"] = model
    settings["llm"] = llm_s
    get_core().save_settings(settings)


def _render_cline_controls(prefix, saved_llm):
    """Render the Cline provider's model / thinking / sandbox controls.

    The model dropdown merges the remembered history with an explicit
    "(default — configured via cline auth)" option; typing a custom model id
    is the "(other — type below)" escape hatch. The thinking dropdown is the
    CLI's own closed ``--thinking`` level set. By default the agent runs in a
    hardened per-chat sandbox (:mod:`md_llm.sandbox`) — the SAME per-session
    directory the OpenCode provider uses, so switching providers mid-session
    keeps the sandbox contents.
    """
    p = prefix

    history = [
        m for m in (saved_llm.get("llm_cline_models") or [])
        if isinstance(m, str) and m
    ]
    last_model = saved_llm.get("llm_cline_last_model", "")

    # Merge history (most-recent-first) + the discovered free catalog,
    # de-duplicated — order-preserving like the OpenRouter dropdown.
    discovered = _cline_cached_models()
    merged: list[str] = []
    for m in [last_model] + history + discovered:
        if m and m not in merged:
            merged.append(m)

    options = [CLINE_DEFAULT_MODEL_LABEL] + merged + ["(other — type below)"]
    sel = st.session_state.get(f"{p}llm_cline_model_sel")
    if not sel and last_model and last_model in options:
        st.session_state[f"{p}llm_cline_model_sel"] = last_model
    if sel and sel not in options:
        options = [sel] + options

    mcol, rcol = st.columns([4, 1])
    mcol.selectbox(
        "Model (default = cline's configured model)",
        options,
        key=f"{p}llm_cline_model_sel",
        help=(
            "Free models from Cline's public catalog (fetched once per "
            "session; Refresh re-fetches), plus models you have already used "
            "here. Keep \"(default)\" to use whatever `cline auth` configured, "
            "or type a custom id. Note: cline persists the model it is handed "
            "as its own new default, so runs outside this app will pick it "
            "up too."
        ),
    )
    if rcol.button("Refresh", key=f"_cline_refresh{p.rstrip('_')}"):
        st.session_state.pop(_CLINE_MODELS_CACHE_KEY, None)
        st.rerun()
    if st.session_state.get(f"{p}llm_cline_model_sel") == "(other — type below)":
        st.text_input(
            "Custom model id",
            value=saved_llm.get(f"{p}llm_cline_model", ""),
            key=f"{p}llm_cline_model",
        )

    st.selectbox(
        "Thinking level",
        ["(provider default)"] + list(llm.CLINE_THINKING_LEVELS),
        key=f"{p}llm_cline_thinking_sel",
        help=(
            "Reasoning effort passed to `cline --thinking`. \"(provider "
            "default)\" omits the flag and leaves the provider's own default."
        ),
    )

    st.checkbox(
        "Hardened sandbox (Seatbelt)",
        value=saved_llm.get(f"{p}llm_cline_hardened", True),
        key=f"{p}llm_cline_hardened",
        help=(
            "Run the agent under a macOS Seatbelt profile: writes are confined "
            "to the working directory plus scratch space, and reads of this "
            "app's data folder (uploads, chats, settings) and of credential "
            "stores (~/.ssh, ~/.gnupg, ...) are blocked. Network stays open "
            "for the model API."
        ),
    )
    st.text_input(
        "Working directory (optional override)",
        value=saved_llm.get(f"{p}llm_cline_workdir", ""),
        key=f"{p}llm_cline_workdir",
        placeholder="fresh per-chat sandbox (recommended)",
        help=(
            "Leave empty to give each chat session its own fresh sandbox "
            "directory — cleared before use, garbage-collected after. Enter a "
            "path to pin a real project directory instead; it is never wiped "
            "automatically. Shared with the OpenCode provider's sandbox."
        ),
    )
    if st.button("Clear this chat's sandbox", key=f"_cline_clear_sandbox{p.rstrip('_')}"):
        doc = docs.active_document()
        sb_key = docs.chat_key("_opencode_sandbox", docs.active_chat(doc), doc)
        path = st.session_state.pop(sb_key, None)
        if path and sandbox.clear_sandbox(path):
            st.toast("Sandbox directory deleted.", icon="🧹")
        else:
            st.caption("No active sandbox yet — it is created on first send.")
    st.caption(
        "_Cline runs headless with all tools auto-approved (`--auto-approve "
        "true`). Authenticate models out-of-band via `cline auth` or provider "
        "env vars._"
    )


# --- control widgets --------------------------------------------------------

def _on_oai_endpoint_change(prefix):
    """on_change callback for the OpenAI-compatible endpoint selector.

    When the user selects/types an endpoint, reload that endpoint's remembered
    model list + API key into the panel's session-state keys — this is what
    makes the model dropdown + key field "follow" the current endpoint. Runs
    before the widgets re-instantiate on this render.
    """
    p = prefix
    endpoint_key = f"{p}llm_oai_endpoint"
    endpoint_custom_key = f"{p}llm_oai_endpoint_custom"
    model_sel_key = f"{p}llm_oai_model_sel"
    api_key_state_key = f"{p}llm_oai_api_key"

    dropdown_value = st.session_state.get(endpoint_key, "")
    if dropdown_value == "(other — type below)":
        endpoint = st.session_state.get(endpoint_custom_key, "").strip()
        if not endpoint:
            return
        # Automatically add a newly-typed endpoint to the registry so it shows
        # up in the dropdown next time.
        _save_oai_registry_entry(endpoint)
    else:
        endpoint = dropdown_value.strip()

    if not endpoint:
        return

    saved_llm = get_core().load_settings().get("llm") or {}
    entry = _oai_registry_entry(saved_llm, endpoint)

    # Clear any stale model selection from a different endpoint.
    cur_sel = st.session_state.get(model_sel_key)
    if cur_sel and cur_sel != "(other — type below)":
        if cur_sel not in entry["models"] and cur_sel != entry["last_model"]:
            st.session_state.pop(model_sel_key, None)
            cur_sel = None

    last_model = entry["last_model"]
    if last_model:
        st.session_state[model_sel_key] = last_model
    elif not cur_sel:
        st.session_state.pop(model_sel_key, None)

    if entry["api_key"]:
        st.session_state[api_key_state_key] = entry["api_key"]
    elif "OPENAI_API_KEY" not in os.environ:
        st.session_state[api_key_state_key] = ""


def _render_oai_controls(prefix, saved_llm):
    """Render the OpenAI-compatible provider's endpoint/model/key controls.

    Models AND the API key are remembered per endpoint URL via the shared
    ``oai_endpoints`` registry, so switching endpoints restores the matching
    model list + key. The endpoint field's on_change reloads the model + key
    for the newly-typed endpoint.
    """
    p = prefix
    endpoint_key = f"{p}llm_oai_endpoint"
    endpoint_custom_key = f"{p}llm_oai_endpoint_custom"

    known_endpoints, last_used_endpoint = _oai_known_endpoints(saved_llm)
    options = known_endpoints + ["(other — type below)"]

    panel_endpoint = saved_llm.get(endpoint_key, "")
    if not panel_endpoint and last_used_endpoint:
        panel_endpoint = last_used_endpoint

    # Preserve a prior selection no longer in the known list so the selectbox
    # never errors on a missing value.
    current_endpoint = st.session_state.get(endpoint_key, panel_endpoint)
    if (current_endpoint and current_endpoint != "(other — type below)"
            and current_endpoint not in options):
        options = [current_endpoint] + options

    st.selectbox(
        "OpenAI-compatible endpoint",
        options,
        key=endpoint_key,
        on_change=_on_oai_endpoint_change, args=(prefix,),
    )

    if st.session_state.get(endpoint_key) == "(other — type below)":
        st.text_input(
            "Custom endpoint URL",
            value=saved_llm.get(endpoint_custom_key, ""),
            key=endpoint_custom_key,
            on_change=_on_oai_endpoint_change, args=(prefix,),
        )

    # Models are scoped per-endpoint: only models actually used with THIS
    # endpoint appear here — no global default is injected.
    actual_endpoint = _current_oai_endpoint(prefix)
    entry = _oai_registry_entry(saved_llm, actual_endpoint)
    models = list(entry["models"])
    options = models + ["(other — type below)"]
    sel = st.session_state.get(f"{p}llm_oai_model_sel")
    if sel and sel != "(other — type below)" and sel not in options:
        # Preserve only if it belongs to this endpoint; drop stale carry-overs.
        if sel == entry["last_model"] or sel in entry["models"]:
            options = [sel] + options
        else:
            sel = None
    if not sel and entry["last_model"]:
        if entry["last_model"] in options:
            st.session_state[f"{p}llm_oai_model_sel"] = entry["last_model"]

    st.selectbox(
        "Model",
        options,
        key=f"{p}llm_oai_model_sel",
    )
    if st.session_state.get(f"{p}llm_oai_model_sel") == "(other — type below)":
        st.text_input(
            "Custom model name",
            value=saved_llm.get(f"{p}llm_oai_model", ""),
            key=f"{p}llm_oai_model",
        )
    _oai_key_ph = (
        "Using OPENAI_API_KEY from env (paste to override)"
        if os.environ.get("OPENAI_API_KEY") and not entry["api_key"]
        else "Paste API key"
    )
    st.text_input(
        "API key",
        type="password",
        key=f"{p}llm_oai_api_key",
        placeholder=_oai_key_ph,
    )


def _render_llm_controls(prefix="", show_instruction=True):
    """Render provider/endpoint/model/api-key/instruction controls.

    ``prefix`` namespaces every widget key so several panels (manual / autopilot
    / chat) each keep independent values. ``show_instruction=False`` hides the
    "Instruction / prompt" field — used by the chat tab, where the prompt comes
    from the chat box instead.
    """
    p = prefix
    saved_llm = get_core().load_settings().get("llm") or {}
    provider = st.radio(
        "Provider",
        ["OpenCode", "Cline", "OpenRouter", "Ollama", "OpenAI-compatible"],
        horizontal=True,
        key=f"{p}llm_provider",
    )

    if provider == "Ollama":
        st.text_input(
            "Ollama endpoint",
            value=saved_llm.get("llm_endpoint", llm.DEFAULT_ENDPOINT),
            key=f"{p}llm_endpoint",
        )
        models = llm.list_ollama_models(
            st.session_state.get(f"{p}llm_endpoint", llm.DEFAULT_ENDPOINT)
        )
        model_options = models + ["(other — type below)"]
        sel = st.session_state.get(f"{p}llm_model_sel")
        if sel and sel not in model_options:
            model_options = [sel] + model_options
        st.selectbox("Model", model_options, key=f"{p}llm_model_sel")
        if st.session_state.get(f"{p}llm_model_sel") == "(other — type below)":
            st.text_input("Custom model name", key=f"{p}llm_model_custom")
    elif provider == "OpenAI-compatible":
        _render_oai_controls(prefix, saved_llm)
    elif provider == "OpenCode":
        _render_opencode_controls(prefix, saved_llm)
    elif provider == "Cline":
        _render_cline_controls(prefix, saved_llm)
    else:
        st.text_input(
            "OpenRouter endpoint",
            value=saved_llm.get("llm_or_endpoint", llm.OPENROUTER_DEFAULT_ENDPOINT),
            key=f"{p}llm_or_endpoint",
        )
        # Live free-model catalog: fetched once per session (Refresh re-fetches),
        # merged with the remembered history so both stay selectable.
        discovered = _openrouter_cached_models(
            st.session_state.get(
                f"{p}llm_or_endpoint", llm.OPENROUTER_DEFAULT_ENDPOINT
            )
        )
        options = _openrouter_dropdown_options(saved_llm, discovered)
        sel = st.session_state.get(f"{p}llm_or_model_sel")
        if not sel:
            sel = _seed_openrouter_last_model(saved_llm, p)
        if sel and sel not in options:
            options = [sel] + options
        scol1, scol2 = st.columns([4, 1])
        scol1.selectbox(
            "Model",
            options,
            key=f"{p}llm_or_model_sel",
        )
        if scol2.button("Refresh", key=f"_openrouter_refresh{p.rstrip('_')}"):
            st.session_state.pop(_OPENROUTER_MODELS_CACHE_KEY, None)
            st.rerun()
        if st.session_state.get(f"{p}llm_or_model_sel") == "(other — type below)":
            st.text_input(
                "Custom model name",
                value=saved_llm.get(f"{p}llm_or_model", ""),
                key=f"{p}llm_or_model",
            )
        _or_key_ph = (
            "Using OPENROUTER_API_KEY from env (paste to override)"
            if os.environ.get("OPENROUTER_API_KEY")
            else "Paste OpenRouter API key"
        )
        st.text_input(
            "API key",
            type="password",
            key=f"{p}llm_or_api_key",
            placeholder=_or_key_ph,
        )

    if show_instruction:
        st.text_input(
            "Instruction / prompt",
            value=saved_llm.get(f"{p}llm_instruction", llm.DEFAULT_INSTRUCTION),
            key=f"{p}llm_instruction",
        )
