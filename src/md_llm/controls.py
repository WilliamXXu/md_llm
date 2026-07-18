"""LLM provider / model / endpoint controls + the per-endpoint OAI registry.

Split out of the host's monolithic ``ui.llm_panel``: the bits the reader and
chat need (provider radio, model dropdown, API-key field, the OpenAI-compatible
per-endpoint registry) — with NONE of the transcript-specific machinery (batch
worker, autopilot, LLM-output grid). Every persistence call goes through the
injected Core (:func:`md_llm.core.get_core`) instead of a host ``tl`` module, so
this module is host-agnostic.

Three providers, toggled by a radio:
  - **Ollama**: a local server; models auto-discovered via /api/tags.
  - **OpenRouter**: a hosted API; API key defaults to OPENROUTER_API_KEY.
  - **OpenAI-compatible**: a generic OpenAI Chat Completions API. Models AND the
    API key are remembered PER endpoint URL (the ``oai_endpoints`` registry), so
    switching endpoints restores the matching model list + key.

The controls are prefix-namespaced (``prefix`` arg) so several panels can each
keep independent values without their Streamlit widget keys colliding — the chat
panel renders them under ``prefix="chat_"``.
"""

from __future__ import annotations

import os

import streamlit as st

from . import llm
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
        help="Previously used endpoints are remembered here. Pick "
             "\"(other — type below)\" to type a new one. Models and the API key "
             "are remembered per endpoint, so switching here restores them.",
    )

    if st.session_state.get(endpoint_key) == "(other — type below)":
        st.text_input(
            "Custom endpoint URL",
            value=saved_llm.get(endpoint_custom_key, ""),
            key=endpoint_custom_key,
            on_change=_on_oai_endpoint_change, args=(prefix,),
            help="Any OpenAI-compatible base URL. e.g. "
                 "https://api.openai.com/v1, "
                 "https://api.groq.com/openai/v1, "
                 "https://api.together.xyz/v1",
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
        help="Previously used models for this endpoint are remembered here. "
             "Pick \"(other — type below)\" to type a new one.",
    )
    if st.session_state.get(f"{p}llm_oai_model_sel") == "(other — type below)":
        st.text_input(
            "Custom model name",
            value=saved_llm.get(f"{p}llm_oai_model", ""),
            key=f"{p}llm_oai_model",
            help="e.g. gpt-4o-mini, qwen/qwen3-32b, "
                 "meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
        help="Remembered per endpoint. Leave empty to fall back to the "
             "OPENAI_API_KEY env var.",
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
        ["OpenRouter", "Ollama", "OpenAI-compatible"],
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
    else:
        st.text_input(
            "OpenRouter endpoint",
            value=saved_llm.get("llm_or_endpoint", llm.OPENROUTER_DEFAULT_ENDPOINT),
            key=f"{p}llm_or_endpoint",
        )
        models = _openrouter_model_history(saved_llm)
        if not any(m == llm.OPENROUTER_DEFAULT_MODEL for m in models):
            models = [llm.OPENROUTER_DEFAULT_MODEL] + models
        options = models + ["(other — type below)"]
        sel = st.session_state.get(f"{p}llm_or_model_sel")
        if sel and sel not in options:
            options = [sel] + options
        st.selectbox(
            "Model",
            options,
            key=f"{p}llm_or_model_sel",
            help="Previously used models are remembered here. Pick "
                 "\"(other — type below)\" to type a new one.",
        )
        if st.session_state.get(f"{p}llm_or_model_sel") == "(other — type below)":
            st.text_input(
                "Custom model name",
                value=saved_llm.get(f"{p}llm_or_model", ""),
                key=f"{p}llm_or_model",
                help="e.g. openai/gpt-4o-mini, anthropic/claude-3.5-sonnet, "
                     "google/gemini-2.0-flash",
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
            help="Write-only: the key is never echoed back. "
                 "Leave empty to use the OPENROUTER_API_KEY env var.",
        )

    if show_instruction:
        st.text_input(
            "Instruction / prompt",
            value=saved_llm.get(f"{p}llm_instruction", llm.DEFAULT_INSTRUCTION),
            key=f"{p}llm_instruction",
        )
