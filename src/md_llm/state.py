"""Generic helpers shared across md_llm modules (file reading, display names,
sizes, and the autossh field/widget-key scaffolding).

Foundation module: other md_llm modules import from here; this one imports only
``.core`` (for path resolution via the injected Core) — no other md_llm module.
Kept free of any host-specific (transcript / YouTube / Whisper) concepts.
"""

from __future__ import annotations

import os
import re

from .core import get_core


# Proportional content zoom for the Reader and Chat panels. Unlike a flat
# font-size override (which flattens the heading/body hierarchy — every element
# becomes the same size), `zoom` scales headings, body text, and code blocks
# together, so the size differences within a document are preserved exactly and
# bold/font-weight styling is untouched. Scoped to Streamlit's markdown/code
# containers (``.stMarkdown`` / ``.stCodeBlock``); chat-message bodies render
# inside the same containers, so they scale too, while widget labels, buttons,
# and avatars keep their theme sizes. `zoom` is supported in all modern browsers
# (Chrome/Edge/Safari always, Firefox ≥ 126). Tunable: change this one number to
# make content larger (2.0 ≈ the earlier 28px body target) or smaller.
_CONTENT_ZOOM = 1.2

# Colour + weight for bold text (``**bold**`` / ``<strong>``). A distinct colour
# makes bold pop against the body — default-weight ``strong`` can look muted once
# the content is zoomed. Tunable: change this one value to recolour.
_BOLD_COLOR = "#d62828"

_BODY_FONT_SIZE_CSS = f"""
<style>
.stMarkdown, [data-testid="stMarkdownContainer"],
.stCodeBlock, [data-testid="stCodeBlock"] {{
  zoom: {_CONTENT_ZOOM};
}}
.stMarkdown strong, .stMarkdown b,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] b {{
  font-weight: 800;
  color: {_BOLD_COLOR};
}}
</style>
"""


def _human_size(nbytes):
    """Format a byte count as e.g. '1.2 KB' / '3.4 MB'."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


def _read_text(path):
    """Read a file's text as UTF-8. Returns '' for missing/unreadable files."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


# Code spans/fences are rendered literally by the markdown processor, so a
# backslash inside them would show up verbatim — never touch a $ there.
_CODE_BLOCK_OR_SPAN = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)

# A $ that is NOT already escaped and is immediately followed by a digit
# (optionally with commas/periods) — e.g. $2.5B, $1,500, $3.7B. Genuine math
# like $x^2$ or $\frac{a}{b}$ doesn't start with a bare digit, so it's safe.
_CURRENCY_DOLLAR = re.compile(r"(?<!\\)\$(?=\d[\d.,]*)")


def _escape_currency_dollars(text):
    """Escape ``$`` signs that look like currency, not LaTeX math.

    Streamlit's ``st.markdown`` parses ``$...$`` as KaTeX math. In prose a lone
    ``$`` almost always denotes a dollar amount (``$2.5B``, ``$1,500``); pairs of
    them get mis-paired into one huge math span that garbles the enclosed text
    (spaces dropped, letters treated as separate variables, ``**`` -> ``*``).
    We escape such ``$`` to ``\\$`` (which renders as a literal ``$``) — but only
    ``$`` followed by a digit, so genuine math (``$x^2$``, ``$\\frac{a}{b}$``)
    is untouched. Code spans/fences are skipped: their content is literal, so a
    backslash there would appear verbatim.
    """
    if not text or "$" not in text:
        return text
    out = []
    last = 0
    for m in _CODE_BLOCK_OR_SPAN.finditer(text):
        out.append(_CURRENCY_DOLLAR.sub(r"\\$", text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_CURRENCY_DOLLAR.sub(r"\\$", text[last:]))
    return "".join(out)


def _load_title_sidecar(path):
    """Read an optional ``<path>.meta.json`` sidecar's ``title`` field.

    A generic, optional display-name hook: if a host (or a previous save) wrote a
    ``{"title": "..."}`` sidecar beside a document, prefer that title. Returns ''
    when no sidecar exists or it's unreadable — callers then fall back to the
    file stem. md_llm itself never writes these sidecars (saved chats are plain
    .md), but this keeps display names friendly when a host provides them.
    """
    if not path:
        return ""
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    if not os.path.isfile(meta_path):
        return ""
    try:
        import json

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("title") or "").strip() if isinstance(data, dict) else ""
    except (OSError, ValueError, TypeError):
        return ""


def _display_name_for_filepath(path):
    """Human-readable name for a document path: sidecar title, else the stem.

    Generic version of transcriber_system's helper — no YouTube oEmbed lookup,
    no transcript/output split. Prefers an optional ``.meta.json`` sidecar title
    (a hook a host may populate), otherwise the bare filename stem.
    """
    title = _load_title_sidecar(path)
    if title:
        return title[:60]
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0][:60]


# ---------------------------------------------------------------------------
# autossh scaffolding (shared by .autossh): default configs + widget-key builder
# ---------------------------------------------------------------------------

# Host-neutral fallback for an LLM-server (Ollama on :11434) autossh tunnel.
# This only seeds settings.json on the FIRST run of a panel (seed-on-first-run,
# mirroring the llm panel pattern); after that the persisted per-panel subkey
# (e.g. ``chat_autossh``) is the source of truth and this constant only
# backfills any missing field. No host-specific address is baked in — a host
# injects its own server via settings.json (the host project does this via a
# one-time migration on startup).
DEFAULT_LLM_AUTOSSH = {
    "identity": "~/.ssh/id_ed25519",
    "local_port": 11434,
    "remote_host": "localhost",
    "remote_port": 11434,
    "ssh_host": "user@remote-host",
    "gatetime": 0,
    "monitor_port": 0,  # -M 0 disables the echo monitoring port
    "server_alive_interval": 1,
    "server_alive_count_max": 1,
    "extra_options": "",  # additional -o options beyond the ones above
}

# autossh field name -> widget-key suffix. The full widget key is built by
# _ssh_widget_key() as ``f"_{prefix}ssh_{suffix}"``; prefix="" yields the legacy
# keys (``_ssh_local_port``, …), while a namespaced panel (e.g. ``"chat_"``)
# gets its own keys (``_chat_ssh_*``) so several tunnels never collide.
_SSH_FIELD_SUFFIXES = {
    "local_port": "local_port",
    "remote_port": "remote_port",
    "remote_host": "remote_host",
    "ssh_host": "host",
    "identity": "identity",
    "monitor_port": "monitor",
    "gatetime": "gatetime",
    "server_alive_interval": "interval",
    "server_alive_count_max": "count",
    "extra_options": "extra",
}


def _ssh_widget_key(prefix, field):
    """Build the session-state widget key for one autossh field.

    ``prefix=""`` yields the legacy keys (``_ssh_local_port``, …); a prefixed
    panel (``"chat_"``) yields its own namespace (``_chat_ssh_local_port``, …) so
    multiple panels — one per tab — never collide even though Streamlit mounts
    every tab's widgets on every run.
    """
    return f"_{prefix}ssh_{_SSH_FIELD_SUFFIXES[field]}"
